"""Matched LIBERO Goal policy training, multi-rank evaluation and full-state resume."""
import argparse, hashlib, json, math, os, time, resource
from pathlib import Path
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .libero_data import WAWindows, sha, ROOT
from .train import seed_all, rng_state, set_rng, move, digest, grad_summary, worker_init
from .eval_protocol import validation_batches, batch_rng
from .initialization import initialize
from .libero_runtime import release_eval_memory, check_resume_compatibility
from .libero_stream_loader import DeviceBatchLoader

def atomic_save(payload, path):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(payload, tmp)
    tmp.replace(path)

def optimizer_summary(opt):
    parts = getattr(opt, 'optimizers', [opt])
    states = [s for part in parts for s in part.state.values()]
    return dict(entries=len(states), steps=sorted(set((float(s['step']) for s in states if 'step' in s))))

def keep_top3(run, payload):
    e = payload['evaluation']
    path = run / f"best_epoch_{payload['epoch'] + 1:04d}.pt"
    index = run / 'top3.json'
    old = json.loads(index.read_text()) if index.exists() else []
    row = dict(epoch=payload['epoch'] + 1, success_rate=e['success_rate'], file=path.name)
    assert all((x['epoch'] != row['epoch'] for x in old))
    ranked = sorted(old + [row], key=lambda x: (-x['success_rate'], x['epoch']))
    if row in ranked[:3]:
        atomic_save(payload, path)
    tmp = run / 'best.tmp.link'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(ranked[0]['file'])
    tmp.replace(run / 'best.pt')
    tmp = run / 'top3.tmp.json'
    tmp.write_text(json.dumps(ranked[:3], indent=2))
    tmp.replace(index)
    for x in ranked[3:]:
        file = run / x['file']
        assert file.parent == run and file.name.startswith('best_epoch_')
        file.unlink(missing_ok=True)

@torch.no_grad()
def validate(model, variant, loader, batches, device, group):
    model.eval()
    rows = []
    for ((idx, _), batch) in zip(batches, loader):
        with batch_rng(1042 + idx, device):
            batch = move(batch, device)
            loss = model(batch)['loss']
        assert torch.isfinite(loss)
        rows.append((idx, float(loss)))
    parts = [None] * dist.get_world_size(group)
    dist.all_gather_object(parts, rows, group=group)
    rows = sorted((x for part in parts for x in part))
    assert [x[0] for x in rows] == list(range(len(rows)))
    return float(np.mean([x[1] for x in rows]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--steps-per-epoch', type=int, default=0)
    p.add_argument('--batch-per-gpu', type=int, default=256)
    p.add_argument('--loader-workers', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=1)
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--num-envs', type=int, default=5)
    p.add_argument('--final-evals', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-checkpoint', help='Full checkpoint for a new fine-tuning schedule')
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--min-learning-rate', type=float, default=0.0)
    p.add_argument('--warmup-fraction', type=float, default=0.01)
    args = p.parse_args()
    assert 0 <= args.min_learning_rate <= args.learning_rate and args.learning_rate > 0
    assert 0 <= args.warmup_fraction < 1

    assert args.epochs > 0 and args.steps_per_epoch == 0 and (args.eval_episodes == 100) and (args.final_evals == 100) and (args.num_envs == 5)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible[0]:
        os.environ['MUJOCO_EGL_DEVICE_ID'] = visible[local]
    dist.init_process_group('nccl', timeout=timedelta(hours=4), device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world in [1, 4, 8]
    control = dist.new_group(backend='gloo', timeout=timedelta(hours=4))
    cache = Path(args.cache)
    run = Path(args.run_dir)
    manifest = json.loads((cache / 'manifest.json').read_text())
    cb = torch.load(cache / 'codebook.pt', map_location='cpu', weights_only=False)
    assert cb['signature']['cache_sha256'] == sha(cache / 'manifest.json')
    assert cb['signature']['passes'] == 1000, 'Use the complete pretrained codebook'
    seed_all(args.seed)
    from .libero_model import LiberoWA
    model = LiberoWA(torch.load(cache / 'stats.pt', weights_only=True)).to(device)
    model.vq.load_state_dict(cb['model'])
    opt = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=args.learning_rate, weight_decay=0.001)
    initial = initialize(model, opt, args.init_checkpoint if not args.resume else None, args.learning_rate)
    trainset = WAWindows(cache, 'train')
    valset = WAWindows(cache, 'val')
    sampler = DistributedSampler(trainset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True, drop_last=False)
    assert args.loader_workers == 0, 'Streaming loader uses no worker processes'
    loader = DeviceBatchLoader(trainset, device, sampler=sampler, batch_size=args.batch_per_gpu)
    batches = validation_batches(len(valset), 16, rank, world, 0)
    val_loader = DeviceBatchLoader(valset, device, batch_sampler=[idx for (_, idx) in batches])
    total_steps = (args.steps_per_epoch or len(loader)) * args.epochs
    warmup = int(total_steps * args.warmup_fraction)

    def factor(s):
        return min(1.0, (s + 1) / warmup) if s < warmup else args.min_learning_rate / args.learning_rate + (1 - args.min_learning_rate / args.learning_rate) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total_steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    config = vars(args).copy()
    config['variant'] = 'wa'
    config.pop('resume')
    config.pop('epochs')
    config.update(data_delivery='stream_to_device_chunk8_v1', schedule_epochs=args.epochs, world_size=world, global_batch=args.batch_per_gpu * world, source_root=str(ROOT), views=2, history_offsets=[-4, -3, -2, -1, 0], action_loss='100*L1+5*focal1+0.5*focal2', scene_tokens=8, refinement_stages=4, world_loss_weight=1.0, cache_sha256=sha(cache / 'manifest.json'), codebook_sha256=sha(cache / 'codebook.pt'), source_sha256={n: sha(Path(__file__).parent / n) for n in [Path(__file__).name, 'libero_model.py', 'libero_data.py', 'libero_evaluate.py', 'architecture.py', 'wa_model.py', 'eval_protocol.py', 'libero_runtime.py', 'libero_stream_loader.py']}, lr=args.learning_rate, weight_decay=0.001, lr_schedule='warmup_cosine', min_lr=args.min_learning_rate, warmup_fraction=args.warmup_fraction, history=5, action_block=5, action_dim=7, execute_steps=1, ensemble_depth=5, metric='success_rate', metric_range=[0, 1], no_simulator_state_inputs=True, checkpoint_retention='top3_full_plus_latest', rng_protocol='libero_unique_seed_fixed_batch_v1')
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        assert args.resume or not (run / 'latest.pt').exists(), 'Existing run requires --resume'
    dist.barrier(group=control)
    start = 0
    global_step = initial["global_step"] if initial else 0
    best = -1.0
    snapshot = None
    if args.resume:
        snapshot = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        compatibility = check_resume_compatibility(snapshot, config, signature)
        if rank == 0 and compatibility:
            (run / f'resume_runtime_patch_{int(time.time())}.json').write_text(json.dumps(compatibility, indent=2))
        model.load_state_dict(snapshot['model'])
        opt.load_state_dict(snapshot['optimizer'])
        scheduler.load_state_dict(snapshot['scheduler'])
        start = snapshot['epoch'] + 1
        global_step = snapshot['global_step']
        best = snapshot['best']
        assert optimizer_summary(opt)['entries'] > 0 and optimizer_summary(opt)['steps'] == [float(global_step)]
    assert start < args.epochs
    if rank == 0:
        (run / 'config.json').write_text(json.dumps(config, indent=2))
    print('INITIAL_STATE', json.dumps(dict(rank=rank, completed_epoch=start, next_epoch=start + 1, model_hash=digest(model), optimizer=optimizer_summary(opt))), flush=True)
    feedback = None
    dist.barrier(group=control)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    seed_all(args.seed + rank)
    if initial and len(initial["rng"]) == world:
        set_rng(initial["rng"][rank])
    if snapshot:
        set_rng(snapshot['rng'][rank])
        assert digest(model) == snapshot['parameter_hash']
        del snapshot
        snapshot = None
    from .libero_evaluate import evaluate
    goal = torch.from_numpy(np.load(cache / 'goals.npy'))
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        losses = []
        component_sums = {}
        grads = None
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        limit = args.steps_per_epoch or len(loader)
        for (step, batch) in enumerate(loader):
            if step >= limit:
                break
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            out = ddp(batch)
            (pred, loss) = (out['actions'], out['loss'])
            actions_shape = list(pred.shape)
            assert torch.isfinite(loss)
            for (k, v) in out['metrics'].items():
                component_sums[k] = component_sums.get(k, 0) + v.detach()
            loss.backward()
            if step == 0 or step + 1 == limit:
                grads = grad_summary(model, 'wa')
            opt.step()
            scheduler.step()
            global_step += 1
            losses.append(float(loss.detach()))
            if rank == 0 and (step == 0 or (step + 1) % 20 == 0):
                print('TRAIN', epoch + 1, step + 1, limit, losses[-1], flush=True)
            del batch, out, pred, loss
        hashes = [None] * world
        dist.all_gather_object(hashes, digest(model), group=control)
        assert len(set(hashes)) == 1
        rank_stats = [None] * world
        dist.all_gather_object(rank_stats, dict(rank=rank, loss=float(np.mean(losses)), grad_norms=grads, components={k: float(v / len(losses)) for (k, v) in component_sums.items()}, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, host_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, actions_shape=actions_shape), group=control)
        train_seconds = time.perf_counter() - t0
        state = rng_state()
        evaluation = None
        val = None
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            released = release_eval_memory(model, opt)
            print('EVAL_MEMORY_RELEASE', epoch + 1, rank, json.dumps(released), flush=True)
            val = validate(model, 'wa', val_loader, batches, device, control)
            release_eval_memory(model, opt)
            evaluation = evaluate(model, 'wa', goal, args.eval_episodes, args.num_envs, 42, True, control, max_steps=300)
            best = max(best, evaluation['success_rate'])
            release_eval_memory(model, opt)
        set_rng(state)
        rngs = [None] * world
        dist.all_gather_object(rngs, rng_state(), group=control)
        if rank == 0:
            record = dict(epoch=epoch, lr=opt.param_groups[0]['lr'], train_loss=float(np.mean([x['loss'] for x in rank_stats])), components={k: float(np.mean([x['components'][k] for x in rank_stats])) for k in component_sums}, validation_loss=val, evaluation=evaluation, ranks=rank_stats, all_ranks_identical=True, parameter_hash=hashes[0], feedback_gradients=feedback, train_seconds=train_seconds, epoch_seconds=time.perf_counter() - t0)
            with (run / 'metrics.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(), rng=rngs, epoch=epoch, global_step=global_step, best=best, signature=signature, config=config, parameter_hash=hashes[0], evaluation=evaluation)
            atomic_save(payload, run / 'latest.pt')
            if evaluation:
                keep_top3(run, payload)
            print('EPOCH_RESULT', json.dumps({k: v for (k, v) in record.items() if k not in ['evaluation', 'ranks']}), flush=True)
        dist.barrier(group=control)
    c = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(c['model'])
    opt.load_state_dict(c['optimizer'])
    assert digest(model) == c['parameter_hash']
    assert optimizer_summary(opt)['entries'] > 0 and optimizer_summary(opt)['steps'] == [float(global_step)]
    release_eval_memory(model, opt)
    final = evaluate(model, 'wa', goal, args.final_evals, args.num_envs, 42, True, control, max_steps=300) if args.final_evals else None
    if rank == 0:
        (run / 'result.json').write_text(json.dumps(dict(status='PASS', variant='wa', world_size=world, first_epoch=start + 1, last_epoch=args.epochs, checkpoint_reload=True, optimizer_states=optimizer_summary(opt)['entries'], optimizer_steps=optimizer_summary(opt)['steps'], rng_ranks=len(c['rng']), final_evaluation=final), indent=2))
    dist.barrier(group=control)
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
