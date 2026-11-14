"""Push-T full-task DDP training and evaluation."""
import argparse, json, os, random, hashlib, math, time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .data import WAWindows, sha
from .eval_protocol import validation_batches, batch_rng
from .initialization import initialize
from .libero_runtime import release_eval_memory

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())

def set_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state(s['cuda'])

def move(batch, device):
    if isinstance(batch, dict):
        return {k: move(v, device) for (k, v) in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)((move(v, device) for v in batch))
    return batch.to(device, non_blocking=True)

def digest(model):
    h = hashlib.sha256()
    for p in model.parameters():
        if p.requires_grad:
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def grad_summary(model, variant):
    prefixes = ['encoder.layer_projections', 'planner.action_decoder.bins', 'planner.action_decoder.offsets', 'planner.world', 'planner.feedback.', 'planner.feedback_gates']
    out = {}
    for prefix in prefixes:
        grads = [p.grad for (n, p) in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads, prefix
        out[prefix] = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
        assert math.isfinite(out[prefix]) and out[prefix] > 0, prefix
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert all((p.grad is None for p in frozen))
    return out

def worker_init(worker_id):
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def loader_options(workers, seed):
    options = dict(num_workers=workers, pin_memory=True, generator=torch.Generator().manual_seed(seed))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=1, multiprocessing_context='spawn', worker_init_fn=worker_init)
    return options

@torch.no_grad()
def validate_parallel(model, variant, loader, batches, device, group):
    model.eval()
    local = []
    for ((batch_id, _), batch) in zip(batches, loader):
        batch = move(batch, device)
        with batch_rng(1042 + batch_id, device):
            loss = model(batch)['loss']
        local.append((batch_id, float(loss)))
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, local, group=group)
    rows = sorted([row for part in gathered for row in part])
    assert [i for (i, _) in rows] == list(range(len(rows))), 'Validation missing/duplicate batches'
    return float(np.mean([loss for (_, loss) in rows]))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--batch-per-gpu', type=int, default=256)
    p.add_argument('--epochs', type=int, default=400)
    p.add_argument('--steps-per-epoch', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--num-envs', type=int, default=10)
    p.add_argument('--final-evals', type=int, default=100)
    p.add_argument('--loader-workers', type=int, default=2)
    p.add_argument('--grad-check-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--init-checkpoint', help='Full checkpoint for a new fine-tuning schedule')
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--min-learning-rate', type=float, default=0.0)
    p.add_argument('--warmup-fraction', type=float, default=0.01)
    args = p.parse_args()
    assert 0 <= args.min_learning_rate <= args.learning_rate and args.learning_rate > 0
    assert 0 <= args.warmup_fraction < 1

    assert args.loader_workers >= 0 and args.grad_check_every > 0
    assert args.epochs > 0 and args.steps_per_epoch == 0 and (args.eval_episodes == 100) and (args.final_evals == 100) and (args.num_envs == 10)
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world in (1, 4), 'Supported launch sizes: one or four GPUs'
    control = dist.new_group(backend='gloo', timeout=timedelta(hours=2))
    cache = Path(args.cache)
    run = Path(args.run_dir)
    manifest = json.loads((cache / 'manifest.json').read_text())
    cb = torch.load(cache / 'codebook.pt', map_location='cpu', weights_only=False)
    assert cb['signature']['cache_sha256'] == sha(cache / 'manifest.json')
    assert cb['signature']['passes'] == 1000, 'Use the complete pretrained codebook'
    seed_all(args.seed)
    from .wa_model import WAJEPAVQ
    model = WAJEPAVQ(torch.load(cache / 'stats.pt', weights_only=True)).to(device)
    model.vq.load_state_dict(cb['model'])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.001)
    initial = initialize(model, opt, args.init_checkpoint if not args.resume else None, args.learning_rate)
    trainset = WAWindows(cache, 'train')
    valset = WAWindows(cache, 'val')
    sampler = DistributedSampler(trainset, num_replicas=world, rank=rank, seed=args.seed, shuffle=True, drop_last=False)
    loader = DataLoader(trainset, batch_size=args.batch_per_gpu, sampler=sampler, drop_last=False, **loader_options(args.loader_workers, 4200 + rank))
    val_batches = validation_batches(len(valset), args.batch_per_gpu, rank, world, 0)
    val_loader = DataLoader(valset, batch_sampler=[indices for (_, indices) in val_batches], **loader_options(args.loader_workers, 5200 + rank))
    nsteps = args.steps_per_epoch or len(loader)
    active_total = max(1, (args.epochs if args.init_checkpoint else max(1, args.epochs - 1)) * nsteps)
    warmup = int(args.warmup_fraction * active_total)

    def factor(s):
        return min(1.0, (s + 1) / warmup) if s < warmup else args.min_learning_rate / args.learning_rate + (1 - args.min_learning_rate / args.learning_rate) * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, active_total - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    config = vars(args).copy()
    config['variant'] = 'wa'
    config.pop('resume')
    config.pop('epochs')
    config.update(world_size=world, global_batch=args.batch_per_gpu * world, cache_sha256=sha(cache / 'manifest.json'), codebook_sha256=sha(cache / 'codebook.pt'), schedule_epochs=args.epochs, seed=args.seed, execute_steps=1, ensemble_depth=5, eval_world_size=world, eval_rng_protocol='batch_seed_v1')
    from .wa_config import experiment_config
    config.update(experiment_config())
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'latest.pt').exists() and (not args.resume):
            raise RuntimeError('Existing run: use --resume or a new directory')

    print('rank', rank, 'assets/model ready', flush=True)
    dist.barrier(group=control)
    start = 0
    global_step = initial["global_step"] if initial else 0
    active_step = 0
    best = -1.0
    snapshot = None
    if args.resume:
        snapshot = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
        assert snapshot['signature'] == signature, 'Resume configuration differs'
        model.load_state_dict(snapshot['model'])
        opt.load_state_dict(snapshot['optimizer'])
        if scheduler:
            scheduler.load_state_dict(snapshot['scheduler'])
        start = snapshot['epoch'] + 1
        global_step = snapshot['global_step']
        active_step = snapshot['active_step']
        best = snapshot['best']
    if rank == 0:
        (run / 'config.json').write_text(json.dumps(config, indent=2))
    feedback_proof = None
    dist.barrier(group=control)
    ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=False)
    seed_all(args.seed + rank)
    if initial and len(initial["rng"]) == world:
        set_rng(initial["rng"][rank])
    if snapshot:
        set_rng(snapshot['rng'][rank])
    resume_digest = digest(model) if snapshot else None
    if snapshot:
        assert resume_digest == snapshot['parameter_hash']
    records = []
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        model.train()
        grads = None
        loss_values = []
        torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        print('rank', rank, 'epoch', epoch, 'start', flush=True)
        if rank == 0:
            print('TRAIN_PHASE', 'initialization (zero-loss epoch)' if epoch == 0 and not args.init_checkpoint else 'learning', 'batches', nsteps, 'global_batch', args.batch_per_gpu * world, flush=True)
        for (step, batch) in enumerate(loader):
            if args.steps_per_epoch and step >= args.steps_per_epoch:
                break
            batch = move(batch, device)
            opt.zero_grad(set_to_none=True)
            out = ddp(batch)
            loss = out['loss']
            shapes = dict(actions=list(out['actions'].shape), future=list(out['future'].shape))
            assert torch.isfinite(loss), loss
            if epoch == 0 and not args.init_checkpoint:
                loss = loss * 0.0
            loss.backward()
            if (epoch > 0 or args.init_checkpoint) and (step == 0 or (step + 1) % args.grad_check_every == 0 or step + 1 == nsteps):
                grads = grad_summary(model, 'wa')
            opt.step()
            if scheduler and (epoch > 0 or args.init_checkpoint):
                scheduler.step()
            global_step += 1
            active_step += int(epoch > 0 or bool(args.init_checkpoint))
            loss_values.append(float(loss.detach()))
            if rank == 0 and (step == 0 or (step + 1) % 20 == 0 or step + 1 == nsteps):
                print(f"TRAIN epoch={epoch} batch={step + 1}/{nsteps} loss={loss_values[-1]:.5f} lr={opt.param_groups[0]['lr']:.3g} elapsed_s={time.perf_counter() - start_time:.1f}", flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        hashes = [None] * world
        dist.all_gather_object(hashes, digest(model), group=control)
        assert len(set(hashes)) == 1, 'DDP parameter divergence'
        rank_stats = [None] * world
        dist.all_gather_object(rank_stats, dict(rank=rank, loss=loss_values, grad_norms=grads, shapes=shapes, peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30, seconds=elapsed), group=control)
        del batch, out, loss
        release_eval_memory(model, opt)
        state = rng_state()
        feedback_grad = feedback_proof
        evaluation_start = time.perf_counter()
        if rank == 0:
            print('VALIDATION epoch', epoch, 'windows', len(valset), 'distributed_ranks', world, flush=True)
        validation = validate_parallel(model, 'wa', val_loader, val_batches, device, control)
        release_eval_memory(model, opt)
        from .evaluate import evaluate
        if rank == 0:
            print('ROLLOUT_EVAL epoch', epoch, 'episodes', args.eval_episodes, 'env_batch', args.num_envs, 'ranks', world, flush=True)
        goal = torch.from_numpy(np.load(cache / 'goal.npy'))
        eval_result = evaluate(model, 'wa', goal, episodes=args.eval_episodes, num_envs=args.num_envs, distributed=True, group=control)
        evaluation_seconds = time.perf_counter() - evaluation_start
        if rank == 0:
            improved = eval_result['mean_final_coverage'] > best or not (run / 'best.pt').exists()
            best = max(best, eval_result['mean_final_coverage'])
        set_rng(state)
        rngs = [None] * world
        dist.all_gather_object(rngs, rng_state(), group=control)
        if rank == 0:
            record = dict(epoch=epoch, global_step=global_step, active_step=active_step, ranks=rank_stats, parameter_hash=hashes[0], all_ranks_identical=True, validation_loss=validation, evaluation=eval_result, evaluation_seconds=evaluation_seconds, epoch_seconds=time.perf_counter() - start_time, final_bc_feedback_gradients=feedback_grad, resumed_from_hash=resume_digest)
            records.append(record)
            with open(run / 'metrics.jsonl', 'a') as f:
                f.write(json.dumps(record) + '\n')
            payload = dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict() if scheduler else None, epoch=epoch, global_step=global_step, active_step=active_step, best=best, rng=rngs, signature=signature, config=config, parameter_hash=hashes[0], vqvae_is_fit=True)
            torch.save(payload, run / 'latest.tmp.pt')
            os.replace(run / 'latest.tmp.pt', run / 'latest.pt')
            if improved:
                torch.save(dict(payload, evaluation=eval_result), run / 'best.tmp.pt')
                os.replace(run / 'best.tmp.pt', run / 'best.pt')
            print('EPOCH_RESULT', json.dumps(dict(epoch=epoch, train_seconds=elapsed, eval_seconds=evaluation_seconds, epoch_seconds=record['epoch_seconds'], train_loss=float(np.mean(loss_values)), validation_loss=validation, mean_final_coverage=eval_result['mean_final_coverage'], mean_max_coverage=eval_result['mean_max_coverage'], success_rate=eval_result['success_rate'], all_ranks_identical=True)), flush=True)
        dist.barrier(group=control)
    saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    opt.load_state_dict(saved['optimizer'])
    if scheduler:
        scheduler.load_state_dict(saved['scheduler'])
    assert digest(model) == saved['parameter_hash']
    final = None
    if args.final_evals:
        release_eval_memory(model, opt)
        from .evaluate import evaluate
        final = evaluate(model, 'wa', torch.from_numpy(np.load(cache / 'goal.npy')), args.final_evals, args.num_envs, distributed=True, group=control)
    if rank == 0:
        (run / 'result.json').write_text(json.dumps(dict(status='PASS', variant='wa', world_size=world, batch_per_gpu=args.batch_per_gpu, checkpoint_reload=True, resume=args.resume, first_epoch=start, last_epoch=args.epochs - 1, global_step=global_step, active_step=active_step, final_evaluation=final), indent=2))
    dist.barrier(group=control)
    dist.destroy_process_group()
if __name__ == '__main__':
    main()
