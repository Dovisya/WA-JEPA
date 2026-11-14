"""Full PushT rollouts partitioned by original batches, with unchanged metrics."""
from collections import deque
import numpy as np
import torch
import torch.distributed as dist
import time
from .eval_protocol import batch_plan, assigned_batches, batch_rng, merge_results
from .data import encode, load_dino
from .wa_config import HISTORY_OFFSETS

def wa_history_batch(history, props, goal, steps):
    """Match training's consecutive history and start-of-episode padding."""
    h = list(history)
    pr = list(props)
    count = h[-1].shape[0]
    device = h[-1].device
    ix = [max(0, len(h) - 1 + offset) for offset in HISTORY_OFFSETS]
    return dict(history=torch.stack([h[i] for i in ix], 1), proprio=torch.stack([pr[i] for i in ix], 1), history_valid=torch.tensor([steps + offset >= 0 for offset in HISTORY_OFFSETS], device=device)[None].expand(count, -1), goal=goal.to(device)[None].expand(count, -1, -1, -1))

@torch.no_grad()
def evaluate(model, variant, goal, episodes=100, num_envs=10, seed=42, *, distributed=False, group=None):
    from envs.pusht import PushWrapper, PushTKeypointsEnv
    device = next(model.parameters()).device
    dino = model.encoder.dino
    model.eval()
    dino.eval()
    results = []
    rank = dist.get_rank(group) if distributed else 0
    world = dist.get_world_size(group) if distributed else 1
    plan = batch_plan(episodes, num_envs, seed)
    started = time.perf_counter()
    print('EVAL_ASSIGN rank', rank, 'batches', [b['batch_index'] for b in assigned_batches(plan, rank, world)], flush=True)
    for spec in assigned_batches(plan, rank, world):
        batch_index = spec['batch_index']
        seeds = spec['seeds']
        count = len(seeds)
        with batch_rng(seed + batch_index, device):
            envs = [PushWrapper(id='pusht', env=PushTKeypointsEnv()) for _ in seeds]
            obs = []
            previous = []
            for (env, s) in zip(envs, seeds):
                env.seed(s)
                obs.append(env.reset())
                previous.append(np.asarray(env.unwrapped.agent.position).copy())
            history = deque(maxlen=5)
            props = deque(maxlen=5)
            action_chunks = deque(maxlen=5)
            done = np.zeros(count, dtype=bool)
            maxima = np.zeros(count)
            finals = np.zeros(count)
            steps = 0
            while not done.all():
                pixels = torch.as_tensor(np.asarray(obs), device=device).float()
                if pixels.ndim == 5:
                    pixels = pixels[:, 0]
                features = encode(dino, pixels)
                history.append(features)
                xy = np.array([np.asarray(e.unwrapped.agent.position) for e in envs])
                props.append(torch.tensor(np.concatenate([xy, xy - np.array(previous)], -1), device=device, dtype=torch.float32))
                previous = xy.copy()
                batch = wa_history_batch(history, props, goal, steps)
                (out, _, _) = model.policy(batch)
                chunk = out['raw_actions'][:, 0]
                action_chunks.append(chunk.cpu().numpy())
                action = np.stack([a[:, 0] for a in action_chunks]).mean(0)
                action_chunks = deque([np.concatenate([a[:, 1:], np.zeros_like(a[:, :1])], 1) for a in action_chunks], maxlen=5)
                for (i, env) in enumerate(envs):
                    if not done[i]:
                        (obs[i], reward, done[i], info) = env.step(action[i])
                        coverage = float(info['final_coverage'])
                        maxima[i] = max(maxima[i], coverage)
                        finals[i] = coverage
                steps += 1
                assert steps <= 1000, 'Environment termination failed'
            for (i, env) in enumerate(envs):
                results.append(dict(episode_index=spec['episode_start'] + i, batch_index=batch_index, seed=seeds[i], steps=steps, max_coverage=float(maxima[i]), final_coverage=float(finals[i])))
                env.close()
            print('rollout_rank', rank, 'rollout_batch', batch_index, 'steps', steps, 'coverage', maxima.tolist(), flush=True)
    if distributed:
        parts = [None] * world
        dist.all_gather_object(parts, results, group=group)
    else:
        parts = [results]
    result = merge_results(parts, plan)
    result.update(eval_world_size=world, rng_protocol='batch_seed_v1', seconds=time.perf_counter() - started)
    return result
