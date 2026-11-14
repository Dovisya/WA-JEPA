"""Rank-partitioned LIBERO Goal evaluation: 10 tasks x 10 seeded trials, 300 steps."""
from collections import deque
import time
import numpy as np
import torch
import torch.distributed as dist
from .libero_data import encode, metadata
from .eval_protocol import batch_rng

@torch.no_grad()
def evaluate(model, variant, goals, episodes=100, num_envs=5, seed=42, distributed=False, group=None, max_steps=300):
    from envs.libero.libero_env import LiberoEnv, GOAL_PREDICATES
    assert num_envs in (1, 5, 10)
    assert metadata()[4] == list(GOAL_PREDICATES), 'Task/goal order mismatch'
    device = next(model.parameters()).device
    rank = dist.get_rank(group) if distributed else 0
    world = dist.get_world_size(group) if distributed else 1
    dino = model.encoder.dino
    model.eval()
    dino.eval()
    rows = []
    started = time.perf_counter()
    for (batch_index, start) in enumerate(range(0, episodes, num_envs)):
        if batch_index % world != rank:
            continue
        indices = list(range(start, min(start + num_envs, episodes)))
        tasks = [i // 10 for i in indices]
        with batch_rng(seed + 10000 + batch_index, device):
            envs = []
            obs = []
            try:
                for (i, task) in zip(indices, tasks):
                    env = LiberoEnv()
                    env.seed(seed + i - 1)
                    envs.append(env)
                    obs.append(env.reset(goal_idx=task))
                count = len(indices)
                history = deque(maxlen=5)
                chunks = deque(maxlen=5)
                done = np.zeros(count, bool)
                success = np.zeros(count, bool)
                step_counts = np.zeros(count, int)
                steps = 0
                while not done.all() and steps < max_steps:
                    x = torch.as_tensor(np.asarray(obs), device=device)
                    features = encode(dino, x.flatten(0, 1)).reshape(count, 2, 3, 256, 384)
                    history.append(features)
                    h = list(history)
                    h = [h[0]] * (5 - len(h)) + h
                    batch = dict(history=torch.stack(h, 1), history_valid=torch.tensor([steps + i >= 0 for i in [-4, -3, -2, -1, 0]], device=device)[None].expand(count, -1), goal=goals[tasks].to(device))
                    chunk = model.policy(batch)[0]['raw_actions'][:, 0]
                    chunks.append(chunk.cpu().numpy())
                    action = np.stack([a[:, 0] for a in chunks]).mean(0).clip(-1, 1)
                    chunks = deque([np.concatenate([a[:, 1:], np.zeros_like(a[:, :1])], 1) for a in chunks], maxlen=5)
                    assert np.isfinite(action).all()
                    for (j, env) in enumerate(envs):
                        if done[j]:
                            continue
                        (obs[j], reward, terminal, info) = env.step(action[j])
                        success[j] |= bool(info['all_completions_ids'])
                        done[j] = terminal or success[j]
                        step_counts[j] += 1
                    steps += 1
                for (j, i) in enumerate(indices):
                    rows.append(dict(episode_index=i, task_id=tasks[j], task=metadata()[4][tasks[j]], seed=seed + i, steps=int(step_counts[j]), success=bool(success[j])))
                print('LIBERO_ROLLOUT', rank, batch_index, 'success', success.tolist(), flush=True)
            finally:
                for env in envs:
                    if env.env is not None:
                        env.close()
    if distributed:
        parts = [None] * world
        dist.all_gather_object(parts, rows, group=group)
    else:
        parts = [rows]
    rows = sorted([r for part in parts for r in part], key=lambda r: r['episode_index'])
    assert [r['episode_index'] for r in rows] == list(range(episodes))
    per_task = {name: float(np.mean([r['success'] for r in rows if r['task'] == name])) for name in metadata()[4] if any((r['task'] == name for r in rows))}
    if episodes == 100:
        assert all((sum((r['task_id'] == t for r in rows)) == 10 for t in range(10)))
    return dict(episodes=rows, success_rate=float(np.mean([r['success'] for r in rows])), per_task=per_task, eval_world_size=world, rng_protocol='libero_unique_seed_fixed_batch_v1', seconds=time.perf_counter() - started)
