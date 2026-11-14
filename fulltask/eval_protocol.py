"""Rank-independent evaluation assignment and RNG streams."""
from contextlib import contextmanager
import random
import numpy as np
import torch


def batch_plan(episodes, num_envs, seed=42):
    assert episodes > 0 and num_envs > 0
    return [dict(batch_index=i, episode_start=start,
                 seeds=[seed + max(0, i - 1) * num_envs + j
                        for j in range(min(num_envs, episodes - start))])
            for i, start in enumerate(range(0, episodes, num_envs))]


def assigned_batches(plan, rank, world):
    assert 0 <= rank < world
    return [batch for batch in plan if batch['batch_index'] % world == rank]


def validation_batches(length, batch_size, rank, world, limit=0):
    # Preserve the serial batch boundaries and mean-of-batch-means metric.
    batches = [list(range(start, min(start + batch_size, length)))
               for start in range(0, length, batch_size)]
    if limit:
        batches = batches[:limit]
    return [(i, indices) for i, indices in enumerate(batches) if i % world == rank]


@contextmanager
def batch_rng(seed, device):
    """Sampling is reproducible regardless of which rank owns a batch."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            # Seed only this process's CPU and current GPU RNG, not all GPUs.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def merge_results(parts, plan):
    results = sorted([row for part in parts for row in part], key=lambda row: row['episode_index'])
    expected = [(batch['episode_start'] + j, seed)
                for batch in plan for j, seed in enumerate(batch['seeds'])]
    assert [(row['episode_index'], row['seed']) for row in results] == expected, 'Missing/duplicate evaluation episodes'
    return dict(episodes=results,
                mean_max_coverage=float(np.mean([row['max_coverage'] for row in results])),
                mean_final_coverage=float(np.mean([row['final_coverage'] for row in results])),
                success_rate=float(np.mean([row['max_coverage'] >= .95 for row in results])))
