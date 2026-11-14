"""Resumable frozen DINO multiview cache, episode-sharded across GPUs."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from .libero_data import metadata, pixels, load_dino, encode, sha, DINO_WEIGHTS, statistics

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--finalize', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    (demos, train, val, goals, tasks) = metadata()
    manifest = dict(version='libero_goal_h5_a5_v1', train=train, val=val, tasks=tasks, goals=goals, layers=[3, 7, 11], views=2, dtype='float32', weights_sha256=sha(DINO_WEIGHTS), goal_source='first seeded training demonstration terminal frame, per task', split='task-stratified 45 train / 5 validation trajectories, seed 42+task', action_sha256={str(i): sha(Path(demos[i]['path']) / 'actions.npy') for i in train + val}, normalization='RGB224; ImageNet; float32 DINO; cudnn_tf32=False', no_simulator_state_inputs=True)
    plan = out / 'preparation.json'
    if plan.exists():
        assert json.loads(plan.read_text()) == manifest
    elif args.rank == 0:
        plan.write_text(json.dumps(manifest, indent=2))
    if args.finalize:
        for i in train + val:
            arr = np.load(out / f'episode_{i:05d}.npy', mmap_mode='r')
            assert arr.shape == (demos[i]['length'], 2, 3, 256, 384)
        np.save(out / 'goals.npy', np.stack([np.load(out / f'episode_{i:05d}.npy', mmap_mode='r')[-1] for i in goals]))
        torch.save(statistics(train), out / 'stats.pt')
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print('CACHE_PASS', flush=True)
        return
    dino = load_dino('cuda')
    for (j, ep) in enumerate(train + val):
        if j % args.world_size != args.rank:
            continue
        n = demos[ep]['length']
        dest = out / f'episode_{ep:05d}.npy'
        if dest.exists():
            assert np.load(dest, mmap_mode='r').shape == (n, 2, 3, 256, 384)
            continue
        tmp = dest.with_suffix('.tmp.npy')
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(n, 2, 3, 256, 384))
        for view in range(2):
            x = pixels(ep, view)
            assert len(x) == n, (ep, view, len(x), n)
            for s in range(0, n, args.batch):
                arr[s:s + args.batch, view] = encode(dino, x[s:min(n, s + args.batch)]).cpu().numpy()
        arr.flush()
        del arr
        tmp.replace(dest)
        print('CACHED', ep, n, flush=True)
if __name__ == '__main__':
    main()
