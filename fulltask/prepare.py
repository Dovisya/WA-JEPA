"""Prepare the complete Push-T frozen-feature cache."""
import argparse, json, os
from pathlib import Path
import numpy as np
import torch
from .data import *

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(args.cache)
    out.mkdir(parents=True, exist_ok=True)
    (lengths, acts, states, train, val) = metadata()
    manifest = dict(version=1, train=train, val=val, layers=[3, 7, 11], dtype='float32', weights_sha256=sha(DINO_WEIGHTS), normalization='ImageNet; uint8/255; original 224x224', source_hashes={str(i): sha(ROOT / 'obses' / f'episode_{i:03d}.pth') for i in train + val}, actions_sha256=sha(ROOT / 'abs_actions.pth'), states_sha256=sha(ROOT / 'states.pth'))
    old = out / 'manifest.json'
    if old.exists():
        assert json.loads(old.read_text()) == manifest, 'Cache provenance/scope mismatch'
        assert all(((out / f'episode_{i:03d}.npy').exists() for i in train + val))
        print('Cache already complete', flush=True)
        return
    dino = load_dino('cuda')
    for ep in train + val:
        n = int(lengths[ep])
        pixels = torch.load(ROOT / 'obses' / f'episode_{ep:03d}.pth', weights_only=False)[:n]
        tmp = out / f'episode_{ep:03d}.tmp.npy'
        arr = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=(n, 3, 256, 384))
        for s in range(0, n, 16):
            arr[s:s + 16] = encode(dino, pixels[s:s + 16]).cpu().numpy()
        arr.flush()
        del arr
        os.replace(tmp, out / f'episode_{ep:03d}.npy')
        print('cached', ep, n, flush=True)
    from envs.pusht import PushTKeypointsEnv
    env = PushTKeypointsEnv()
    env.seed(42)
    env.reset()
    env._set_state(np.array([96.0, 96.0, 256.0, 256.0, np.pi / 4]))
    goal = env.render(mode='rgb_array')
    assert goal.shape == (224, 224, 3), goal.shape
    np.save(out / 'goal.npy', encode(dino, torch.from_numpy(goal.copy())[None])[0].cpu().numpy())
    np.save(out / 'goal_image.npy', goal)
    env.close()
    torch.save(statistics(train), out / 'stats.pt')
    old.write_text(json.dumps(manifest, indent=2))
    print('PREPARE PASS', flush=True)
if __name__ == '__main__':
    main()
