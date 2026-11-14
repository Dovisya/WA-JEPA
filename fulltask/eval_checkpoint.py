"""Evaluate either full-task VQ-BeT checkpoint, without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .train import seed_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['pusht', 'libero_goal'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cache = Path(args.cache)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    stats = torch.load(cache / 'stats.pt', map_location='cpu', weights_only=True)
    if args.task == 'pusht':
        from .wa_model import WAJEPAVQ
        from .evaluate import evaluate
        model = WAJEPAVQ(stats)
        goals = torch.from_numpy(np.load(cache / 'goal.npy'))
        num_envs = 10
    else:
        from .libero_model import LiberoWA
        from .libero_evaluate import evaluate
        model = LiberoWA(stats)
        goals = torch.from_numpy(np.load(cache / 'goals.npy'))
        num_envs = 5
    model.load_state_dict(saved['model'], strict=True)
    model.cuda().eval()
    seed_all(args.seed)
    result = evaluate(model, 'wa', goals, episodes=100, num_envs=num_envs, seed=args.seed)
    result.update(task=args.task, checkpoint=str(Path(args.checkpoint).resolve()),
                  completed_epoch=saved.get('epoch', -1) + 1)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'episodes'}, indent=2))

if __name__ == '__main__':
    main()
