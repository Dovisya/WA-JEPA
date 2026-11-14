"""Optional fine-tuning initialization; exact resume stays in the trainers."""
from pathlib import Path
import torch

def initialize(model, optimizer, checkpoint, learning_rate):
    if not checkpoint:
        return None
    saved = torch.load(Path(checkpoint), map_location='cpu', weights_only=False)
    if 'optimizer' not in saved:
        raise ValueError('Fine-tuning requires a full checkpoint with optimizer state.')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    for group in optimizer.param_groups:
        group['lr'] = learning_rate
        group['initial_lr'] = learning_rate
    return {key: saved[key] for key in ('epoch', 'global_step', 'rng')}
