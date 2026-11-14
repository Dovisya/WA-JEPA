"""FP32 visual encoding shared by cache construction and LIBERO rollout."""
import torch
from .data import encode as _encode

@torch.no_grad()
def encode(dino, pixels):
    with torch.backends.cudnn.flags(enabled=True, benchmark=False,
                                   deterministic=False, allow_tf32=False):
        return _encode(dino, pixels)
