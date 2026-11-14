"""Release transient training allocations before EGL; retain learned state and RNG."""
import gc
import hashlib
import json
import torch



def release_eval_memory(model, optimizer=None):
    # Caller must also drop its batch/output references. Neither this function
    # nor empty_cache changes model weights, optimizer moments or RNG state.
    device = next(model.parameters()).device
    before = dict(allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                  reserved_gib=torch.cuda.memory_reserved(device) / 2**30)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    model.planner.action_decoder.records.clear()
    gc.collect()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return dict(before=before, after=dict(
        allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
        reserved_gib=torch.cuda.memory_reserved(device) / 2**30))


def check_resume_compatibility(snapshot, config, signature):
    """Resume only an identical training configuration."""
    if snapshot['signature'] != signature:
        raise ValueError('Resume configuration or source changed; use a new run.')
    return None
