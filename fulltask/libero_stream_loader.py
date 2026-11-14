"""Assemble unchanged logical batches on device with bounded host staging.

WAWindows returns copied history/future arrays, so its mmap references can be
dropped after each staging chunk. No workers, shared-memory queues, sampling,
augmentation, precision changes, or gradient accumulation are introduced.
"""
import torch
from torch.utils.data import BatchSampler, default_collate


class DeviceBatchLoader:
    def __init__(self, dataset, device, *, sampler=None, batch_size=None,
                 batch_sampler=None, host_chunk=8):
        assert host_chunk > 0
        assert hasattr(dataset, 'maps')
        if batch_sampler is None:
            assert sampler is not None and batch_size is not None
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
        self.dataset = dataset
        self.device = torch.device(device)
        self.batch_sampler = batch_sampler
        self.host_chunk = host_chunk

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        for indices in self.batch_sampler:
            count = len(indices)
            packed = None
            try:
                for start in range(0, count, self.host_chunk):
                    selected = indices[start:start + self.host_chunk]
                    part = default_collate([self.dataset[i] for i in selected])
                    # All returned feature tensors own their storage; mmap
                    # pages need not remain attached to this process.
                    self.dataset.maps.clear()
                    if packed is None:
                        packed = {key: torch.empty((count, *value.shape[1:]),
                            dtype=value.dtype, device=self.device)
                            for key, value in part.items()}
                    end = start + len(selected)
                    for key, value in part.items():
                        packed[key][start:end].copy_(value, non_blocking=False)
                    del part, value
                yield packed
            finally:
                # Also release references when an iterator is closed early.
                self.dataset.maps.clear()
                del packed
