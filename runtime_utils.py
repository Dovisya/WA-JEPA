"""Small dataset/evaluation helpers extracted from the short-horizon experiments."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset


class NoSIGReg(torch.nn.Module):
    """Zero regularization, preserving the training RNG consumption."""

    def forward(self, embeddings):
        torch.randn(embeddings.size(-1), 1024, device=embeddings.device)
        return embeddings.new_zeros(())


def atomic_json(path, content):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(content, indent=2, allow_nan=False))
    temp.replace(path)


def split_by_episode(dataset, val_fraction: float, seed: int):
    if not hasattr(dataset, "clip_indices"):
        raise TypeError(
            "Episode-level split requires dataset.clip_indices; "
            f"got {type(dataset).__name__}"
        )
    clip_episodes = np.fromiter(
        (index[0] for index in dataset.clip_indices),
        dtype=np.int64,
        count=len(dataset.clip_indices),
    )
    episodes = np.unique(clip_episodes)
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(episodes)
    val_count = max(1, int(round(len(shuffled) * val_fraction)))
    val_episodes = shuffled[:val_count]
    is_val = np.isin(clip_episodes, val_episodes)
    train_indices = np.flatnonzero(~is_val)
    val_indices = np.flatnonzero(is_val)
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        len(episodes) - val_count,
        val_count,
    )


def episode_lengths(dataset, episodes: np.ndarray) -> np.ndarray:
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_index = dataset.get_col_data(column)
    step_index = dataset.get_col_data("step_idx")
    return np.asarray(
        [np.max(step_index[episode_index == episode]) + 1 for episode in episodes]
    )


class ChunkedPlanner(torch.nn.Module):
    """Chunk network inference only; environment vectorization stays unchanged."""

    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.batch_size = batch_size

    def plan(self, context_pixels, context_proprio, goal_pixels, history_valid=None):
        chunks = []
        for start in range(0, len(context_pixels), self.batch_size):
            end = start + self.batch_size
            chunks.append(
                self.model.plan(
                    context_pixels[start:end],
                    context_proprio[start:end],
                    goal_pixels[start:end],
                    None if history_valid is None else history_valid[start:end],
                )
            )
        return {
            key: torch.cat(
                [chunk[key] for chunk in chunks],
                dim=1 if key.startswith("intermediate_") else 0,
            )
            for key in chunks[0]
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
