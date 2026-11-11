"""Evaluate WA-JEPA on Push-T without CEM or online action search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
import hdf5plugin  # noqa: F401
import numpy as np
import stable_worldmodel as swm
import torch

from runtime_utils import episode_lengths, ChunkedPlanner
from wa_jepa import WorldActionJEPA
from wa_policy import WorldActionRecedingHorizonPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("LOCAL_DATASET_DIR", "data"),
    )
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--execute-blocks", type=int, default=5)
    parser.add_argument("--adaptive-horizon", action="store_true")
    parser.add_argument("--continue-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 1 <= args.execute_blocks <= 5:
        raise ValueError("--execute-blocks must be between 1 and 5")
    torch.set_float32_matmul_precision("high")
    model, payload = WorldActionJEPA.load_weights(args.checkpoint)
    if args.inference_batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    model = ChunkedPlanner(model, args.inference_batch_size)
    policy = WorldActionRecedingHorizonPolicy(
        model=model,
        execute_blocks=args.execute_blocks,
        device="cuda",
        adaptive_horizon=args.adaptive_horizon,
        continue_threshold=args.continue_threshold,
    )

    # The evaluator needs state only to recreate the dataset's initial Push-T
    # simulator state. State never enters the WA-JEPA model or policy.
    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train",
        keys_to_cache=["action", "proprio", "state"],
        cache_dir=args.dataset_root,
    )
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.unique(dataset.get_col_data(column))
    lengths = episode_lengths(dataset, episodes)
    max_start = lengths - args.goal_offset - 1
    maximum = {episode: max_start[index] for index, episode in enumerate(episodes)}
    row_maximum = np.asarray(
        [maximum[episode] for episode in dataset.get_col_data(column)]
    )
    valid = np.nonzero(dataset.get_col_data("step_idx") <= row_maximum)[0]
    generator = np.random.default_rng(args.seed)
    selected = generator.choice(len(valid) - 1, size=args.num_eval, replace=False)
    selected = np.sort(valid[selected])
    selected_rows = dataset.get_row_data(selected)
    eval_episodes = selected_rows[column]
    eval_starts = selected_rows["step_idx"]

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.eval_budget,
        image_shape=(224, 224),
    )
    world.set_policy(policy)
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=False)
    video_dir = None if args.no_video else result_dir / "videos"

    start = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_starts.tolist(),
        goal_offset=args.goal_offset,
        eval_budget=args.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=[
            {"method": "_set_state", "args": {"state": {"value": "state"}}},
            {
                "method": "_set_goal_state",
                "args": {"goal_state": {"value": "goal_state"}},
            },
        ],
        video=video_dir,
    )
    elapsed = time.perf_counter() - start
    world.close()
    block_counts = np.asarray(policy.executed_block_counts, dtype=np.float32)
    current_success = np.asarray(policy.current_success_probabilities, dtype=np.float32)
    plan_success = np.asarray(policy.plan_success_probabilities, dtype=np.float32)
    plan_distance = np.asarray(policy.plan_goal_distances, dtype=np.float32)
    serializable = {
        "architecture": "WA-JEPA",
        "selected_rows": selected.tolist(),
        "episodes": eval_episodes.tolist(),
        "start_steps": eval_starts.tolist(),
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_epoch": payload["epoch"],
        "success_rate": float(metrics["success_rate"]),
        "episode_successes": np.asarray(metrics["episode_successes"])
        .astype(bool)
        .tolist(),
        "num_eval": args.num_eval,
        "goal_offset": args.goal_offset,
        "eval_budget": args.eval_budget,
        "execute_blocks": args.execute_blocks,
        "adaptive_horizon": args.adaptive_horizon,
        "continue_threshold": args.continue_threshold,
        "mean_executed_blocks": (
            float(block_counts.mean()) if block_counts.size else None
        ),
        "model_forward_calls": policy.forward_calls,
        "internal_world_rollouts": (policy.forward_calls * model.cfg.refinement_steps),
        "evaluation_time_seconds": elapsed,
        "current_success_probability_mean": (
            float(current_success.mean()) if current_success.size else None
        ),
        "plan_success_probability_mean_by_block": (
            plan_success.mean(axis=0).astype(float).tolist()
            if plan_success.size
            else None
        ),
        "predicted_goal_distance_mean_by_block": (
            plan_distance.mean(axis=0).astype(float).tolist()
            if plan_distance.size
            else None
        ),
        "cem_used": False,
        "candidate_action_search_used": False,
        "gradient_action_optimization_used": False,
        "model_inputs": ["pixels", "goal_pixels", "proprio"],
        "state_used_by_model": False,
        "dino_model": model.cfg.dino_model,
        "dino_layers": list(model.cfg.dino_layers),
        "scene_tokens_per_frame": model.cfg.scene_tokens,
    }
    output = result_dir / "results.json"
    with output.open("w") as handle:
        json.dump(serializable, handle, indent=2)
    print(json.dumps(serializable, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
