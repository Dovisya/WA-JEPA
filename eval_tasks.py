"""Short-horizon TwoRoom/Cube evaluation with the benchmark case sampling protocol.

No model/training changes, CEM, search, or action-gradient optimization.
Cube's seven robot-proprio fields are aliased to their dataset names before
World installs the dataset start state; they are refreshed on every env step.
"""

import argparse
import json
import os
from pathlib import Path
import re
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")
import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from omegaconf import OmegaConf
import torch

from runtime_utils import ChunkedPlanner, sha256_file
from wa_jepa_v4 import WorldActionJEPAV4
from wa_policy_v4 import WorldActionRecedingHorizonPolicyV4


class CubeProprioAliases(gym.Wrapper):
    """Match HDF5 naming, including World's first-step dataset state injection."""

    @staticmethod
    def alias(info):
        return {
            **info,
            **{
                k.replace("/", "_", 1): v
                for k, v in info.items()
                if k.startswith("proprio/")
            },
        }

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, self.alias(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self.alias(info)


class TaskPolicy(WorldActionRecedingHorizonPolicyV4):
    def __init__(self, *args, proprio_columns, expected_initial_proprio, **kwargs):
        super().__init__(*args, **kwargs)
        self.proprio_columns = proprio_columns
        self.expected_initial_proprio = expected_initial_proprio
        self.checked_initial = False

    def get_action(self, info_dict, **kwargs):
        # Never read object state, goal proprio, qpos/qvel or observation28.
        parts = [np.asarray(info_dict[k]) for k in self.proprio_columns]
        proprio = np.concatenate(parts, axis=-1)
        if (
            proprio.shape[-1] != self.model.cfg.proprio_dim
            or not np.isfinite(proprio).all()
        ):
            raise ValueError("Invalid robot proprioception")
        if not self.checked_initial:
            np.testing.assert_allclose(
                proprio[:, -1], self.expected_initial_proprio, rtol=1e-6, atol=1e-6
            )
            self.checked_initial = True
        return super().get_action({**info_dict, "proprio": proprio}, **kwargs)


def baseline_cases(source, cfg):
    with h5py.File(source, "r") as f:
        ep_key = "episode_idx" if "episode_idx" in f else "ep_idx"
        episodes = np.asarray(f[ep_key]).reshape(-1)
        steps = np.asarray(f["step_idx"]).reshape(-1)
        unique, inverse = np.unique(episodes, return_inverse=True)
        last_step = np.zeros(len(unique), dtype=np.int64)
        np.maximum.at(last_step, inverse, steps)
        valid = np.flatnonzero(steps <= last_step[inverse] - cfg.eval.goal_offset_steps)
        # Keep the baseline's len(valid)-1 sampling convention exactly.
        rows = np.sort(
            valid[
                np.random.default_rng(cfg.seed).choice(
                    len(valid) - 1, size=cfg.eval.num_eval, replace=False
                )
            ]
        )
        cases = [
            dict(row=int(r), episode=int(episodes[r]), start=int(steps[r]))
            for r in rows
        ]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["tworoom", "cube"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--cuda-memory-fraction", type=float, default=0)
    parser.add_argument(
        "--dataset-root", default=os.environ.get("LOCAL_DATASET_DIR", "data")
    )
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument(
        "--allow-smoke", action="store_true", help="For integration checks only"
    )
    args = parser.parse_args()
    if args.inference_batch_size < 1 or not 0 <= args.cuda_memory_fraction <= 1:
        parser.error("Invalid inference resource limits")
    if args.cuda_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    model, payload = WorldActionJEPAV4.load_weights(args.checkpoint)
    provenance = payload["training_provenance"]
    if (payload.get("is_smoke_test") and not args.allow_smoke) or provenance[
        "task"
    ] != args.task:
        raise ValueError("Smoke-test checkpoint or task mismatch")
    assert model.cfg.horizon == 5 and model.cfg.refinement_steps == 4
    relative = (
        "tworoom.h5" if args.task == "tworoom" else "ogbench/cube_single_expert.h5"
    )
    source = Path(args.dataset_root) / "datasets" / relative
    config_path = Path(__file__).parent / "config" / "eval" / f"{args.task}.yaml"
    cfg = OmegaConf.load(config_path)
    cfg.eval.num_eval = args.num_eval
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    cases = baseline_cases(source, cfg)
    columns = provenance["proprio_columns"]
    assert all(k == "proprio" or k.startswith("proprio_") for k in columns)
    with h5py.File(source, "r") as f:
        expected = np.concatenate(
            [np.asarray(f[k][[c["row"] for c in cases]]) for k in columns], axis=-1
        )
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        task=args.task,
        seed=int(cfg.seed),
        cases=cases,
        dataset=str(source),
        dataset_size=source.stat().st_size,
        dataset_mtime_ns=source.stat().st_mtime_ns,
        baseline_eval_config=str(config_path),
        baseline_eval_config_sha256=sha256_file(config_path),
        goal_offset=int(cfg.eval.goal_offset_steps),
        eval_budget=int(cfg.eval.eval_budget),
        execute_blocks=5,
        action_block=5,
        adaptive_horizon=False,
        env_reset_seed="dataset seed if present, else None; same as baseline",
        world=OmegaConf.to_container(cfg.world, resolve=True),
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
    )
    (root / "manifest.json").write_text(json.dumps(protocol, indent=2))
    model = ChunkedPlanner(
        model.cuda().eval().requires_grad_(False), args.inference_batch_size
    ).eval()
    policy = TaskPolicy(
        model,
        execute_blocks=5,
        device="cuda",
        adaptive_horizon=False,
        proprio_columns=columns,
        expected_initial_proprio=expected,
    )
    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(path=source)

    def physical_success(world):
        if args.task == "cube":
            return [
                bool(np.all(env.unwrapped._compute_successes()))
                for env in world.envs.envs
            ]
        return [
            bool(
                torch.norm(env.unwrapped.agent_position - env.unwrapped.target_position)
                < 16.0
            )
            for env in world.envs.envs
        ]

    class AuditedWorld(swm.World):
        def _get_actions(self):
            if not hasattr(self, "initial_successes"):
                self.initial_successes = physical_success(self)
            return super()._get_actions()

    world = AuditedWorld(
        **protocol["world"],
        image_shape=(224, 224),
        pre_wrappers=[CubeProprioAliases] if args.task == "cube" else [],
    )
    world.set_policy(policy)
    start = time.perf_counter()
    try:
        metrics = world.evaluate(
            dataset=dataset,
            episodes_idx=[c["episode"] for c in cases],
            start_steps=[c["start"] for c in cases],
            goal_offset=protocol["goal_offset"],
            eval_budget=protocol["eval_budget"],
            callables=protocol["callables"],
        )
        final_successes = physical_success(world)
        np.testing.assert_array_equal(final_successes, metrics["episode_successes"])
    finally:
        # MuJoCo renderers otherwise defer cleanup until interpreter shutdown.
        if args.task == "cube":
            for env in world.envs.envs:
                renderer = getattr(env.unwrapped, "_renderer", None)
                if renderer is not None and hasattr(renderer, "close"):
                    renderer.close()
        world.close()
    result = dict(
        is_smoke_test=bool(payload.get("is_smoke_test")),
        task=args.task,
        epoch=int(payload["epoch"]),
        format=payload["format"],
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=sha256_file(args.checkpoint),
        protocol=protocol,
        manifest_sha256=sha256_file(root / "manifest.json"),
        inference_batch_size=args.inference_batch_size,
        precision="bf16-mixed",
        proprio_columns=columns,
        initial_proprio_verified=policy.checked_initial,
        initial_physical_successes=world.initial_successes,
        final_physical_successes=final_successes,
        executed_steps=policy._step_count.tolist(),
        planner_forward_calls=policy.forward_calls,
        privileged_state_used_by_model=False,
        planner="original V4 direct policy, no CEM/search",
        success_rate=float(metrics["success_rate"]),
        episode_successes=np.asarray(metrics["episode_successes"])
        .astype(bool)
        .tolist(),
        seconds=time.perf_counter() - start,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    (root / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(
        f"{args.task} E{payload['epoch']}: {sum(result['episode_successes'])}/{len(cases)} = {result['success_rate']:.4f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
