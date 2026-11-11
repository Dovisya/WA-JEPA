"""CEM-free receding-horizon deployment policy for WA-JEPA."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import torch

from stable_worldmodel.policy import BasePolicy

from wa_jepa import WorldActionJEPA


class WorldActionRecedingHorizonPolicy(BasePolicy):
    """Plan 25 steps once and execute a configurable number of 5-step blocks."""

    def __init__(
        self,
        model: WorldActionJEPA,
        execute_blocks: int = 5,
        device: str = "cuda",
        adaptive_horizon: bool = False,
        continue_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 1 <= execute_blocks <= model.cfg.horizon:
            raise ValueError(
                f"execute_blocks must be in [1, {model.cfg.horizon}], "
                f"got {execute_blocks}"
            )
        self.type = "wa_jepa"
        self.model = model.to(device).eval().requires_grad_(False)
        self.execute_blocks = execute_blocks
        self.device = torch.device(device)
        self.adaptive_horizon = adaptive_horizon
        self.continue_threshold = continue_threshold
        self._action_buffers: list[deque[np.ndarray]] = []
        self._frame_history: list[deque[np.ndarray]] = []
        self._proprio_history: list[deque[np.ndarray]] = []
        self._step_count: np.ndarray | None = None
        self.forward_calls = 0
        self.executed_block_counts: list[int] = []
        self.current_success_probabilities: list[float] = []
        self.plan_success_probabilities: list[list[float]] = []
        self.plan_goal_distances: list[list[float]] = []

    def set_env(self, env: Any) -> None:
        self.env = env
        number = env.num_envs
        self._action_buffers = [deque() for _ in range(number)]
        self._frame_history = [
            deque(maxlen=self.model.cfg.history_size) for _ in range(number)
        ]
        self._proprio_history = [
            deque(maxlen=self.model.cfg.history_size) for _ in range(number)
        ]
        self._step_count = np.zeros(number, dtype=np.int64)

    @staticmethod
    def _latest_image(value: Any) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim == 4:
            return value[-1]
        if value.ndim == 3:
            return value
        raise ValueError(f"Unexpected image shape: {value.shape}")

    @staticmethod
    def _latest_proprio(value: Any) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim == 2:
            return value[-1]
        if value.ndim == 1:
            return value
        raise ValueError(f"Unexpected proprio shape: {value.shape}")

    def _reset_env(self, index: int) -> None:
        self._action_buffers[index].clear()
        self._frame_history[index].clear()
        self._proprio_history[index].clear()
        assert self._step_count is not None
        self._step_count[index] = 0

    def _append_observations(
        self,
        pixels: np.ndarray,
        proprio: np.ndarray,
        flush: np.ndarray,
    ) -> None:
        assert self._step_count is not None
        for index in range(len(self._action_buffers)):
            if flush[index]:
                self._reset_env(index)
            if self._step_count[index] % self.model.cfg.action_block == 0:
                self._frame_history[index].append(
                    self._latest_image(pixels[index]).copy()
                )
                self._proprio_history[index].append(
                    self._latest_proprio(proprio[index]).copy()
                )

    @staticmethod
    def _consecutive_blocks(
        probabilities: np.ndarray,
        threshold: float,
    ) -> int:
        count = 0
        for probability in probabilities:
            if probability < threshold:
                break
            count += 1
        return max(1, count)

    def _plan(
        self,
        indices: list[int],
        pixels: np.ndarray,
        proprio: np.ndarray,
        goal: np.ndarray,
    ) -> None:
        contexts = []
        proprio_contexts = []
        goals = []
        validity = []
        for index in indices:
            history = list(self._frame_history[index])
            proprio_history = list(self._proprio_history[index])
            if not history:
                history = [self._latest_image(pixels[index]).copy()]
                proprio_history = [self._latest_proprio(proprio[index]).copy()]
            history = history[-self.model.cfg.history_size :]
            proprio_history = proprio_history[-self.model.cfg.history_size :]
            missing = self.model.cfg.history_size - len(history)
            contexts.append(np.stack([history[0]] * missing + history))
            proprio_contexts.append(
                np.stack([proprio_history[0]] * missing + proprio_history)
            )
            validity.append([False] * missing + [True] * len(history))
            goals.append(self._latest_image(goal[index]))

        context_tensor = torch.from_numpy(np.stack(contexts)).permute(0, 1, 4, 2, 3)
        proprio_tensor = torch.from_numpy(np.stack(proprio_contexts).astype(np.float32))
        goal_tensor = torch.from_numpy(np.stack(goals)).permute(0, 3, 1, 2)
        validity_tensor = torch.as_tensor(validity, dtype=torch.bool)
        context_tensor = context_tensor.to(self.device, non_blocking=True)
        proprio_tensor = proprio_tensor.to(self.device, non_blocking=True)
        goal_tensor = goal_tensor.to(self.device, non_blocking=True)
        validity_tensor = validity_tensor.to(self.device, non_blocking=True)

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ),
        ):
            output = self.model.plan(
                context_tensor,
                proprio_tensor,
                goal_tensor,
                validity_tensor,
            )
        plan = (
            output["actions"]
            .float()
            .cpu()
            .reshape(
                len(indices),
                self.model.cfg.horizon,
                self.model.cfg.action_block,
                self.model.cfg.raw_action_dim,
            )
        )
        continue_probabilities = (
            output["continue_logits"].float().sigmoid().cpu().numpy()
        )
        plan_success = output["success_logits"].float().sigmoid().cpu().numpy()
        current_success = (
            output["current_success_logit"].float().sigmoid().cpu().numpy()
        )
        goal_distance = output["goal_distance"].float().cpu().numpy()

        low = np.asarray(self.env.single_action_space.low)
        high = np.asarray(self.env.single_action_space.high)
        for row, index in enumerate(indices):
            blocks = self.execute_blocks
            if self.adaptive_horizon:
                predicted = self._consecutive_blocks(
                    continue_probabilities[row], self.continue_threshold
                )
                blocks = min(blocks, predicted)
            actions = plan[row, :blocks].reshape(-1, self.model.cfg.raw_action_dim)
            actions = np.clip(actions.numpy(), low, high)
            self._action_buffers[index].extend(actions)
            self.executed_block_counts.append(blocks)
            self.current_success_probabilities.append(float(current_success[row]))
            self.plan_success_probabilities.append(
                plan_success[row].astype(float).tolist()
            )
            self.plan_goal_distances.append(goal_distance[row].astype(float).tolist())
        self.forward_calls += 1

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        if self.env is None:
            raise RuntimeError("Policy environment is not configured")
        if "proprio" not in info_dict:
            raise KeyError(
                "WA-JEPA requires robot proprioception in info_dict['proprio']"
            )
        pixels = np.asarray(info_dict["pixels"])
        proprio = np.asarray(info_dict["proprio"])
        goal = np.asarray(info_dict["goal"])
        number = self.env.num_envs
        flush = np.asarray(info_dict.get("_needs_flush", np.zeros(number)), dtype=bool)
        terminated = np.asarray(
            info_dict.get("terminated", np.zeros(number)), dtype=bool
        )
        if terminated.ndim > 1:
            terminated = terminated.reshape(number, -1)[:, -1]

        self._append_observations(pixels, proprio, flush)
        replan = [
            index
            for index in range(number)
            if not terminated[index] and not self._action_buffers[index]
        ]
        if replan:
            self._plan(replan, pixels, proprio, goal)

        actions = np.full(
            (number, self.model.cfg.raw_action_dim), np.nan, dtype=np.float32
        )
        assert self._step_count is not None
        for index in range(number):
            if not terminated[index] and self._action_buffers[index]:
                actions[index] = self._action_buffers[index].popleft()
                self._step_count[index] += 1
        return actions.reshape(self.env.action_space.shape)
