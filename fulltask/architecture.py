"""Shared WA-JEPA architecture used by the full-task adaptations."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


@dataclass
class WAJEPAConfig:
    image_size: int = 224
    embed_dim: int = 192
    history_size: int = 3
    horizon: int = 5
    action_block: int = 5
    raw_action_dim: int = 2
    proprio_dim: int = 4
    action_low: float = -1.0
    action_high: float = 1.0
    refinement_steps: int = 4
    world_depth: int = 3
    planner_heads: int = 6
    planner_mlp_dim: int = 768
    dropout: float = 0.1
    dino_model: str = "dinov2_vits14"
    dino_dim: int = 384
    dino_layers: tuple[int, ...] = (3, 7, 11)
    dino_repo: str = os.environ.get("DINO_REPO", "third_party/dinov2")
    dino_checkpoint: str = os.environ.get("DINO_CHECKPOINT", "pretrained/dinov2_vits14_pretrain.pth")
    scene_tokens: int = 8
    scene_pool_depth: int = 1
    frame_fusion_depth: int = 1
    temporal_fusion_depth: int = 2
    goal_match_depth: int = 1
    freeze_dino: bool = True

    @property
    def action_dim(self) -> int:
        return self.action_block * self.raw_action_dim

    @property
    def tokens_per_context_frame(self) -> int:
        return self.scene_tokens + 1


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryUpdate(nn.Module):
    """Residual self-attention, cross-attention, and feed-forward update."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized = self.self_norm(query)
        query = query + self.self_attn(
            normalized,
            normalized,
            normalized,
            attn_mask=self_mask,
            need_weights=False,
        )[0]
        normalized = self.cross_norm(query)
        memory = self.memory_norm(memory)
        query = query + self.cross_attn(
            normalized,
            memory,
            memory,
            attn_mask=cross_mask,
            need_weights=False,
        )[0]
        return query + self.ff(query)


class MultiLayerDINOSceneEncoder(nn.Module):
    """Fuse aligned intermediate DINOv2 patches into ordered scene slots."""

    def __init__(self, cfg: WAJEPAConfig) -> None:
        super().__init__()
        repo = Path(cfg.dino_repo)
        checkpoint = Path(cfg.dino_checkpoint)
        if not repo.is_dir():
            raise FileNotFoundError(f"Local DINOv2 repository not found: {repo}")
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Pretrained DINOv2 checkpoint not found: {checkpoint}"
            )
        self.cfg = cfg
        self.dino = torch.hub.load(
            str(repo), cfg.dino_model, source="local", pretrained=False
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.dino.load_state_dict(state, strict=True)
        if getattr(self.dino, "embed_dim", None) != cfg.dino_dim:
            raise ValueError(
                f"DINO dimension is {getattr(self.dino, 'embed_dim', None)}, "
                f"expected {cfg.dino_dim}"
            )
        if cfg.freeze_dino:
            self.dino.requires_grad_(False)

        self.layer_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.dino_dim),
                    nn.Linear(cfg.dino_dim, cfg.embed_dim),
                )
                for _ in cfg.dino_layers
            ]
        )
        self.layer_logits = nn.Parameter(torch.zeros(len(cfg.dino_layers)))
        self.scene_queries = nn.Parameter(
            torch.randn(1, cfg.scene_tokens, cfg.embed_dim) * 0.02
        )
        self.scene_pool = nn.ModuleList(
            [
                QueryUpdate(
                    cfg.embed_dim,
                    cfg.planner_heads,
                    cfg.planner_mlp_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.scene_pool_depth)
            ]
        )
        self.scene_norm = nn.LayerNorm(cfg.embed_dim)
        self.global_query = nn.Parameter(
            torch.randn(1, 1, cfg.embed_dim) * 0.02
        )
        self.global_pool = QueryUpdate(
            cfg.embed_dim,
            cfg.planner_heads,
            cfg.planner_mlp_dim,
            cfg.dropout,
        )
        self.global_norm = nn.LayerNorm(cfg.embed_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cfg.freeze_dino:
            self.dino.eval()
        return self

    def _intermediate(self, pixels: torch.Tensor) -> Sequence[torch.Tensor]:
        arguments = {
            "n": self.cfg.dino_layers,
            "reshape": False,
            "return_class_token": False,
            "norm": True,
        }
        if self.cfg.freeze_dino:
            with torch.no_grad():
                return self.dino.get_intermediate_layers(pixels, **arguments)
        return self.dino.get_intermediate_layers(pixels, **arguments)

    def forward(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self._intermediate(pixels)
        weights = self.layer_logits.softmax(dim=0)
        patches = sum(
            weight * projection(layer)
            for weight, projection, layer in zip(
                weights, self.layer_projections, layers
            )
        )
        scene = self.scene_queries.expand(patches.shape[0], -1, -1)
        for layer in self.scene_pool:
            scene = layer(scene, patches)
        scene = self.scene_norm(scene)
        global_embedding = self.global_pool(
            self.global_query.expand(patches.shape[0], -1, -1), scene
        )
        return scene, self.global_norm(global_embedding[:, 0])


class DenseWorldUpdateLayer(nn.Module):
    """Causally update every scene slot using context and executable actions."""

    def __init__(self, cfg: WAJEPAConfig) -> None:
        super().__init__()
        dim = cfg.embed_dim
        self.update = QueryUpdate(
            dim, cfg.planner_heads, cfg.planner_mlp_dim, cfg.dropout
        )
        self.dynamics = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, cfg.planner_mlp_dim),
            nn.GELU(),
            nn.Linear(cfg.planner_mlp_dim, dim),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        future: torch.Tensor,
        context_memory: torch.Tensor,
        action_features: torch.Tensor,
        current_scene: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, slots, dim = future.shape
        flat = future.reshape(batch, horizon * slots, dim)
        time_ids = torch.arange(horizon, device=future.device).repeat_interleave(
            slots
        )
        causal = time_ids[None, :] > time_ids[:, None]
        memory = torch.cat([context_memory, action_features], dim=1)
        cross_mask = torch.zeros(
            horizon * slots,
            context_memory.shape[1] + horizon,
            dtype=torch.bool,
            device=future.device,
        )
        action_ids = torch.arange(horizon, device=future.device)
        cross_mask[:, context_memory.shape[1] :] = (
            action_ids[None, :] > time_ids[:, None]
        )
        flat = self.update(
            flat,
            memory,
            self_mask=causal,
            cross_mask=cross_mask,
        )
        future = flat.reshape(batch, horizon, slots, dim)
        action = action_features[:, :, None, :].expand(-1, -1, slots, -1)
        current = current_scene[:, None].expand(-1, horizon, -1, -1)
        delta = self.dynamics(torch.cat([future, action, current], dim=-1))
        return future + torch.sigmoid(self.gate) * delta


class DenseExecutableActionWorldModel(nn.Module):
    """Predict dense future scene slots and normalized future proprio."""

    def __init__(self, cfg: WAJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.future_queries = nn.Parameter(
            torch.randn(1, cfg.horizon, cfg.scene_tokens, dim) * 0.02
        )
        self.time_pos = nn.Parameter(
            torch.randn(1, cfg.horizon, 1, dim) * 0.02
        )
        self.slot_pos = nn.Parameter(
            torch.randn(1, 1, cfg.scene_tokens, dim) * 0.02
        )
        self.action_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.action_encoder = nn.Sequential(
            nn.Linear(cfg.action_dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )
        self.layers = nn.ModuleList(
            [DenseWorldUpdateLayer(cfg) for _ in range(cfg.world_depth)]
        )
        self.scene_norm = nn.LayerNorm(dim)
        self.scene_head = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.proprio_head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, cfg.proprio_dim),
        )

    def forward(
        self,
        context_memory: torch.Tensor,
        current_scene: torch.Tensor,
        executable_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (
            context_memory.shape[0],
            self.cfg.horizon,
            self.cfg.action_dim,
        )
        if tuple(executable_actions.shape) != expected:
            raise ValueError(
                f"Expected actions {expected}, got {tuple(executable_actions.shape)}"
            )
        batch = context_memory.shape[0]
        action_features = (
            self.action_encoder(executable_actions) + self.action_type
        )
        future = (
            self.future_queries.expand(batch, -1, -1, -1)
            + self.time_pos
            + self.slot_pos
            + current_scene[:, None]
        )
        for layer in self.layers:
            future = layer(
                future, context_memory, action_features, current_scene
            )
        future = self.scene_head(self.scene_norm(future))
        pooled = future.mean(dim=2)
        future_proprio = self.proprio_head(
            torch.cat([pooled, action_features], dim=-1)
        )
        return future, future_proprio


class SceneGoalEvaluator(nn.Module):
    """Score completion, progress, and distance from dense future/goal slots."""

    def __init__(self, cfg: WAJEPAConfig) -> None:
        super().__init__()
        dim = cfg.embed_dim
        self.future_norm = nn.LayerNorm(dim)
        self.goal_norm = nn.LayerNorm(dim)
        self.goal_attention = nn.MultiheadAttention(
            dim,
            cfg.planner_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.features = nn.Sequential(
            nn.LayerNorm(4 * dim),
            nn.Linear(4 * dim, cfg.planner_mlp_dim),
            nn.GELU(),
            nn.Linear(cfg.planner_mlp_dim, dim),
            nn.GELU(),
        )
        self.success = nn.Linear(dim, 1)
        self.progress = nn.Linear(dim, 1)
        self.distance = nn.Linear(dim, 1)

    def forward(
        self,
        future_scene: torch.Tensor,
        goal_scene: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, horizon, slots, dim = future_scene.shape
        future = future_scene.reshape(batch * horizon, slots, dim)
        goal = goal_scene[:, None].expand(-1, horizon, -1, -1)
        goal = goal.reshape(batch * horizon, slots, dim)
        attended = self.goal_attention(
            self.future_norm(future),
            self.goal_norm(goal),
            self.goal_norm(goal),
            need_weights=False,
        )[0]
        future_pool = future.mean(dim=1)
        goal_pool = goal.mean(dim=1)
        attended_pool = attended.mean(dim=1)
        delta = goal_pool - future_pool
        features = self.features(
            torch.cat(
                [future_pool, attended_pool, delta, delta.abs()], dim=-1
            )
        )
        return {
            "success_logits": self.success(features).reshape(batch, horizon),
            "progress_logits": self.progress(features).reshape(batch, horizon),
            "goal_distance": F.softplus(
                self.distance(features).reshape(batch, horizon)
            ),
        }


class WorldActionPlanner(nn.Module):
    """Fixed-depth dense world/action interaction without online search."""

    def __init__(self, cfg: WAJEPAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.embed_dim
        self.action_queries = nn.Parameter(
            torch.randn(1, cfg.horizon, dim) * 0.02
        )
        self.plan_pos = nn.Parameter(
            torch.randn(1, cfg.horizon, dim) * 0.02
        )
        self.action_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.future_time = nn.Parameter(
            torch.randn(1, cfg.horizon, 1, dim) * 0.02
        )
        self.goal_condition = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.proposal_layers = nn.ModuleList(
            [
                QueryUpdate(
                    dim,
                    cfg.planner_heads,
                    cfg.planner_mlp_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.refinement_steps)
            ]
        )
        self.action_norm = nn.LayerNorm(dim)
        self.action_decoder = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, cfg.action_dim),
        )
        self.world = DenseExecutableActionWorldModel(cfg)
        self.future_proprio_encoder = nn.Sequential(
            nn.Linear(cfg.proprio_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.feedback = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(4 * dim),
                    nn.Linear(4 * dim, cfg.planner_mlp_dim),
                    nn.GELU(),
                    nn.Linear(cfg.planner_mlp_dim, dim),
                )
                for _ in range(max(0, cfg.refinement_steps - 1))
            ]
        )
        self.feedback_gates = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(-2.0))
                for _ in range(max(0, cfg.refinement_steps - 1))
            ]
        )
        self.evaluator = SceneGoalEvaluator(cfg)

    def bound_actions(self, actions: torch.Tensor) -> torch.Tensor:
        midpoint = 0.5 * (self.cfg.action_high + self.cfg.action_low)
        half_range = 0.5 * (self.cfg.action_high - self.cfg.action_low)
        return midpoint + half_range * torch.tanh(actions)

    def rollout_actions(
        self,
        context_memory: torch.Tensor,
        current_scene: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.world(context_memory, current_scene, actions)

    def forward(
        self,
        context_memory: torch.Tensor,
        current_scene: torch.Tensor,
        goal_scene: torch.Tensor,
        goal_memory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch = context_memory.shape[0]
        current_pool = current_scene.mean(dim=1)
        goal_pool = goal_scene.mean(dim=1)
        goal_delta = self.goal_condition(
            torch.cat([current_pool, goal_pool - current_pool], dim=-1)
        )[:, None]
        action_tokens = (
            self.action_queries.expand(batch, -1, -1)
            + self.plan_pos
            + self.action_type
            + goal_delta
        )
        future_scene = current_scene[:, None].expand(
            -1, self.cfg.horizon, -1, -1
        )
        future_proprio = current_pool.new_zeros(
            batch, self.cfg.horizon, self.cfg.proprio_dim
        )
        intermediate_actions = []
        intermediate_futures = []
        intermediate_proprio = []
        goal_expanded = goal_pool[:, None].expand(-1, self.cfg.horizon, -1)

        for index, proposal in enumerate(self.proposal_layers):
            future_memory = future_scene + self.future_time
            memory = torch.cat(
                [
                    context_memory,
                    goal_memory,
                    future_memory.flatten(1, 2),
                ],
                dim=1,
            )
            action_tokens = proposal(action_tokens, memory)
            actions = self.bound_actions(
                self.action_decoder(self.action_norm(action_tokens))
            )
            future_scene, future_proprio = self.world(
                context_memory, current_scene, actions
            )
            intermediate_actions.append(actions)
            intermediate_futures.append(future_scene)
            intermediate_proprio.append(future_proprio)
            if index < len(self.feedback):
                future_pool = future_scene.mean(dim=2)
                proprio_feature = self.future_proprio_encoder(future_proprio)
                correction = self.feedback[index](
                    torch.cat(
                        [
                            action_tokens,
                            future_pool,
                            goal_expanded - future_pool,
                            proprio_feature,
                        ],
                        dim=-1,
                    )
                )
                action_tokens = action_tokens + torch.sigmoid(
                    self.feedback_gates[index]
                ) * correction

        scores = self.evaluator(future_scene, goal_scene)
        current_scores = self.evaluator(
            current_scene[:, None], goal_scene
        )
        return {
            "actions": intermediate_actions[-1],
            "future_scene": future_scene,
            "future_proprio": future_proprio,
            "intermediate_actions": torch.stack(intermediate_actions),
            "intermediate_future_scene": torch.stack(intermediate_futures),
            "intermediate_future_proprio": torch.stack(intermediate_proprio),
            "success_logits": scores["success_logits"],
            "progress_logits": scores["progress_logits"],
            "goal_distance": scores["goal_distance"],
            "continue_logits": -scores["success_logits"],
            "current_success_logit": current_scores["success_logits"][:, 0],
            "action_tokens": action_tokens,
        }


class WorldActionJEPA(nn.Module):
    """Full  model used by training and receding-horizon deployment."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        if "dino_layers" in kwargs:
            kwargs["dino_layers"] = tuple(kwargs["dino_layers"])
        self.cfg = WAJEPAConfig(**kwargs)
        cfg = self.cfg
        dim = cfg.embed_dim
        self.encoder = MultiLayerDINOSceneEncoder(cfg)
        self.visual_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.proprio_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.goal_type = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.context_time = nn.Parameter(
            torch.randn(1, cfg.history_size, 1, dim) * 0.02
        )
        self.history_valid_embedding = nn.Embedding(2, dim)
        self.proprio_encoder = nn.Sequential(
            nn.Linear(cfg.proprio_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.frame_fusion = nn.ModuleList(
            [self._transformer_layer(cfg) for _ in range(cfg.frame_fusion_depth)]
        )
        self.temporal_fusion = nn.ModuleList(
            [
                self._transformer_layer(cfg)
                for _ in range(cfg.temporal_fusion_depth)
            ]
        )
        self.fusion_norm = nn.LayerNorm(dim)
        self.current_to_goal = nn.ModuleList(
            [
                QueryUpdate(
                    dim,
                    cfg.planner_heads,
                    cfg.planner_mlp_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.goal_match_depth)
            ]
        )
        self.goal_to_current = nn.ModuleList(
            [
                QueryUpdate(
                    dim,
                    cfg.planner_heads,
                    cfg.planner_mlp_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.goal_match_depth)
            ]
        )
        self.planner = WorldActionPlanner(cfg)
        self.inverse_head = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, cfg.action_dim),
        )
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]))
        self.register_buffer("action_mean", torch.zeros(cfg.raw_action_dim))
        self.register_buffer("action_std", torch.ones(cfg.raw_action_dim))
        self.register_buffer("proprio_mean", torch.zeros(cfg.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(cfg.proprio_dim))

    @staticmethod
    def _transformer_layer(cfg: WAJEPAConfig) -> nn.Module:
        return nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.planner_heads,
            dim_feedforward=cfg.planner_mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    @property
    def model_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_mean.copy_(mean.reshape_as(self.action_mean))
        self.action_std.copy_(std.reshape_as(self.action_std).clamp_min(1e-6))

    def set_proprio_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.proprio_mean.copy_(mean.reshape_as(self.proprio_mean))
        self.proprio_std.copy_(std.reshape_as(self.proprio_std).clamp_min(1e-6))

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return (
            torch.nan_to_num(proprio.float(), nan=0.0)
            - self.proprio_mean.view(1, 1, -1)
        ) / self.proprio_std.view(1, 1, -1)

    def preprocess_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        if pixels.numel() and pixels.detach().amax() > 2.0:
            pixels = pixels / 255.0
        mean = self.pixel_mean.view(1, 1, 3, 1, 1)
        std = self.pixel_std.view(1, 1, 3, 1, 1)
        return (pixels - mean) / std

    def encode_pixels(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if pixels.ndim != 5:
            raise ValueError(
                f"Expected pixels [B,T,C,H,W], got {tuple(pixels.shape)}"
            )
        pixels = self.preprocess_pixels(pixels)
        batch, steps = pixels.shape[:2]
        flat = rearrange(pixels, "b t c h w -> (b t) c h w")
        scene, global_embedding = self.encoder(flat)
        scene = rearrange(scene, "(b t) n d -> b t n d", b=batch, t=steps)
        global_embedding = rearrange(
            global_embedding, "(b t) d -> b t d", b=batch, t=steps
        )
        return scene, global_embedding

    def fuse_context(
        self,
        scene: torch.Tensor,
        proprio: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, slots, dim = scene.shape
        if steps != self.cfg.history_size:
            raise ValueError(f"Expected {self.cfg.history_size} context frames")
        normalized = self.normalize_proprio(proprio)
        proprio_tokens = self.proprio_encoder(normalized).unsqueeze(2)
        valid = self.history_valid_embedding(history_valid.long()).unsqueeze(2)
        visual = (
            scene
            + self.visual_type.view(1, 1, 1, -1)
            + self.context_time
            + valid
        )
        proprio_tokens = (
            proprio_tokens
            + self.proprio_type.view(1, 1, 1, -1)
            + self.context_time
            + valid
        )
        fused = torch.cat([visual, proprio_tokens], dim=2)
        tokens = fused.shape[2]
        fused = fused.reshape(batch * steps, tokens, dim)
        for layer in self.frame_fusion:
            fused = layer(fused)
        fused = fused.reshape(batch, steps * tokens, dim)
        for layer in self.temporal_fusion:
            fused = layer(fused)
        fused = self.fusion_norm(fused)
        return fused, fused.reshape(batch, steps, tokens, dim)

    def match_goal(
        self,
        context_by_time: torch.Tensor,
        goal_scene: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = context_by_time[:, -1]
        goal = goal_scene + self.goal_type
        for current_layer, goal_layer in zip(
            self.current_to_goal, self.goal_to_current
        ):
            previous_current = current
            current = current_layer(current, goal)
            goal = goal_layer(goal, previous_current)
        updated = context_by_time.clone()
        updated[:, -1] = current
        return updated.flatten(1, 2), goal

    @staticmethod
    def _gather_time(
        values: torch.Tensor, one_based_steps: torch.Tensor
    ) -> torch.Tensor:
        batch = torch.arange(values.shape[0], device=values.device)
        return values[batch, one_based_steps - 1]

    def forward_train(
        self,
        pixels: torch.Tensor,
        proprio: torch.Tensor,
        teacher_actions: torch.Tensor,
        history_valid: torch.Tensor,
        goal_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        expected_steps = self.cfg.history_size + self.cfg.horizon
        if pixels.shape[1] != expected_steps:
            raise ValueError(
                f"Expected {expected_steps} frames, got {pixels.shape[1]}"
            )
        scene, embeddings = self.encode_pixels(pixels)
        context_scene = scene[:, : self.cfg.history_size]
        target_scene = scene[:, self.cfg.history_size :]
        if goal_steps is None:
            goal_steps = torch.full(
                (pixels.shape[0],),
                self.cfg.horizon,
                dtype=torch.long,
                device=pixels.device,
            )
        goal_steps = goal_steps.long()
        if goal_steps.shape != (pixels.shape[0],):
            raise ValueError(
                f"Expected goal_steps [{pixels.shape[0]}], "
                f"got {tuple(goal_steps.shape)}"
            )
        if (goal_steps < 1).any() or (goal_steps > self.cfg.horizon).any():
            raise ValueError(
                f"goal_steps must be in [1, {self.cfg.horizon}]"
            )
        context_memory, context_by_time = self.fuse_context(
            context_scene,
            proprio[:, : self.cfg.history_size],
            history_valid,
        )
        goal_scene = self._gather_time(target_scene, goal_steps)
        context_memory, goal_memory = self.match_goal(
            context_by_time, goal_scene.detach()
        )
        actions = torch.nan_to_num(
            teacher_actions.float(), nan=0.0
        ).clamp(self.cfg.action_low, self.cfg.action_high)
        policy = self.planner(
            context_memory,
            context_scene[:, -1],
            goal_scene.detach(),
            goal_memory,
        )
        teacher_scene, teacher_proprio = self.planner.rollout_actions(
            context_memory, context_scene[:, -1], actions
        )
        negative_actions = actions.roll(1, dims=0) if actions.shape[0] > 1 else -actions
        negative_scene, negative_proprio = self.planner.rollout_actions(
            context_memory, context_scene[:, -1], negative_actions
        )
        target_scores = self.planner.evaluator(
            target_scene.detach(), goal_scene.detach()
        )
        teacher_scores = self.planner.evaluator(
            teacher_scene, goal_scene.detach()
        )
        negative_scores = self.planner.evaluator(
            negative_scene, goal_scene.detach()
        )
        pooled_targets = embeddings[:, self.cfg.history_size :]
        pooled_context = embeddings[:, : self.cfg.history_size]
        previous = torch.cat(
            [pooled_context[:, -1:], pooled_targets[:, :-1]], dim=1
        )
        inverse_actions = self.planner.bound_actions(
            self.inverse_head(torch.cat([previous, pooled_targets], dim=-1))
        )
        policy.update(
            {
                "embeddings": embeddings,
                "context_scene": context_scene,
                "target_future_scene": target_scene,
                "goal_scene": goal_scene,
                "goal_steps": goal_steps,
                "target_actions": actions,
                "target_future_proprio": self.normalize_proprio(
                    proprio[:, self.cfg.history_size :]
                ),
                "teacher_future_scene": teacher_scene,
                "teacher_future_proprio": teacher_proprio,
                "negative_future_scene": negative_scene,
                "negative_future_proprio": negative_proprio,
                "target_scores": target_scores,
                "teacher_scores": teacher_scores,
                "negative_scores": negative_scores,
                "inverse_actions": inverse_actions,
            }
        )
        return policy

    @torch.inference_mode()
    def plan(
        self,
        context_pixels: torch.Tensor,
        context_proprio: torch.Tensor,
        goal_pixels: torch.Tensor,
        history_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if goal_pixels.ndim == 4:
            goal_pixels = goal_pixels[:, None]
        if history_valid is None:
            history_valid = torch.ones(
                context_pixels.shape[:2],
                dtype=torch.bool,
                device=context_pixels.device,
            )
        context_pixels = context_pixels[:, -self.cfg.history_size :]
        context_proprio = context_proprio[:, -self.cfg.history_size :]
        history_valid = history_valid[:, -self.cfg.history_size :]
        pixels = torch.cat([context_pixels, goal_pixels[:, -1:]], dim=1)
        scene, _ = self.encode_pixels(pixels)
        context_scene = scene[:, : self.cfg.history_size]
        goal_scene = scene[:, -1]
        context_memory, context_by_time = self.fuse_context(
            context_scene, context_proprio, history_valid
        )
        context_memory, goal_memory = self.match_goal(
            context_by_time, goal_scene
        )
        return self.planner(
            context_memory,
            context_scene[:, -1],
            goal_scene,
            goal_memory,
        )

    def save_weights(
        self,
        path: str | Path,
        *,
        epoch: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "wa_jepa_",
            "epoch": int(epoch),
            "model_config": self.model_config,
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_weights(
        cls,
        path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> tuple["WorldActionJEPA", dict[str, Any]]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != "wa_jepa_":
            raise ValueError(f"Unsupported WA-JEPA  checkpoint: {path}")
        model = cls(**payload["model_config"])
        model.load_state_dict(payload["model_state_dict"], strict=True)
        return model, payload
