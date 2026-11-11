"""Train WA-JEPA on the fixed 25-step Push-T goal task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import lightning as pl
import numpy as np
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from runtime_utils import NoSIGReg, split_by_episode
from wa_jepa import WorldActionJEPA


class WAJEPALitModule(pl.LightningModule):
    def __init__(
        self,
        model: WorldActionJEPA,
        run_dir: str,
        learning_rate: float,
        representation_lr_scale: float,
        weight_decay: float,
        sigreg_weight: float,
        sigreg_reference_batch_size: int,
        history_dropout_prob: float,
        full_goal_probability: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.run_dir = Path(run_dir)
        self.learning_rate = learning_rate
        self.representation_lr_scale = representation_lr_scale
        self.weight_decay = weight_decay
        self.sigreg_weight = sigreg_weight
        self.sigreg_reference_batch_size = sigreg_reference_batch_size
        self.history_dropout_prob = history_dropout_prob
        self.full_goal_probability = full_goal_probability
        if sigreg_weight != 0:
            raise ValueError("WA-JEPA does not use SIGReg: its weight must be zero")
        self.sigreg = NoSIGReg()
        self.is_smoke_test = False
        self.save_hyperparameters(ignore=["model"])

    def augment_history(
        self,
        pixels: torch.Tensor,
        proprio: torch.Tensor,
        stage: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = pixels.shape[0]
        history = self.model.cfg.history_size
        valid = torch.ones(batch, history, dtype=torch.bool, device=pixels.device)
        if stage != "train" or self.history_dropout_prob <= 0:
            return pixels, proprio, valid
        draw = torch.rand(batch, device=pixels.device)
        one_frame = draw < 0.5 * self.history_dropout_prob
        two_frames = (draw >= 0.5 * self.history_dropout_prob) & (
            draw < self.history_dropout_prob
        )
        if not (one_frame.any() or two_frames.any()):
            return pixels, proprio, valid
        pixels = pixels.clone()
        proprio = proprio.clone()
        if one_frame.any():
            pixels[one_frame, :history] = pixels[
                one_frame, history - 1 : history
            ].expand(-1, history, -1, -1, -1)
            proprio[one_frame, :history] = proprio[
                one_frame, history - 1 : history
            ].expand(-1, history, -1)
            valid[one_frame, : history - 1] = False
        if two_frames.any():
            pixels[two_frames, 0] = pixels[two_frames, 1]
            proprio[two_frames, 0] = proprio[two_frames, 1]
            valid[two_frames, 0] = False
        return pixels, proprio, valid

    def compute_losses(self, batch: dict, stage: str) -> dict[str, torch.Tensor]:
        cfg = self.model.cfg
        pixels, proprio, history_valid = self.augment_history(
            batch["pixels"], batch["proprio"], stage
        )
        actions = batch["action"][
            :,
            cfg.history_size - 1 : cfg.history_size - 1 + cfg.horizon,
        ]
        if stage == "train":
            goal_steps = torch.randint(
                1,
                cfg.horizon + 1,
                (pixels.shape[0],),
                device=pixels.device,
            )
            keep_full_horizon = (
                torch.rand(pixels.shape[0], device=pixels.device)
                < self.full_goal_probability
            )
            goal_steps[keep_full_horizon] = cfg.horizon
        else:
            goal_steps = torch.full(
                (pixels.shape[0],),
                cfg.horizon,
                dtype=torch.long,
                device=pixels.device,
            )
        output = self.model.forward_train(
            pixels, proprio, actions, history_valid, goal_steps
        )

        time_ids = torch.arange(1, cfg.horizon + 1, device=pixels.device)[None]
        before_goal = time_ids <= goal_steps[:, None]

        def masked_mean(values: torch.Tensor) -> torch.Tensor:
            mask = before_goal
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(values)
            return values.masked_select(mask).mean()

        def masked_smooth_l1(
            prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return masked_mean(F.smooth_l1_loss(prediction, target, reduction="none"))

        target_scene = output["target_future_scene"].detach().float()
        goal_scene = output["goal_scene"].detach().float()
        policy_scene = output["future_scene"].float()
        teacher_scene = output["teacher_future_scene"].float()
        negative_scene = output["negative_future_scene"].float()
        target_proprio = output["target_future_proprio"].detach().float()
        policy_proprio = output["future_proprio"].float()
        teacher_proprio = output["teacher_future_proprio"].float()

        action_mean = (
            self.model.action_mean.float().repeat(cfg.action_block).view(1, 1, -1)
        )
        action_std = (
            self.model.action_std.float()
            .repeat(cfg.action_block)
            .view(1, 1, -1)
            .clamp_min(1e-6)
        )
        target_actions = output["target_actions"].float()
        predicted_actions = output["actions"].float()
        normalized_target = (target_actions - action_mean) / action_std
        normalized_predicted = (predicted_actions - action_mean) / action_std

        action_loss = masked_smooth_l1(normalized_predicted, normalized_target)
        intermediate = output["intermediate_actions"].float()
        intermediate_action_loss = torch.stack(
            [
                masked_smooth_l1(
                    (proposal - action_mean) / action_std,
                    normalized_target,
                )
                for proposal in intermediate[:-1]
            ]
        ).mean()
        inverse_loss = F.smooth_l1_loss(
            (output["inverse_actions"].float() - action_mean) / action_std,
            normalized_target,
        )

        teacher_scene_loss = F.mse_loss(teacher_scene, target_scene)
        policy_scene_loss = masked_mean(
            F.mse_loss(policy_scene, target_scene, reduction="none")
        )
        teacher_proprio_loss = F.smooth_l1_loss(teacher_proprio, target_proprio)
        policy_proprio_loss = masked_smooth_l1(policy_proprio, target_proprio)
        final_policy = self.model._gather_time(policy_scene, goal_steps)
        final_goal_loss = F.mse_loss(final_policy, goal_scene)
        final_goal_loss = final_goal_loss + (
            1.0
            - F.cosine_similarity(
                final_policy.mean(dim=1),
                goal_scene.mean(dim=1),
                dim=-1,
            ).mean()
        )

        teacher_at_goal = self.model._gather_time(teacher_scene, goal_steps)
        negative_at_goal = self.model._gather_time(negative_scene, goal_steps)
        teacher_distance = (teacher_at_goal - goal_scene).square().mean(dim=(-1, -2))
        negative_distance = (negative_at_goal - goal_scene).square().mean(dim=(-1, -2))
        ranking_loss = F.relu(0.05 + teacher_distance - negative_distance).mean()

        success_target = torch.zeros_like(output["success_logits"].float())
        success_target.scatter_(1, (goal_steps - 1)[:, None], 1.0)
        progress_target = (time_ids.float() / goal_steps[:, None].float()).clamp_max(
            1.0
        )
        target_scores = output["target_scores"]
        teacher_scores = output["teacher_scores"]
        negative_scores = output["negative_scores"]
        success_loss = (
            F.binary_cross_entropy_with_logits(
                output["success_logits"].float(), success_target
            )
            + F.binary_cross_entropy_with_logits(
                output["current_success_logit"].float(),
                torch.zeros_like(output["current_success_logit"].float()),
            )
            + F.binary_cross_entropy_with_logits(
                target_scores["success_logits"].float(), success_target
            )
            + F.binary_cross_entropy_with_logits(
                teacher_scores["success_logits"].float(), success_target
            )
            + F.binary_cross_entropy_with_logits(
                negative_scores["success_logits"].float(),
                torch.zeros_like(success_target),
            )
        ) / 5.0
        progress_loss = (
            masked_smooth_l1(
                output["progress_logits"].float().sigmoid(), progress_target
            )
            + masked_smooth_l1(
                target_scores["progress_logits"].float().sigmoid(),
                progress_target,
            )
            + masked_smooth_l1(
                teacher_scores["progress_logits"].float().sigmoid(),
                progress_target,
            )
        ) / 3.0
        target_distance = (
            (target_scene - goal_scene[:, None])
            .square()
            .mean(dim=(-1, -2))
            .clamp_min(0)
            .sqrt()
        )
        distance_loss = F.smooth_l1_loss(
            torch.log1p(target_scores["goal_distance"].float()),
            torch.log1p(target_distance),
        )
        sigreg_raw_loss = self.sigreg(output["embeddings"].float().transpose(0, 1))
        # SIGReg is a sample-count-scaled goodness-of-fit statistic.  Keep its
        # effective weight equal to the original batch-128 objective when the
        # physical batch size changes.
        sigreg_loss = sigreg_raw_loss * (
            self.sigreg_reference_batch_size / pixels.shape[0]
        )

        total = (
            teacher_scene_loss
            + 0.5 * policy_scene_loss
            + 0.25 * teacher_proprio_loss
            + 0.15 * policy_proprio_loss
            + 2.0 * action_loss
            + 0.25 * intermediate_action_loss
            + 0.25 * inverse_loss
            + final_goal_loss
            + 0.2 * ranking_loss
            + 0.3 * success_loss
            + 0.15 * progress_loss
            + 0.1 * distance_loss
            + self.sigreg_weight * sigreg_loss
        )
        return {
            "loss": total,
            "action_loss": action_loss,
            "intermediate_action_loss": intermediate_action_loss,
            "inverse_loss": inverse_loss,
            "teacher_scene_loss": teacher_scene_loss,
            "policy_scene_loss": policy_scene_loss,
            "teacher_proprio_loss": teacher_proprio_loss,
            "policy_proprio_loss": policy_proprio_loss,
            "final_goal_loss": final_goal_loss,
            "ranking_loss": ranking_loss,
            "success_loss": success_loss,
            "progress_loss": progress_loss,
            "distance_loss": distance_loss,
            "sigreg_loss": sigreg_loss,
            "sigreg_raw_loss": sigreg_raw_loss,
            "cold_start_fraction": (~history_valid).any(dim=1).float().mean(),
            "full_goal_fraction": (goal_steps == cfg.horizon).float().mean(),
            "mean_goal_blocks": goal_steps.float().mean(),
        }

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        losses = self.compute_losses(batch, stage)
        self.log_dict(
            {f"{stage}/{key}": value for key, value in losses.items()},
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=stage == "val",
            batch_size=batch["pixels"].shape[0],
            sync_dist=False,
        )
        return losses["loss"]

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        if not self.trainer.is_global_zero:
            return
        epoch = self.current_epoch + 1
        metrics = {
            key: float(value.detach().cpu())
            for key, value in self.trainer.callback_metrics.items()
            if torch.is_tensor(value) and value.numel() == 1
        }
        path = self.run_dir / f"weights_epoch_{epoch}.pt"
        self.model.save_weights(
            path,
            epoch=epoch,
            metadata={
                "metrics": metrics,
                "architecture": "WA-JEPA",
                "is_smoke_test": self.is_smoke_test,
            },
        )
        print(f"WA-JEPA weights saved: {path}")

    def configure_optimizers(self):
        representation_prefixes = (
            "encoder.layer_",
            "encoder.scene_",
            "encoder.global_",
            "visual_type",
            "proprio_type",
            "goal_type",
            "context_time",
            "history_valid_embedding.",
            "proprio_encoder.",
            "frame_fusion.",
            "temporal_fusion.",
            "fusion_norm.",
            "current_to_goal.",
            "goal_to_current.",
        )
        representation = []
        planner = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = (
                representation if name.startswith(representation_prefixes) else planner
            )
            target.append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": representation,
                    "lr": self.learning_rate * self.representation_lr_scale,
                },
                {"params": planner, "lr": self.learning_rate},
            ],
            weight_decay=self.weight_decay,
        )
        total_steps = max(2, int(self.trainer.estimated_stepping_batches))
        warmup_steps = max(1, int(0.01 * total_steps))

        def schedule(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=os.environ.get("LOCAL_DATASET_DIR", "data"),
    )
    parser.add_argument("--dataset-name", default="pusht_expert_train.lance")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--representation-lr-scale", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.0)
    parser.add_argument("--sigreg-reference-batch-size", type=int, default=128)
    parser.add_argument("--history-dropout-prob", type=float, default=0.25)
    parser.add_argument("--full-goal-probability", type=float, default=0.5)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--world-depth", type=int, default=3)
    parser.add_argument("--scene-tokens", type=int, default=8)
    parser.add_argument("--frame-fusion-depth", type=int, default=1)
    parser.add_argument("--temporal-fusion-depth", type=int, default=2)
    parser.add_argument("--goal-match-depth", type=int, default=1)
    parser.add_argument(
        "--dino-repo",
        default=os.environ.get("DINO_REPO", "third_party/dinov2"),
    )
    parser.add_argument(
        "--dino-checkpoint",
        default=(
            os.environ.get("DINO_CHECKPOINT", "pretrained/dinov2_vits14_pretrain.pth")
        ),
    )
    parser.add_argument("--resume-weights", default="")
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def finite_column(dataset, key: str) -> np.ndarray:
    values = np.asarray(dataset.get_col_data(key), dtype=np.float32)
    return values[np.isfinite(values).all(axis=-1)]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if Path(args.run_dir).exists():
        raise FileExistsError(
            "Use a new run directory; existing runs are not overwritten"
        )
    os.environ["DINO_REPO"] = args.dino_repo
    os.environ["DINO_CHECKPOINT"] = args.dino_checkpoint
    if not 0.0 <= args.full_goal_probability <= 1.0:
        raise ValueError("--full-goal-probability must be in [0, 1]")
    if args.sigreg_reference_batch_size <= 0:
        raise ValueError("--sigreg-reference-batch-size must be positive")
    if args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed, workers=True)
    dataset = swm.data.load_dataset(
        args.dataset_name,
        transform=None,
        cache_dir=args.dataset_root,
        num_steps=8,
        frameskip=5,
        keys_to_load=["pixels", "action", "proprio"],
    )
    action_data = finite_column(dataset, "action")
    proprio_data = finite_column(dataset, "proprio")
    action_mean = torch.from_numpy(action_data.mean(axis=0))
    action_std = torch.from_numpy(action_data.std(axis=0)).clamp_min(1e-6)
    proprio_mean = torch.from_numpy(proprio_data.mean(axis=0))
    proprio_std = torch.from_numpy(proprio_data.std(axis=0)).clamp_min(1e-6)
    train_set, val_set, train_episodes, val_episodes = split_by_episode(
        dataset, val_fraction=0.1, seed=args.seed
    )
    loader_common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if args.num_workers > 0:
        loader_common["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_common,
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_common)

    if args.resume_weights:
        model, payload = WorldActionJEPA.load_weights(args.resume_weights)
        print(f"Loaded WA-JEPA epoch {payload['epoch']} from {args.resume_weights}")
    else:
        model = WorldActionJEPA(
            refinement_steps=args.refinement_steps,
            world_depth=args.world_depth,
            scene_tokens=args.scene_tokens,
            frame_fusion_depth=args.frame_fusion_depth,
            temporal_fusion_depth=args.temporal_fusion_depth,
            goal_match_depth=args.goal_match_depth,
            dino_repo=args.dino_repo,
            dino_checkpoint=args.dino_checkpoint,
        )
        model.set_action_stats(action_mean, action_std)
        model.set_proprio_stats(proprio_mean, proprio_std)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    configuration = {
        "architecture": "WA-JEPA",
        "dataset": args.dataset_name,
        "dataset_samples": len(dataset),
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "split": "episode-level",
        "goal_horizon_blocks": 5,
        "goal_horizon_environment_steps": 25,
        "full_horizon_all_action_chunks_supervised": True,
        "short_horizon_actions_after_goal_supervised": False,
        "goal_horizon_training": "50% fixed 25-step plus 50% uniform 5-25-step",
        "full_goal_probability": args.full_goal_probability,
        "history_dropout_prob": args.history_dropout_prob,
        "state_used_by_model": False,
        "success_supervision": "temporal image-goal proxy; no state/reward column",
        "model": model.model_config,
        "dino_checkpoint_sha256": sha256_file(model.cfg.dino_checkpoint),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "proprio_mean": proprio_mean.tolist(),
        "proprio_std": proprio_std.tolist(),
        "training": vars(args),
    }
    with (run_dir / "config.json").open("w") as handle:
        json.dump(configuration, handle, indent=2)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dataset samples={len(dataset):,}")
    print(
        f"Episode split train={train_episodes:,}, val={val_episodes:,}; "
        f"clips train={len(train_set):,}, val={len(val_set):,}"
    )
    print(f"WA-JEPA parameters={total:,}, trainable={trainable:,}")
    print("Goal task: fixed 5 blocks / 25 environment steps")
    print("Full-horizon samples directly supervise all five action chunks")
    print(
        "Replanning curriculum: "
        f"full 25-step probability={args.full_goal_probability:.1%}; "
        "otherwise sample 5/10/15/20/25 remaining steps"
    )
    print(f"Cold-start history augmentation={args.history_dropout_prob:.1%}")
    print("Simulator state is not loaded or consumed")
    print(f"Run directory: {run_dir}")
    if args.dry_run:
        print("Dry run complete; training was not started.")
        return

    module = WAJEPALitModule(
        model=model,
        run_dir=str(run_dir),
        learning_rate=args.lr,
        representation_lr_scale=args.representation_lr_scale,
        weight_decay=args.weight_decay,
        sigreg_weight=args.sigreg_weight,
        sigreg_reference_batch_size=args.sigreg_reference_batch_size,
        history_dropout_prob=args.history_dropout_prob,
        full_goal_probability=args.full_goal_probability,
    )
    module.is_smoke_test = args.limit_train_batches is not None
    logger = CSVLogger(save_dir=str(run_dir), name="metrics")
    trainer_kwargs = {
        "accelerator": "gpu",
        "devices": 1,
        "precision": "bf16-mixed",
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": 1.0,
        "logger": logger,
        "enable_checkpointing": False,
        "log_every_n_steps": 50,
        "num_sanity_val_steps": 1,
        "limit_val_batches": args.val_batches,
    }
    if args.limit_train_batches is not None:
        trainer_kwargs["limit_train_batches"] = args.limit_train_batches
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
