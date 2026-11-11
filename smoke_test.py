"""Real-data CUDA smoke test: train loss/backward, optimizer step, plan, checkpoint reload.

This checks execution only, not convergence or benchmark accuracy.
"""

import argparse
import gc
import json
from pathlib import Path
import tempfile
import hdf5plugin  # noqa: F401
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader
from train_tasks import TASKS, load_task_data, make_loss, seed_all
from train_pusht import finite_column
from wa_jepa_v4 import WorldActionJEPAV4


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["pusht", "tworoom", "cube"],
        default=["pusht", "tworoom", "cube"],
    )
    p.add_argument("--output", default=None)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    reports = []
    for task in args.tasks:
        seed_all(3072)
        if task == "pusht":
            dataset = swm.data.load_dataset(
                "pusht_expert_train.lance",
                cache_dir=args.dataset_root,
                num_steps=8,
                frameskip=5,
                keys_to_load=["pixels", "action", "proprio"],
            )
            cfg = dict(history=3, action_dim=2, proprio_dim=4)
            stats = {}
            for key in ("action", "proprio"):
                values = finite_column(dataset, key)
                stats[key] = (
                    torch.from_numpy(values.mean(0)),
                    torch.from_numpy(values.std(0)).clamp_min(1e-6),
                )
        else:
            dataset, stats, _, _ = load_task_data(task, args.dataset_root)
            cfg = TASKS[task]
        batch = next(
            iter(DataLoader(dataset, batch_size=args.batch_size, num_workers=0))
        )
        batch = {k: v.cuda() for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats()
        model = WorldActionJEPAV4(
            history_size=cfg["history"],
            raw_action_dim=cfg["action_dim"],
            proprio_dim=cfg["proprio_dim"],
        )
        model.set_action_stats(*stats["action"])
        model.set_proprio_stats(*stats["proprio"])
        module = make_loss(model, ".").cuda().train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=5e-5
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = module.compute_losses(
                {k: v.clone() for k, v in batch.items()}, "train"
            )
        assert all(torch.isfinite(v).all() for v in losses.values())
        losses["loss"].backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert all(p.grad is None for p in model.encoder.dino.parameters())
        optimizer.step()
        loss_value = float(losses["loss"].detach())
        del losses
        model.eval()
        history = cfg["history"]
        pixels = batch["pixels"][:, :history]
        proprio = batch["proprio"][:, :history]
        goal = batch["pixels"][:, history + 4]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            planned = model.plan(pixels, proprio, goal)
        assert planned["actions"].shape == (args.batch_size, 5, 5 * cfg["action_dim"])
        assert torch.isfinite(planned["actions"]).all()
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "smoke.pt"
            model.save_weights(file, epoch=0, metadata={"is_smoke_test": True})
            restored, payload = WorldActionJEPAV4.load_weights(file)
            restored.cuda().eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                replay = restored.plan(pixels, proprio, goal)
            torch.testing.assert_close(
                planned["actions"], replay["actions"], rtol=0, atol=0
            )
        reports.append(
            dict(
                task=task,
                batch_size=args.batch_size,
                status="passed",
                loss=loss_value,
                gradient_norm=float(grad),
                action_shape=list(planned["actions"].shape),
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
        )
        print(json.dumps(reports[-1]), flush=True)
        del (
            module,
            model,
            restored,
            optimizer,
            planned,
            replay,
            dataset,
            batch,
            pixels,
            proprio,
            goal,
        )
        gc.collect()
        torch.cuda.empty_cache()
    if args.output:
        Path(args.output).write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
