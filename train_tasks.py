"""V4-1 on TwoRoom/Cube: only SIGReg removed; automatic evaluation every epoch.

Architecture and losses follow V4-1. Input dimensions/history adapt to
each task. Preserve baseline window split membership, filtering only windows
too short for V4's five future blocks. No PushT task checkpoint is loaded.
"""

import argparse
import os
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import stable_pretraining as spt
from stable_pretraining.optim.lr_scheduler import create_scheduler
import stable_worldmodel as swm
import torch
from torch.utils.data import DataLoader, Subset

from train_pusht import WAJEPAV4LitModule, finite_column, sha256_file
from wa_jepa_v4 import WorldActionJEPAV4
from runtime_utils import atomic_json
from task_evaluation import evaluate_checkpoint


TASKS = {
    "tworoom": dict(dataset="tworoom.h5", history=1, action_dim=2, proprio_dim=2),
    "cube": dict(
        dataset="ogbench/cube_single_expert.h5", history=3, action_dim=5, proprio_dim=19
    ),
}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def baseline_aligned_split(dataset, history, seed=3072):
    """Use EXACT baseline random split; map its surviving starts into V4 indices."""
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    base_counts = np.maximum(0, lengths - (history + 1) * 5 + 1)
    long_counts = np.maximum(0, lengths - (history + 5) * 5 + 1)
    mapping = np.full(int(base_counts.sum()), -1, dtype=np.int64)
    base_start = long_start = 0
    for short, long in zip(base_counts, long_counts):
        mapping[base_start : base_start + long] = np.arange(
            long_start, long_start + long
        )
        base_start += short
        long_start += long
    assert long_start == len(dataset)
    rng = torch.Generator().manual_seed(seed)
    original = spt.data.random_split(range(len(mapping)), [0.9, 0.1], generator=rng)
    subsets = []
    metadata = dict(
        baseline_windows=len(mapping),
        v4_windows=len(dataset),
        split="baseline random-window 90/10; NOT episode-level",
    )
    for name, subset in zip(("train", "val"), original):
        ids = np.asarray(subset.indices, dtype=np.int64)
        mapped = mapping[ids]
        keep = mapped >= 0
        metadata[name + "_baseline_count"] = len(ids)
        metadata[name + "_count"] = int(keep.sum())
        metadata[name + "_baseline_ids_sha256"] = hashlib.sha256(
            ids[keep].tobytes()
        ).hexdigest()
        subsets.append(Subset(dataset, mapped[keep]))
    return *subsets, rng, metadata


def load_task_data(task, root):
    cfg = TASKS[task]
    path = Path(root) / "datasets" / cfg["dataset"]
    with h5py.File(path, "r") as f:
        proprio_keys = (
            ["proprio"]
            if task == "tworoom"
            else [k for k in f.keys() if k.startswith("proprio_")]
        )
        assert f["action"].shape[-1] == cfg["action_dim"]
        assert sum(f[k].shape[-1] for k in proprio_keys) == cfg["proprio_dim"]
        assert f["pixels"].shape[1:] == (224, 224, 3)
    kwargs = dict(
        path=path,
        num_steps=cfg["history"] + 5,
        frameskip=5,
        keys_to_load=["pixels", "action", "proprio"],
        keys_to_cache=["action"] + (["proprio"] if task == "tworoom" else []),
    )
    if task == "cube":
        kwargs["keys_to_merge"] = {"proprio": proprio_keys}
    dataset = swm.data.HDF5Dataset(**kwargs)
    stats = {}
    for key in ("action", "proprio"):
        values = finite_column(dataset, key)
        if not len(values):
            raise ValueError(f"Empty finite column: {key}")
        if key == "action" and (values.min() < -1.0001 or values.max() > 1.0001):
            raise ValueError("Action bounds differ from V4 [-1,1]")
        stats[key] = (
            torch.from_numpy(values.mean(0)),
            torch.from_numpy(values.std(0)).clamp_min(1e-6),
        )
        del values
    # HDF5 handles must not be inherited by forked workers. They reopen lazily.
    if dataset.h5_file is not None:
        dataset.h5_file.close()
        dataset.h5_file = None
    return dataset, stats, proprio_keys, path


def make_loss(model, root):
    return WAJEPAV4LitModule(
        model=model,
        run_dir=str(root),
        learning_rate=5e-5,
        representation_lr_scale=1.0,
        weight_decay=0.001,
        sigreg_weight=0.0,
        sigreg_reference_batch_size=128,
        history_dropout_prob=0.0 if model.cfg.history_size == 1 else 0.25,
        full_goal_probability=0.5,
    )


def make_scheduler(optimizer, total_steps):
    # Baseline spt.Module is MANUAL optimization: despite interval='epoch' in
    # train.py, module.training_step calls scheduler.step() EACH optimizer step.
    class Context:
        estimated_stepping_batches = total_steps

    class Module:
        trainer = Context()

    return create_scheduler(
        optimizer, {"type": "LinearWarmupCosineAnnealingLR"}, Module()
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--dataset-root", default=os.environ.get("LOCAL_DATASET_DIR", "data")
    )
    p.add_argument(
        "--memory-cap-gib", type=float, help="Optional PyTorch allocator limit"
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--smoke-batches",
        type=int,
        default=0,
        help="Real-data full-batch preflight, no formal accuracy",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="I/O-only safety override: baseline used 6; concurrent jobs use 2 to fit host RAM",
    )
    p.add_argument("--prefetch-factor", type=int, default=1)
    a = p.parse_args()
    if (
        a.smoke_batches < 0
        or a.num_workers < 0
        or a.prefetch_factor < 1
        or a.batch_size < 1
        or a.epochs < 1
    ):
        p.error("Negative batch/worker count")
    return a


def main():
    args = parse_args()
    cfg = TASKS[args.task]
    cap_gib = args.memory_cap_gib
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Training requires a CUDA GPU with BF16 support")
    if cap_gib is not None:
        free, total = torch.cuda.mem_get_info()
        if cap_gib <= 0 or cap_gib * 2**30 > total:
            raise ValueError("Invalid memory cap")
        torch.cuda.set_per_process_memory_fraction(cap_gib * 2**30 / total)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        root / "status.json",
        dict(
            status="initializing",
            task=args.task,
            is_smoke_test=bool(args.smoke_batches),
        ),
    )
    seed_all(3072)
    started = time.perf_counter()
    dataset, stats, proprio_keys, data_path = load_task_data(
        args.task, args.dataset_root
    )
    train_set, val_set, rng, split = baseline_aligned_split(dataset, cfg["history"])
    loader = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True
    )
    if args.num_workers:
        loader.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context="fork",
        )
    train_loader = DataLoader(
        train_set, shuffle=True, drop_last=True, generator=rng, **loader
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader)
    model = WorldActionJEPAV4(
        history_size=cfg["history"],
        raw_action_dim=cfg["action_dim"],
        proprio_dim=cfg["proprio_dim"],
    )
    model.set_action_stats(*stats["action"])
    model.set_proprio_stats(*stats["proprio"])
    module = make_loss(model, root).cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.001)
    train_batches = min(len(train_loader), args.smoke_batches or len(train_loader))
    val_batches = min(len(val_loader), args.smoke_batches or len(val_loader))
    if train_batches == 0 or val_batches == 0:
        raise ValueError("Empty loader; check dataset and batch size")
    epochs = 1 if args.smoke_batches else args.epochs
    # At least two scheduler steps are needed for a one-batch smoke run.
    scheduler = make_scheduler(optimizer, max(2, epochs * train_batches))
    config = dict(
        task=args.task,
        architecture="WA-JEPA V4-1",
        model=model.model_config,
        data_path=str(data_path),
        data_size=data_path.stat().st_size,
        data_mtime_ns=data_path.stat().st_mtime_ns,
        proprio_columns=proprio_keys,
        privileged_state_inputs=False,
        split=split,
        training=dict(
            epochs=epochs,
            batch_size=args.batch_size,
            lr=5e-5,
            weight_decay=0.001,
            seed=3072,
            precision="bf16-mixed",
            gradient_clip_norm=1.0,
            train_batches=train_batches,
            val_batches=val_batches,
            history_dropout_prob=module.history_dropout_prob,
            full_goal_probability=0.5,
            sigreg_weight=0.0,
            num_workers=args.num_workers,
        ),
        scheduler=dict(
            type="LinearWarmupCosineAnnealingLR",
            interval="step",
            max_steps=scheduler.max_steps,
            warmup_steps=scheduler.warmup_steps,
        ),
        normalization={
            k: dict(mean=m.tolist(), std=s.tolist()) for k, (m, s) in stats.items()
        },
        is_smoke_test=bool(args.smoke_batches),
        source_sha256={
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "wa_jepa_v4.py",
                "train_pusht.py",
                "train_tasks.py",
                "runtime_utils.py",
                "eval_tasks.py",
            )
        },
    )
    (root / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)
    print(
        f"{args.task}: batch={args.batch_size}, epochs={epochs}, train={train_batches}, val={val_batches}, GPU cap={cap_gib} GiB",
        flush=True,
    )
    # Match baseline's one validation sanity batch before training.
    module.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        sanity = next(iter(val_loader))
        sanity = {k: v.cuda(non_blocking=True) for k, v in sanity.items()}
        sanity_losses = module.compute_losses(sanity, "val")
        assert all(torch.isfinite(v).all() for v in sanity_losses.values())
    del sanity, sanity_losses
    training_seconds = training_batches = validation_seconds = validation_epochs = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        atomic_json(
            root / "status.json",
            dict(
                status="training",
                epoch=epoch,
                epochs=epochs,
                is_smoke_test=bool(args.smoke_batches),
            ),
        )
        epoch_start = time.perf_counter()
        module.train()
        sums, samples = {}, 0
        for step, batch in enumerate(itertools.islice(train_loader, train_batches), 1):
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = module.compute_losses(batch, "train")
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch} step={step}")
            losses["loss"].backward()
            grad = torch.nn.utils.clip_grad_norm_(
                parameters, 1.0, error_if_nonfinite=True
            )
            lr_used = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            global_step += 1
            size = len(batch["pixels"])
            values = {k: float(v.detach()) for k, v in losses.items()}
            samples += size
            for k, v in values.items():
                sums[k] = sums.get(k, 0.0) + size * v
            del losses, batch
            if step == 1 or step % 50 == 0 or step == train_batches:
                now = time.perf_counter()
                rate = (training_seconds + now - epoch_start) / (
                    training_batches + step
                )
                val_est = (
                    validation_seconds / validation_epochs
                    if validation_epochs
                    else rate * val_batches
                )
                remaining = (
                    (train_batches - step) * rate
                    + val_est
                    + (epochs - epoch) * (train_batches * rate + val_est)
                )
                print(
                    f"{args.task} E{epoch}/{epochs} {step}/{train_batches} updates={global_step} "
                    f"loss={values['loss']:.5f} action={values['action_loss']:.5f} lr={lr_used:.4g} grad={float(grad):.3f} | "
                    f"elapsed {duration(now - started)} remaining~{duration(remaining)} "
                    f"GPU peak={torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                    flush=True,
                )
        training_seconds += time.perf_counter() - epoch_start
        training_batches += train_batches
        train_metrics = {k: v / samples for k, v in sums.items()}
        optimizer.zero_grad(set_to_none=True)
        module.eval()
        val_start = time.perf_counter()
        sums, samples = {}, 0
        with torch.no_grad():
            for step, batch in enumerate(itertools.islice(val_loader, val_batches), 1):
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    losses = module.compute_losses(batch, "val")
                size = len(batch["pixels"])
                samples += size
                for k, v in losses.items():
                    value = float(v)
                    if not math.isfinite(value):
                        raise FloatingPointError(f"Nonfinite validation {k}")
                    sums[k] = sums.get(k, 0.0) + size * value
                del batch, losses
                if step == 1 or step % 100 == 0 or step == val_batches:
                    print(
                        f"{args.task} E{epoch} validation {step}/{val_batches} elapsed {duration(time.perf_counter() - started)}",
                        flush=True,
                    )
        validation_seconds += time.perf_counter() - val_start
        validation_epochs += 1
        metrics = dict(
            epoch=epoch,
            train=train_metrics,
            val={k: v / samples for k, v in sums.items()},
            optimizer_updates=global_step,
            lr_last_used=lr_used,
            elapsed_seconds=time.perf_counter() - started,
        )
        payload = dict(
            format="wa_jepa_v4",
            epoch=epoch,
            model_config=model.model_config,
            model_state_dict=model.state_dict(),
            metadata=dict(metrics=metrics, task=args.task),
            training_provenance=config,
            is_smoke_test=bool(args.smoke_batches),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=dict(
                python=random.getstate(),
                numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),
                train_generator=rng.get_state(),
            ),
        )
        path = root / f"weights_epoch_{epoch}.pt"
        if path.exists():
            raise FileExistsError(path)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        del payload
        with (root / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(
            f"SAVED {path} val action={metrics['val']['action_loss']:.6f}", flush=True
        )
        if not args.smoke_batches and not args.skip_eval:
            atomic_json(
                root / "status.json",
                dict(status="evaluating", epoch=epoch, checkpoint=str(path)),
            )
            try:
                evaluate_checkpoint(args.task, path, root, epoch, args.dataset_root)
            except Exception as exc:
                atomic_json(
                    root / "status.json",
                    dict(
                        status="evaluation_failed",
                        epoch=epoch,
                        checkpoint=str(path),
                        error=repr(exc),
                    ),
                )
                raise
    atomic_json(
        root / "status.json",
        dict(
            status="completed",
            epochs=epochs,
            is_smoke_test=bool(args.smoke_batches),
            automatic_evaluations=0 if args.smoke_batches or args.skip_eval else epochs,
        ),
    )


if __name__ == "__main__":
    main()
