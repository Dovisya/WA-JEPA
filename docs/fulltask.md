# Full-task execution: Push-T and LIBERO-Goal

This package releases the paper's **VQ-BeT full-task adaptation**: data preparation, frozen feature caching, action-codebook pretraining, policy training, fine-tuning, and simulator evaluation. The short-horizon code remains independent. The high-level AP planner is not included in this release.

## Environment

Use a separate **Python 3.9** environment on Linux with NVIDIA GPUs. These dependencies are separate from the short-horizon environment. The evaluation wrappers and VQ-BeT modules come from the [Patch Policy repository](https://github.com/gaoyuezhou/patch_policy), pinned to commit `674b09d9de0a7ea6226583c13078b4685b358485`.

```bash
conda create -n wa-jepa-fulltask python=3.9 -y
conda activate wa-jepa-fulltask
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
bash scripts/setup_fulltask_dependencies.sh
```

Run commands from the WA-JEPA repository root. Obtain the DINOv2 code and weights below. This revision supports Python 3.9; newer DINOv2 revisions may require a newer Python version. If you already have another DINOv2 checkout, clone this revision into a separate directory and set `DINO_REPO` accordingly.

```bash
mkdir -p third_party pretrained
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
git -C third_party/dinov2 checkout --detach 85a24602099d397264d5b30461ad7f3bfd726ca1
curl -fL https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  -o pretrained/dinov2_vits14_pretrain.pth
source scripts/fulltask_env.sh
```

The launchers set EGL rendering and keep the LIBERO configuration in `.libero_fulltask/`. A working NVIDIA EGL driver is required for rollouts. No simulator or pretrained model assets are bundled in this repository.

## Data and feature preparation

Use the [Patch Policy full-task datasets](https://huggingface.co/datasets/gaoyuezhou/patch-policy-datasets), not the short-horizon goal-reaching datasets.

```bash
huggingface-cli download gaoyuezhou/patch-policy-datasets --repo-type dataset \
  --include "pusht_dataset.zip" "libero_dataset.zip" --local-dir data/fulltask
unzip -q data/fulltask/pusht_dataset.zip -d data/fulltask
unzip -q data/fulltask/libero_dataset.zip -d data/fulltask
```

Default data paths are `data/fulltask/pusht_dataset` and `data/fulltask/libero_dataset/libero_goal`. Override `PUSHT_DATA_ROOT` and `LIBERO_DATA_ROOT` before sourcing the environment file to use existing datasets. Similarly, `PATCH_POLICY_ROOT`, `DINO_REPO`, and `DINO_CHECKPOINT` can point to existing dependencies.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_fulltask.sh pusht cache/fulltask/pusht
CUDA_VISIBLE_DEVICES=0 bash scripts/prepare_fulltask.sh libero_goal cache/fulltask/libero_goal
```

Each command caches frozen DINOv2 features, writes training-only normalization statistics and goal features, and fits the VQ action autoencoder for 1,000 passes. A completed cache or codebook is reused after its metadata is checked. Feature caches contain dense FP32 tensors and are substantially larger than the image datasets; use storage with sufficient capacity. Cache preparation uses one GPU by default. For distributed LIBERO feature extraction, `fulltask.libero_prepare` exposes `--rank` and `--world-size`; run `--finalize` only after all shards finish. Push-T preparation uses a single process.

The Push-T split contains 185 training and 21 validation trajectories, with split seed 42. LIBERO-Goal uses 45 training and five validation demonstrations per task, with split seed `42 + task_id`. Each LIBERO goal is the terminal frame of a seeded training demonstration. Held-out trajectories do not supply training goals or normalization statistics.

## Training settings

| Setting | Push-T | LIBERO-Goal |
|---|---|---|
| Visual backbone | Frozen DINOv2 ViT-S/14 | Frozen DINOv2 ViT-S/14 |
| Views / history frames | 1 / 5 | 2 / 5 |
| Scene tokens / embedding width | 8 / 192 | 8 / 192 |
| Action output | One five-action chunk, 2D absolute position | One five-action chunk, 7D control |
| Proprioception | Position and displacement, 4D | Visual inputs only |
| Proposal stages / feedback updates | 4 / 3 | 4 / 3 |
| Epochs | 400 | 400 |
| Windows per GPU | 256 | 256 |
| Default GPUs / global batch | 4 / 1,024 | 8 / 2,048 |
| Optimizer | AdamW | AdamW |
| Learning rate / weight decay | 5e-5 / 1e-3 | 5e-5 / 1e-3 |
| Schedule | 1% warmup, cosine decay | 1% warmup, cosine decay |
| L1 reconstruction weight | 10 | 100 |
| Focal code weights / gamma | 5 and 0.5 / 2 | 5 and 0.5 / 2 |
| Proposal supervision | Final + 0.25 × mean intermediate | Final + 0.25 × mean intermediate |
| Scene / proprioception prediction weights | 1 / 0.25 | 1 / omitted |

These are the paper's base training settings. LIBERO uses the eight-GPU launch layout of the archived training implementation; the paper specifies its per-GPU batch size. The action autoencoder and codebooks stay frozen during policy training. World-model supervision uses recorded demonstration actions and subsequent observations; missing future targets at trajectory tails are masked. Action tails repeat the last valid action.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/train_fulltask_pusht.sh cache/fulltask/pusht runs/fulltask/pusht_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_fulltask_libero_goal.sh cache/fulltask/libero_goal runs/fulltask/libero_seed42
```

Set `SEED` to change the training seed. Dataset splits and the 100 evaluation cases remain fixed. `NPROC_PER_NODE` changes the GPU count (Push-T: 1 or 4; LIBERO: 1, 4, or 8), which also changes the global batch size at a fixed per-GPU batch. The LIBERO loader assembles a logical batch on the device in small host transfers; it does not replace the configured batch with gradient accumulation.

Training uses FP32; BF16 in the paper's cost table refers to inference measurement. The Push-T trainer retains the archived initialization convention: epoch 0 runs with zero-weighted loss, followed by 399 learning epochs within the 400-epoch schedule. Each epoch includes validation and a 100-episode simulator evaluation. `metrics.jsonl` records training loss, gradients, rollout metrics, and timing. `best.pt` selects the best rollout checkpoint; `latest.pt` stores the most recent complete training state. LIBERO additionally retains the top three checkpoints.

### Resume and optional fine-tuning

To resume an interrupted run, repeat the same launch command and append `--resume`. Keep its original schedule, batch size, seed, paths, and source unchanged. Model, optimizer, scheduler, and per-rank random states are restored.

For a separate fine-tuning run, use a **new run directory** and `--init-checkpoint` with a full checkpoint. This restores weights and optimizer moments while starting a new learning-rate schedule. The base launchers above do not silently add fine-tuning. For example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/train_fulltask_pusht.sh cache/fulltask/pusht runs/fulltask/pusht_finetune \
  --init-checkpoint runs/fulltask/pusht_seed42/best.pt \
  --epochs 100 --batch-per-gpu 64 --learning-rate 5e-6 \
  --min-learning-rate 5e-7 --warmup-fraction 0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  bash scripts/train_fulltask_libero_goal.sh cache/fulltask/libero_goal runs/fulltask/libero_finetune \
  --init-checkpoint runs/fulltask/libero_seed42/best.pt \
  --epochs 100 --batch-per-gpu 512 --learning-rate 1e-5 \
  --min-learning-rate 2e-6 --warmup-fraction 0
```

These examples expose the separate fine-tuning settings used during development, rather than redefining the paper's 400-epoch base schedule. Fine-tuning epoch numbers restart at zero in the new directory. Older model-only exports can be evaluated but cannot restore optimizer state for fine-tuning.

## Evaluation

```bash
source scripts/fulltask_env.sh
CUDA_VISIBLE_DEVICES=0 python -m fulltask.eval_checkpoint \
  --task pusht --cache cache/fulltask/pusht \
  --checkpoint runs/fulltask/pusht_seed42/best.pt \
  --output runs/fulltask/pusht_seed42/evaluation.json

CUDA_VISIBLE_DEVICES=0 python -m fulltask.eval_checkpoint \
  --task libero_goal --cache cache/fulltask/libero_goal \
  --checkpoint runs/fulltask/libero_seed42/best.pt \
  --output runs/fulltask/libero_seed42/evaluation.json
```

Push-T evaluates 100 rollouts with at most 302 actions and reports **final coverage**, not maximum coverage over a rollout. LIBERO-Goal evaluates ten episodes per task (100 total) with a 300-action budget and reports environment success. Both predict five actions per decision and execute one temporally ensembled action. Evaluation uses seed 42 and fixed logical environment batches, preserving the sampling order independently of training GPU count. The Push-T evaluator retains the upstream reset order, including reuse of the first seed group in the second batch. The evaluation JSON retains per-episode results and aggregate metrics.

The paper reports **71 ± 2% final coverage** and **97 ± 1% success** across three independently trained models. A single training run produces one result; reproduce the mean and standard deviation by training independent seeds and aggregating the same metric. These numbers are reported experimental results, not a guarantee for an arbitrary new seed or modified training configuration. Trained full-task checkpoints are not included in this code release.

## Source map

- `fulltask/architecture.py`: scene encoding, action-conditioned world model, and both feedback paths.
- `fulltask/wa_model.py`, `fulltask/libero_model.py`: task adapters and VQ-BeT heads.
- `fulltask/data.py`, `fulltask/libero_data.py`: splits, windows, and future-target masks.
- `fulltask/prepare.py`, `fulltask/libero_prepare.py`: frozen-feature caches.
- `fulltask/codebook.py`, `fulltask/libero_codebook.py`: training-only VQ pretraining.
- `fulltask/train.py`, `fulltask/train_libero.py`: policy training and checkpointing.
- `fulltask/evaluate.py`, `fulltask/libero_evaluate.py`: simulator rollout protocols.
- `fulltask/eval_checkpoint.py`: standalone checkpoint evaluation.
