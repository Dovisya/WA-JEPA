# WA-JEPA — Short-Horizon Goal Reaching

Code for the **WA-JEPA V4-1** short-horizon experiments on **PushT, TwoRoom, and OGBench-Cube**. V4-1 uses the V4 network and removes SIGReg during training.

This release contains the network, training objectives, deployment policy, data preparation, and evaluation entry points. It does not include full-task policy heads, long-horizon planning, alternative architectures, datasets, checkpoints, or experiment logs.

## Installation

The tested setup is Linux, Python 3.10, an NVIDIA A40, CUDA 12.1 PyTorch wheels, and BF16. Training and evaluation expect a CUDA GPU; macOS can be used to edit the source.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
export MUJOCO_GL=egl
export MPLBACKEND=Agg
```

The requirements record the principal packages in the tested environment, rather than a lockfile of every transitive dependency. The code was checked in the existing experiment environment; installation in a completely new environment has not been tested.

### Frozen visual encoder

Obtain the official [DINOv2](https://github.com/facebookresearch/dinov2) code and ViT-S/14 pretrained weights:

```bash
mkdir -p third_party pretrained
git clone https://github.com/facebookresearch/dinov2.git third_party/dinov2
curl -fL https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth \
  -o pretrained/dinov2_vits14_pretrain.pth
export DINO_REPO="$PWD/third_party/dinov2"
export DINO_CHECKPOINT="$PWD/pretrained/dinov2_vits14_pretrain.pth"
```

The experiment's pretrained weight SHA-256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. DINOv2 stays frozen. xFormers was not installed in the checked environment and is not required. Run commands from this repository's root. The two environment variables also relocate paths stored in older checkpoints without changing their learned tensors.

## Data

Use the short-horizon datasets distributed with [LeWorldModel](https://github.com/lucas-maes/le-wm) through its [data collection](https://huggingface.co/collections/quentinll/lewm). These are the goal-reaching benchmark datasets; do not substitute the full-task Push-T imitation-learning dataset.

Arrange the extracted data as follows:

```text
data/
└── datasets/
    ├── pusht_expert_train.h5
    ├── pusht_expert_train.lance/  # derived training copy
    ├── tworoom.h5
    └── ogbench/
        └── cube_single_expert.h5
```

PushT training uses the Lance reader; evaluation uses the original HDF5 file. Convert once, retaining the HDF5 file:

```bash
python prepare_pusht.py --dataset-root data
```

The converter uses the installed stable-worldmodel writer (JPEG quality 95) and refuses to overwrite an existing Lance table. Reuse the original converted training copy when reproducing an existing run exactly; conversion is not lossless. TwoRoom and Cube use HDF5 directly. All scripts accept `--dataset-root`; its value is the **parent of `datasets/`**, not the `datasets/` directory itself. `LOCAL_DATASET_DIR` is also accepted by training/evaluation entry points.

## Training

| Setting | PushT | TwoRoom | Cube |
|---|---:|---:|---:|
| Epochs | 20 | 10 | 10 |
| Batch size | 512 | 128 | 128 |
| Observation history | 3 | 1 | 3 |
| Action dimension per step | 2 | 2 | 5 |
| Proprioceptive dimension | 4 | 2 | 19 |
| Training/validation split | Episode-level 90/10 | Mapped random windows 90/10 | Mapped random windows 90/10 |

Shared defaults: seed 3072, AdamW, learning rate `5e-5`, weight decay `1e-3`, gradient clipping at 1.0, BF16, five action chunks of five steps, four proposals with three feedback updates, eight scene tokens, and a three-layer world model. Each task is trained from scratch apart from the frozen DINOv2 encoder.

```bash
python train_pusht.py --dataset-root data --run-dir runs/pusht
python train_tasks.py --task tworoom --dataset-root data --run-dir runs/tworoom
python train_tasks.py --task cube --dataset-root data --run-dir runs/cube
```

Run these separately, or assign different GPUs through `CUDA_VISIBLE_DEVICES`. These are single-GPU training entry points. Use a new output directory for each run; existing runs are not overwritten. Lowering `--batch-size` is supported for smoke tests but changes the experimental configuration.

TwoRoom/Cube evaluate the new checkpoint after every epoch in a separate process. Pass `--skip-eval` to evaluate later. PushT saves one checkpoint per epoch; use the evaluation command below for each checkpoint. `--resume-weights` in `train_pusht.py` loads model weights only, not optimizer/scheduler state; it is not an exact interrupted-run resume.

### Fidelity to the original short-horizon experiments

- PushT uses an episode-level split. TwoRoom/Cube reproduce the original random-window split, then filter starts without enough future observations; they are **not episode-disjoint splits**.
- Normalization statistics are computed with the original full-dataset procedure.
- The goal curriculum keeps the 25-step goal with probability 0.5 and otherwise samples uniformly from the five chunk endpoints. Actions after a sampled early goal are masked as in the original trainer.
- History dropout is disabled for TwoRoom's single-frame input. PushT/Cube use probability 0.25.
- All original non-SIGReg training losses and auxiliary heads are retained. Removing their code would change the trained model. Their predictions do not rank candidate actions in the default deployment policy.
- The zero-SIGReg module retains the original random-number draw to preserve the stochastic training sequence.
- PushT retains its LambdaLR warmup/cosine schedule. TwoRoom/Cube retain stable-pretraining's `LinearWarmupCosineAnnealingLR`, advanced on every optimizer step. These schedules have not been unified.

## Short-horizon evaluation

The default protocol uses **50 cases**, seed **42**, a target **25 environment steps** ahead, and a maximum execution budget of **50 steps**. The model generates five five-step chunks, executes the buffered plan, and replans when the buffer is empty; an episode can terminate earlier on success. No candidate-action search or high-level planner is used.

```bash
python eval_pusht.py --checkpoint runs/pusht/weights_epoch_20.pt \
  --dataset-root data --result-dir results/pusht_epoch20 --no-video
python eval_tasks.py --task tworoom --checkpoint runs/tworoom/weights_epoch_10.pt \
  --dataset-root data --run-dir results/tworoom_epoch10
python eval_tasks.py --task cube --checkpoint runs/cube/weights_epoch_10.pt \
  --dataset-root data --run-dir results/cube_epoch10
```

The epoch numbers above are command examples, not instructions to use the final checkpoint as the best checkpoint. Evaluate the desired checkpoints and identify the selected epoch in any reported result. Results are written to JSON, with per-case success flags and the case-selection metadata. `success_rate` follows stable-worldmodel's percentage scale (0–100).

Evaluation preserves the benchmark's original `len(valid)-1` case-sampling convention. Simulator state is used to restore initial conditions and goals, not as network input. The network receives images and agent/robot proprioception. Cube preserves the seven proprioceptive column names and their concatenation order from training.

`--inference-batch-size` controls only how many environments are forwarded through the network together (default 8); it does not change the number of evaluated environments. TwoRoom/Cube validate the initial proprioception and cross-check terminal success against the physical environment state.

## Smoke tests

```bash
python smoke_test.py --dataset-root data --batch-size 2
python train_pusht.py --dataset-root data --run-dir runs/smoke_pusht \
  --epochs 1 --batch-size 2 --num-workers 0 --limit-train-batches 1 --val-batches 1
python train_tasks.py --task tworoom --dataset-root data --run-dir runs/smoke_tworoom \
  --batch-size 2 --num-workers 0 --smoke-batches 1
python train_tasks.py --task cube --dataset-root data --run-dir runs/smoke_cube \
  --batch-size 2 --num-workers 0 --smoke-batches 1
```

Smoke checkpoints are marked and refused for normal benchmark scoring. `--allow-smoke` exists only for checking the evaluation pipeline. See [verification notes](docs/VERIFICATION.md) for the checks performed on this export.

## Source map

| File | Role |
|---|---|
| `wa_jepa_v4.py` | Frozen visual encoder, representation fusion, action proposals, world model, two feedback paths |
| `wa_policy_v4.py` | Observation history and action buffering for environment execution |
| `train_pusht.py` | Shared training losses and PushT training |
| `train_tasks.py` | TwoRoom/Cube data preparation, original split and training loop |
| `eval_pusht.py` / `eval_tasks.py` | Short-horizon environment evaluation |
| `runtime_utils.py` | Extracted dataset, hashing, microbatching, and no-SIGReg helpers |
| `task_evaluation.py` | Per-epoch evaluation subprocess |
| `config/eval/` | TwoRoom/Cube evaluation protocols |
| `prepare_pusht.py` / `smoke_test.py` | Data conversion and integration checks |

Legacy V2/V3 implementations are not required. The `V4` class names and checkpoint format are retained for compatibility with existing V4-1 weights.

## Acknowledgments

This release was extracted from the WA-JEPA implementation developed in the [LeWorldModel](https://github.com/lucas-maes/le-wm) codebase. It uses [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel), [stable-pretraining](https://github.com/galilai-group/stable-pretraining), and [DINOv2](https://github.com/facebookresearch/dinov2). The existing upstream MIT notice is preserved in `LICENSE`; external dependencies and pretrained assets retain their respective licenses. See [third-party notices](THIRD_PARTY_NOTICES.md).
