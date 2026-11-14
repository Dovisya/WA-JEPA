#!/usr/bin/env bash
# Source from another launcher; user paths always take precedence.
WA_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATCH_POLICY_ROOT="${PATCH_POLICY_ROOT:-$WA_REPO_ROOT/third_party/patch_policy}"
export DINO_REPO="${DINO_REPO:-$WA_REPO_ROOT/third_party/dinov2}"
export DINO_CHECKPOINT="${DINO_CHECKPOINT:-$WA_REPO_ROOT/pretrained/dinov2_vits14_pretrain.pth}"
export PUSHT_DATA_ROOT="${PUSHT_DATA_ROOT:-$WA_REPO_ROOT/data/fulltask/pusht_dataset}"
export LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-$WA_REPO_ROOT/data/fulltask/libero_dataset/libero_goal}"
export PYTHONPATH="$WA_REPO_ROOT:$PATCH_POLICY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$WA_REPO_ROOT/.libero_fulltask}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=disabled
cd "$WA_REPO_ROOT"
