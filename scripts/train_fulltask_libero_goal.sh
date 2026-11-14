#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/fulltask_env.sh"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CACHE_DIR RUN_DIR [trainer options]" >&2
  exit 2
fi
cache="$1"; run="$2"; shift 2
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  -m fulltask.train_libero --cache "$cache" --run-dir "$run" \
  --epochs 400 --batch-per-gpu 256 --seed "${SEED:-42}" "$@"
