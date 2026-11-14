#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/fulltask_env.sh"
if [[ $# != 2 ]]; then echo "Usage: $0 pusht|libero_goal CACHE_DIR" >&2; exit 2; fi
case "$1" in
  pusht)
    python -m fulltask.prepare --cache "$2"
    python -m fulltask.codebook --cache "$2" ;;
  libero_goal)
    python -m fulltask.libero_prepare --cache "$2"
    python -m fulltask.libero_prepare --cache "$2" --finalize
    python -m fulltask.libero_codebook --cache "$2" ;;
  *) echo "Unknown task: $1" >&2; exit 2 ;;
esac
