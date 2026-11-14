#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/fulltask_env.sh"
commit=674b09d9de0a7ea6226583c13078b4685b358485
if [[ ! -d "$PATCH_POLICY_ROOT" ]]; then
  mkdir -p "$(dirname "$PATCH_POLICY_ROOT")"
  git clone https://github.com/gaoyuezhou/patch_policy.git "$PATCH_POLICY_ROOT"
  git -C "$PATCH_POLICY_ROOT" checkout --detach "$commit"
fi
if [[ "$(git -C "$PATCH_POLICY_ROOT" rev-parse HEAD)" != "$commit" ]]; then
  echo "Expected Patch Policy commit $commit. Use a separate checkout at that revision." >&2
  exit 1
fi
python -m pip install -r requirements-fulltask.txt
