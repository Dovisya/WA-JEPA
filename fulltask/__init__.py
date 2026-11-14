"""Full-task training and evaluation for WA-JEPA."""
import os
import sys
from pathlib import Path

# Keep the upstream environment/VQ implementation external and version-pinned.
patch = Path(os.environ.get('PATCH_POLICY_ROOT',
             str(Path(__file__).resolve().parents[1] / 'third_party' / 'patch_policy')))
if patch.is_dir():
    sys.path.insert(0, str(patch.resolve()))
