# Export verification

Checked on 2026-11-11 in a temporary directory on an NVIDIA A40, using the existing Python 3.10 experiment environment. Original code, datasets, and checkpoints were read; existing runs were not modified.

| Check | PushT | TwoRoom | Cube |
|---|---|---|---|
| Real-data minibatch loss and backward (batch 2) | Passed | Passed | Passed |
| Finite gradients and frozen DINO parameters | Passed | Passed | Passed |
| Optimizer update and 25-step action output | Passed | Passed | Passed |
| Checkpoint round-trip: identical planned actions | Passed | Passed | Passed |
| Training CLI: one training + validation batch and checkpoint save | Passed | Passed | Passed |
| Existing checkpoint evaluation: 2 environment cases | Passed | Passed | Passed |

Additional checks:

- Python syntax and undefined-name checks passed.
- Network computation methods match the original source at the Python AST level; checkpoint asset-path relocation is the only changed model method.
- The shared loss, history augmentation, Lightning train/validation steps, and optimizer configuration match the original V4 trainer. V4-1's zero-SIGReg module is substituted at initialization.
- A synthetic HDF5 dataset was converted to Lance and read back successfully.
- Installed stable-worldmodel and stable-pretraining Python files match their package RECORD hashes; no undocumented package patch is required by these checks.
- No private absolute server paths, checkpoints, datasets, or unrelated experiment code are included.

These are integration checks, not a rerun of training convergence or the full 50-case benchmark. Small test results must not be used as paper accuracy numbers. Fresh-environment installation and full retraining have not been performed.

## Portability changes

DINO and dataset paths are configurable. Historical server-log checks and scheduling locks were removed. Required helper functions were extracted instead of copying legacy architectures. Evaluation defaults to executing all five chunks, matching the short-horizon V4-1 protocol. Microbatched inference controls memory usage. Output directories are protected against overwrite.

For a one-step smoke run only, the TwoRoom/Cube scheduler receives a minimum two-step horizon to avoid a zero cosine denominator. Standard 10-epoch schedules are unaffected.
