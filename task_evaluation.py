"""Run per-epoch evaluation in a subprocess, without perturbing training RNG."""

import json
from pathlib import Path
import subprocess
import sys


def evaluate_checkpoint(task, checkpoint, root, epoch, dataset_root):
    folder = Path(root) / f"eval_epoch_{epoch}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("eval_tasks.py")),
        "--task",
        task,
        "--checkpoint",
        str(checkpoint),
        "--run-dir",
        str(folder),
        "--dataset-root",
        str(dataset_root),
        "--inference-batch-size",
        "8",
    ]
    with (Path(root) / f"eval_epoch_{epoch}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((folder / "results.json").read_text())
    with (Path(root) / "evaluations.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "epoch": epoch,
                    "success_rate": result["success_rate"],
                    "num_eval": len(result["episode_successes"]),
                }
            )
            + "\n"
        )
    print(f"Evaluation epoch {epoch}: {result['success_rate']:.2f}%", flush=True)
