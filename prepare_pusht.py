"""Convert the PushT HDF5 training data to the Lance reader used for training."""

import argparse
from pathlib import Path
import hdf5plugin  # noqa: F401
from stable_worldmodel.data import convert

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--jpeg-quality", type=int, default=95)
    a = p.parse_args()
    root = Path(a.dataset_root) / "datasets"
    source = root / "pusht_expert_train.h5"
    dest = root / "pusht_expert_train.lance"
    if not source.is_file():
        raise FileNotFoundError(source)
    if dest.exists():
        raise FileExistsError(dest)
    convert(
        str(source),
        str(dest),
        dest_format="lance",
        jpeg_quality=a.jpeg_quality,
        mode="error",
    )
