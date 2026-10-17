"""Load raw G2Net samples from disk.

Two layouts are supported:
  * packed chunks written by gwsciml/kaggle/pack_subset: <name>_<c>.npy with
    shape (n, 3, 4096) float64 and <name>_<c>.csv with columns id, target;
  * the competition layout train/<a>/<b>/<c>/<id>.npy (one sample per file).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from gwsciml.data.split import sample_path


def load_packed(data_dir: Path, name: str, mmap: bool = True) -> tuple[np.ndarray, pd.DataFrame]:
    """Concatenate all packed chunks <name>_<c>.npy / .csv in data_dir, in chunk order."""
    data_dir = Path(data_dir)
    chunks = sorted(data_dir.glob(f"{name}_*.npy"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not chunks:
        raise FileNotFoundError(f"no {name}_*.npy in {data_dir}")
    xs = [np.load(p, mmap_mode="r" if mmap else None) for p in chunks]
    meta = pd.concat([pd.read_csv(p.with_suffix(".csv")) for p in chunks], ignore_index=True)
    x = xs[0] if len(xs) == 1 else np.concatenate(xs, axis=0)
    if len(x) != len(meta):
        raise ValueError(f"{name}: {len(x)} arrays but {len(meta)} label rows")
    return x, meta


def load_files(ids, data_dir: Path) -> np.ndarray:
    """Stack single-file samples (competition layout) into (n, 3, 4096) float64."""
    return np.stack([np.load(sample_path(i, data_dir)) for i in ids])
