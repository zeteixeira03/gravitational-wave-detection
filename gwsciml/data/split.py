"""Fixed train/val split of the G2Net training set.

The split is the one the frozen baseline used (src/model_runs.py, shard mode):
shards are written in training_labels.csv row order, and validation takes the
first shards until 20 % of the samples is reached. With 50,000 samples per
shard that is the first 3 shards, i.e. rows 0 .. 149,999 of the CSV (which is
sorted by id). All later comparisons use this definition.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_VAL_ROWS = 150_000
COMPETITION = "g2net-gravitational-wave-detection"


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Read training_labels.csv and add a 'split' column ('val' or 'train')."""
    df = pd.read_csv(Path(data_dir) / "training_labels.csv")
    df["split"] = np.where(np.arange(len(df)) < N_VAL_ROWS, "val", "train")
    return df


def sample_path(sample_id: str, data_dir: Path, subset: str = "train") -> Path:
    """G2Net file layout: <subset>/<id[0]>/<id[1]>/<id[2]>/<id>.npy."""
    return Path(data_dir) / subset / sample_id[0] / sample_id[1] / sample_id[2] / f"{sample_id}.npy"


def balanced_subset(df: pd.DataFrame, split: str, n_per_class: int, seed: int = 0) -> pd.DataFrame:
    """Draw n_per_class positives and n_per_class negatives from one split, without replacement."""
    rng = np.random.default_rng(seed)
    parts = []
    for target in (0, 1):
        pool = df[(df["split"] == split) & (df["target"] == target)]
        idx = rng.choice(len(pool), size=n_per_class, replace=False)
        parts.append(pool.iloc[np.sort(idx)])
    return pd.concat(parts).sort_values("id").reset_index(drop=True)
