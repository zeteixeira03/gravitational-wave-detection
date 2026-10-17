"""Kaggle kernel: pack a class-balanced subset of raw G2Net train samples.

Why: the competition API rate-limits single-file downloads (HTTP 429), so the
cloud session cannot fetch thousands of .npy files one by one. This kernel
reads the files inside Kaggle and writes a few large arrays to /kaggle/working.

Split definition (ROADMAP.md): val = rows 0..149,999 of training_labels.csv,
train = rows 150,000..559,999. Each output chunk holds raw float64 strain with
shape (n, 3, 4096) and a CSV with the ids and targets in the same order.
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_VAL_ROWS = 150_000
N_PER_CLASS = 5000
CHUNK = 5000
SEED = 0

root = next(p.parent for p in Path("/kaggle/input").rglob("training_labels.csv"))
print("competition root:", root)
df = pd.read_csv(root / "training_labels.csv")
df["split"] = np.where(np.arange(len(df)) < N_VAL_ROWS, "val", "train")

out = Path("/kaggle/working")
for split in ("val", "train"):
    rng = np.random.default_rng(SEED)
    parts = []
    for target in (0, 1):
        pool = df[(df["split"] == split) & (df["target"] == target)]
        idx = rng.choice(len(pool), size=N_PER_CLASS, replace=False)
        parts.append(pool.iloc[np.sort(idx)])
    sub = pd.concat(parts).sort_values("id").reset_index(drop=True)
    for c, start in enumerate(range(0, len(sub), CHUNK)):
        rows = sub.iloc[start:start + CHUNK]
        x = np.empty((len(rows), 3, 4096), dtype=np.float64)
        for i, sid in enumerate(rows["id"]):
            x[i] = np.load(root / "train" / sid[0] / sid[1] / sid[2] / f"{sid}.npy")
        np.save(out / f"{split}_{c}.npy", x)
        rows[["id", "target"]].to_csv(out / f"{split}_{c}.csv", index=False)
        print(split, c, x.shape, flush=True)
print("done")
