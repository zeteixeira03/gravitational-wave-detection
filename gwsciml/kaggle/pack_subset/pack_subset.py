"""Kaggle kernel: pack a class-balanced subset of raw G2Net train samples.

Why: the competition API rate-limits single-file downloads (HTTP 429), so the
cloud session cannot fetch thousands of .npy files one by one. This kernel
reads the files inside Kaggle and writes a few large arrays to /kaggle/working.

Split definition (ROADMAP.md): val = rows 0..149,999 of training_labels.csv,
train = rows 150,000..559,999. Each output chunk holds raw float64 strain with
shape (n, 3, 4096) and a CSV with the ids and targets in the same order.
"""

import glob
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

N_VAL_ROWS = 150_000
N_PER_CLASS = 5000
CHUNK = 5000
SEED = 0

# depth-limited lookup: a recursive walk of the mount visits ~786k files
root = Path(sorted(glob.glob("/kaggle/input/*/training_labels.csv")
                   + glob.glob("/kaggle/input/*/*/training_labels.csv"))[0]).parent
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
        paths = [root / "train" / s[0] / s[1] / s[2] / f"{s}.npy" for s in rows["id"]]
        with ThreadPoolExecutor(32) as ex:  # the input mount is latency-bound
            x = np.stack(list(ex.map(np.load, paths)))
        np.save(out / f"{split}_{c}.npy", x)
        rows[["id", "target"]].to_csv(out / f"{split}_{c}.csv", index=False)
        print(time.strftime("%H:%M:%S"), split, c, x.shape, x.dtype, flush=True)
print("done")
