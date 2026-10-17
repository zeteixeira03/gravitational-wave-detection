"""Run the P1 classical detector on a packed split and log the result.

Usage:
  python -m gwsciml.detect.run_p1 --split val --bank data/bank/<name>.npz
Writes data/results/p1_<split>_<tag>.csv (per-sample statistics, git-ignored)
and appends one row to results/runs.jsonl (committed, append-only).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from gwsciml.constants import N
from gwsciml.data.io import load_packed
from gwsciml.data.psd import tukey
from gwsciml.detect.classical import ClassicalDetector, DetectorConfig
from gwsciml.templates.bank import FREQS, duration_limited_f_start
from gwsciml.templates.waveforms import lal_fd_waveform

SCORES = ["inc_max", "inc_lse", "coh_max", "rw_net2"]


def bank_fd(params, approximant, max_duration, f_lower=20.0):
    return np.stack([
        lal_fd_waveform(FREQS, m1, m2, s1, s2, approximant=approximant,
                        f_lower=float(duration_limited_f_start(m1, m2, max_duration, f_lower)))
        for m1, m2, s1, s2 in params])


def auc_with_ci(y, s, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    a = roc_auc_score(y, s)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(roc_auc_score(y[i], s[i]))
    return float(a), float(np.std(boots))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--packed", type=Path, default=Path("data/g2net/packed"))
    ap.add_argument("--psd", type=Path, default=None)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--t-lo", type=float, default=0.1)
    ap.add_argument("--t-hi", type=float, default=1.9)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-log", action="store_true", help="smoke test: do not append to results/runs.jsonl")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    bank = np.load(args.bank)
    bcfg = json.loads(str(bank["config"]))
    psd = np.load(args.psd or args.packed / "psd_avg_train_neg_tukey0.1.npy")
    cfg = DetectorConfig(t_range=(args.t_lo, args.t_hi))
    det = ClassicalDetector(bank_fd(bank["params"], bcfg["approximant"], bcfg["max_duration"]), psd, tukey(N, 0.1), cfg)

    x, meta = load_packed(args.packed, args.split)
    if args.limit:
        x, meta = x[: args.limit], meta.iloc[: args.limit]
    t0 = time.time()
    rows = []
    out_dir = Path("data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.bank.stem
    lmaps = np.lib.format.open_memmap(out_dir / f"p1_{args.split}_{tag}_lmap.npy", mode="w+", dtype=np.float32,
                                      shape=(len(x), det.hw.shape[1], det.n_time_bins))
    for i in range(len(x)):
        st, lmaps[i] = det.statistics(np.asarray(x[i]), return_map=True)
        rows.append(st)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(x)}  {(time.time() - t0) / (i + 1):.3f} s/sample", flush=True)
    lmaps.flush()
    res = pd.concat([meta.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    res.to_csv(out_dir / f"p1_{args.split}_{tag}.csv", index=False)

    y = res.target.values
    aucs = {s: auc_with_ci(y, res[s].values) for s in SCORES}
    for s, (a, sd) in aucs.items():
        print(f"AUC {s:10s} {a:.4f} +- {sd:.4f}")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": "P1", "git": git, "split": args.split,
        "n": int(len(res)), "pos_frac": float(y.mean()), "bank": args.bank.name, "bank_config": bcfg,
        "detector_config": vars(cfg), "auc": {s: a for s, (a, _) in aucs.items()},
        "auc_bootstrap_sd": {s: sd for s, (_, sd) in aucs.items()},
        "seconds_per_sample": (time.time() - t0) / len(res), "per_sample_csv": str(out_dir / f"p1_{args.split}_{tag}.csv"),
    }
    if args.no_log:
        return
    Path("results").mkdir(exist_ok=True)
    with open("results/runs.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
