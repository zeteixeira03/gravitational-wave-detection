"""Download a class-balanced subset of raw G2Net files with the Kaggle API.

The full raw set is about 70 GB, so local work uses subsets. The subset is
deterministic (seed) and its id list is written next to the data, so a later
session can reproduce it.

Usage:
    KAGGLE_API_TOKEN=... python -m gwsciml.data.download --split val --n-per-class 2500
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gwsciml.data.split import COMPETITION, balanced_subset, load_labels, sample_path


def _fetch(api, sample_id: str, data_dir: Path, retries: int = 4) -> bool:
    dest = sample_path(sample_id, data_dir)
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote = f"train/{sample_id[0]}/{sample_id[1]}/{sample_id[2]}/{sample_id}.npy"
    for _ in range(retries):
        try:
            api.competition_download_file(COMPETITION, remote, path=str(dest.parent), quiet=True)
            if dest.exists():
                return True
        except Exception:
            pass
    return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/g2net"))
    p.add_argument("--split", choices=["val", "train"], required=True)
    p.add_argument("--n-per-class", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    # kaggle>=2 reads new-style tokens from KAGGLE_API_TOKEN
    if "KAGGLE_API_TOKEN" not in os.environ and os.environ.get("KAGGLE_KEY", "").startswith("KGAT"):
        os.environ["KAGGLE_API_TOKEN"] = os.environ["KAGGLE_KEY"]
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    df = load_labels(args.data_dir)
    sub = balanced_subset(df, args.split, args.n_per_class, args.seed)
    list_path = args.data_dir / f"subset_{args.split}_{args.n_per_class}x2_seed{args.seed}.csv"
    sub[["id", "target"]].to_csv(list_path, index=False)

    failed = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_fetch, api, sid, args.data_dir): sid for sid in sub["id"]}
        for i, f in enumerate(as_completed(futs), 1):
            if not f.result():
                failed.append(futs[f])
            if i % 500 == 0:
                print(f"{i}/{len(sub)} done, {len(failed)} failed", flush=True)
    print(f"finished: {len(sub) - len(failed)} ok, {len(failed)} failed; list at {list_path}")


if __name__ == "__main__":
    main()
