"""
Build a small exploration sample from the full G2Net dataset.

Selects 1000 samples (500 signal + 500 noise), copies them with the original
nested directory layout, and bundles avg_psd.npz and the latest trained model.
The output is ready to upload as a Kaggle dataset.

Usage
-----
    python src/data/create_sample_dataset.py \
        --input  D:/Programming/g2net-gravitational-wave-detection \
        --output D:/Programming/g2net-exploration-sample
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow running as `python src/data/create_sample_dataset.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g2net import load_labels, sample_path, find_project_root


# ============================================================================================
#                                     configuration
# ============================================================================================

N_PER_CLASS = 500
RANDOM_SEED = 42


# ============================================================================================
#                                       main
# ============================================================================================

def create_sample_dataset(input_dir: Path, output_dir: Path) -> None:
    project_root = find_project_root()

    # load full labels
    labels_df = load_labels(input_dir)
    print(f"Full dataset: {len(labels_df)} samples")

    # stratified sample
    rng = np.random.RandomState(RANDOM_SEED)
    signal_ids = labels_df[labels_df["target"] == 1].sample(n=N_PER_CLASS, random_state=rng)
    noise_ids = labels_df[labels_df["target"] == 0].sample(n=N_PER_CLASS, random_state=rng)
    subset = pd.concat([signal_ids, noise_ids]).sort_values("id").reset_index(drop=True)
    print(f"Selected {len(subset)} samples ({N_PER_CLASS} signal + {N_PER_CLASS} noise)")

    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # write subset labels
    subset.to_csv(output_dir / "training_labels.csv", index=False)
    print(f"Wrote training_labels.csv")

    # copy .npy files preserving nested layout
    train_in = input_dir / "train"
    train_out = output_dir / "train"
    copied = 0
    for _, row in subset.iterrows():
        src = sample_path(row["id"], train_in)
        dst = sample_path(row["id"], train_out)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        if copied % 200 == 0:
            print(f"  copied {copied}/{len(subset)} samples")
    print(f"Copied {copied} .npy files")

    # copy avg_psd.npz
    psd_candidates = [
        project_root / "kaggle" / "output" / "avg_psd.npz",
        project_root / "avg_psd.npz",
    ]
    for psd_path in psd_candidates:
        if psd_path.exists():
            shutil.copy2(psd_path, output_dir / "avg_psd.npz")
            print(f"Copied avg_psd.npz from {psd_path}")
            break
    else:
        print("WARNING: avg_psd.npz not found, skipping")

    # copy latest model weights + config + metrics
    models_dir = project_root / "kaggle" / "output" / "models" / "saved"
    if models_dir.exists():
        model_out = output_dir / "models" / "saved"
        model_out.mkdir(parents=True, exist_ok=True)

        # find latest model by date in filename
        weight_files = sorted(models_dir.glob("diy_*_weights.npz"))
        if weight_files:
            latest = weight_files[-1]
            date_tag = latest.stem.replace("diy_", "").replace("_weights", "")

            for suffix in ["_weights.npz", "_config.json", "_metrics.json"]:
                src = models_dir / f"diy_{date_tag}{suffix}"
                if src.exists():
                    shutil.copy2(src, model_out / src.name)
            print(f"Copied model diy_{date_tag}")
        else:
            print("WARNING: no model weights found, skipping")
    else:
        print("WARNING: models directory not found, skipping")

    # summary
    total_size_mb = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"\nDone. Output: {output_dir}")
    print(f"Total size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build G2Net exploration sample dataset")
    parser.add_argument("--input", type=Path, required=True, help="path to full G2Net dataset")
    parser.add_argument("--output", type=Path, required=True, help="output directory for sample dataset")
    args = parser.parse_args()

    create_sample_dataset(args.input, args.output)
