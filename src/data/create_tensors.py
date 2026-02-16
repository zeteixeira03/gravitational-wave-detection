"""
Create PyTorch tensor dataset with preprocessed (whitened) signals.

Run this locally once to generate .pt shard files, then upload to Kaggle as a dataset.

Usage:
    python src/data/create_tensors.py --input /path/to/g2net-dataset --output /path/to/output

The output directory will contain:
    - shard_00.pt, shard_01.pt, ... (preprocessed signals + labels)
    - metadata.json (sample count, shard info)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import load_labels, load_sample
from data.preprocessing import preprocess_sample, load_psd
from data.compute_psd import compute_and_save_average_psd

SHARD_SIZE = 50000  # ~2.4 GB per shard


# ============================================================================
#                                 MAIN
# ============================================================================

def _save_shard(signals_list, labels_list, output_dir, shard_idx):
    """Save a shard to disk and clear the lists."""
    signals = torch.tensor(np.stack(signals_list))
    labels = torch.tensor(np.array(labels_list, dtype=np.int64))
    path = output_dir / f"shard_{shard_idx:02d}.pt"
    torch.save({'signals': signals, 'labels': labels}, str(path))
    print(f"\n  Saved {path.name} ({len(labels_list)} samples)")
    return path


def create_tensors(input_dir: Path, output_dir: Path, psd_path: Path = None):
    """
    Create PyTorch tensor dataset from raw G2Net data.

    Writes sharded .pt files to keep memory usage constant (~2.4 GB per shard).

    Parameters
    ----------
    input_dir
        Path to g2net-gravitational-wave-detection directory (contains train/, training_labels.csv)
    output_dir
        Path to save tensor files
    psd_path
        Path to precomputed PSD file. If None, computes from noise samples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # load or compute PSD
    if psd_path and psd_path.exists():
        print(f"Loading PSD from: {psd_path}")
        avg_psd = load_psd(psd_path)
    else:
        print("Computing average PSD from noise samples...")
        psd_path = output_dir / "avg_psd.npz"
        avg_psd = compute_and_save_average_psd(
            n_samples=10000, output_path=psd_path, dataset_dir=input_dir
        )

    # load labels
    print(f"\nLoading labels from: {input_dir}")
    df = load_labels(input_dir)
    n_samples = len(df)
    print(f"Total samples: {n_samples}")

    # process and save in shards
    print(f"\nProcessing samples (shard size: {SHARD_SIZE})...")
    signals_list = []
    labels_list = []
    shard_idx = 0
    shard_paths = []
    written = 0
    failed = 0

    for _, row in tqdm(df.iterrows(), total=n_samples, desc="Processing"):
        sample_id = str(row['id'])
        label = int(row['target'])

        try:
            raw = load_sample(sample_id, dataset_dir=input_dir)
            processed = preprocess_sample(raw, avg_psd)

            if np.isnan(processed).any() or np.isinf(processed).any():
                failed += 1
                continue

            signals_list.append(processed.astype(np.float32))
            labels_list.append(label)
            written += 1

            # flush shard to disk when full
            if len(labels_list) >= SHARD_SIZE:
                path = _save_shard(signals_list, labels_list, output_dir, shard_idx)
                shard_paths.append(path.name)
                signals_list.clear()
                labels_list.clear()
                shard_idx += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\nError processing {sample_id}: {e}")

    # save remaining samples
    if labels_list:
        path = _save_shard(signals_list, labels_list, output_dir, shard_idx)
        shard_paths.append(path.name)

    # save metadata
    metadata = {
        'n_samples': written,
        'n_failed': failed,
        'n_shards': len(shard_paths),
        'shard_files': shard_paths,
        'signal_shape': [3, 4096],
        'dtype': 'float32',
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Written: {written}")
    print(f"  Failed: {failed}")
    print(f"  Shards: {len(shard_paths)}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PyTorch tensor dataset from G2Net data")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to g2net-gravitational-wave-detection directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save tensor files")
    parser.add_argument("--psd", type=str, default=None,
                        help="Path to precomputed avg_psd.npz (optional)")

    args = parser.parse_args()

    create_tensors(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        psd_path=Path(args.psd) if args.psd else None,
    )
