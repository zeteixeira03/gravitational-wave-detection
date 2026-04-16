"""
Create PyTorch tensor dataset with preprocessed (whitened) signals and S2 sky features.

Run this locally once to generate .pt shard files, then upload to Kaggle as a dataset.

Usage:
    python src/data/create_tensors.py --input /path/to/g2net-dataset --output /path/to/output

The output directory will contain:
    - shard_00.pt, shard_01.pt, ... (preprocessed signals + labels + sh_coeffs)
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
from sky_feasibility import SkyGeometry

SHARD_SIZE = 50000  # ~2.4 GB per shard


# ============================================================================
#                                 MAIN
# ============================================================================

def _save_shard(signals_list, labels_list, sh_coeffs_list, output_dir, shard_idx):
    """Save a shard to disk and clear the lists."""
    signals = torch.tensor(np.stack(signals_list))
    labels = torch.tensor(np.array(labels_list, dtype=np.int64))
    sh_coeffs = torch.tensor(np.stack(sh_coeffs_list), dtype=torch.float32)
    path = output_dir / f"shard_{shard_idx:02d}.pt"
    torch.save({'signals': signals, 'labels': labels, 'sh_coeffs': sh_coeffs}, str(path))
    print(f"\n  Saved {path.name} ({len(labels_list)} samples)")
    return path


def create_tensors(input_dir: Path, output_dir: Path, psd_path: Path = None):
    """
    Create PyTorch tensor dataset from raw G2Net data.

    Writes sharded .pt files to keep memory usage constant (~2.4 GB per shard).
    Each shard contains preprocessed signals, labels, and S2 spherical harmonic
    coefficients.

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

    # sky geometry
    n_pix, l_max = 192, 10
    print(f"\nInitializing S2 sky geometry (n_pix={n_pix}, l_max={l_max})...")
    sky_geo = SkyGeometry(n_pix=n_pix, l_max=l_max)
    print(f"  SH coefficients per sample: {sky_geo.n_coeffs}")

    # process and save in shards
    print(f"\nProcessing samples (shard size: {SHARD_SIZE})...")
    signals_list = []
    labels_list = []
    sh_coeffs_list = []
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
            sh_coeffs_list.append(sky_geo.extract(processed))

            written += 1

            # flush shard to disk when full
            if len(labels_list) >= SHARD_SIZE:
                path = _save_shard(signals_list, labels_list, sh_coeffs_list, output_dir, shard_idx)
                shard_paths.append(path.name)
                signals_list.clear()
                labels_list.clear()
                sh_coeffs_list.clear()
                shard_idx += 1

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\nError processing {sample_id}: {e}")

    # save remaining samples
    if labels_list:
        path = _save_shard(signals_list, labels_list, sh_coeffs_list, output_dir, shard_idx)
        shard_paths.append(path.name)

    # save metadata
    metadata = {
        'n_samples': written,
        'n_failed': failed,
        'n_shards': len(shard_paths),
        'shard_files': shard_paths,
        'signal_shape': [3, 4096],
        'dtype': 'float32',
        'sky_n_coeffs': sky_geo.n_coeffs,
        'sky_n_pix': n_pix,
        'sky_l_max': l_max,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nDone!")
    print(f"  Written: {written}")
    print(f"  Failed: {failed}")
    print(f"  Shards: {len(shard_paths)}")
    print(f"  Metadata: {metadata_path}")


def add_sky_features(shard_dir: Path, n_pix: int = 192, l_max: int = 10) -> None:
    """
    Add SH coefficients to existing shard files that lack them.

    Reads each shard's preprocessed signals, computes sky maps and SH
    coefficients, and rewrites the shard with sh_coeffs included.
    Much faster than regenerating from raw data (~1.5 ms/sample vs minutes
    for full preprocessing).

    Parameters
    ----------
    shard_dir
        Directory containing shard_*.pt files.
    n_pix
        Sky grid resolution.
    l_max
        Maximum SH degree.
    """
    shard_files = sorted(shard_dir.glob('shard_*.pt'))
    if not shard_files:
        raise FileNotFoundError(f"No shard files in {shard_dir}")

    # skip only if shards already contain sh_coeffs at the requested l_max.
    # different l_max -> recompute and overwrite.
    target_n_coeffs = (l_max + 1) ** 2
    first = torch.load(str(shard_files[0]), weights_only=True)
    if 'sh_coeffs' in first and first['sh_coeffs'].shape[1] == target_n_coeffs:
        print(f"Shards already contain sh_coeffs at l_max={l_max} ({target_n_coeffs} coefs), skipping.")
        del first
        return
    if 'sh_coeffs' in first:
        existing = first['sh_coeffs'].shape[1]
        print(f"Overwriting sh_coeffs: existing has {existing} coefs, target l_max={l_max} -> {target_n_coeffs} coefs.")
    del first

    print(f"Adding SH coefficients to {len(shard_files)} shards (n_pix={n_pix}, l_max={l_max})...")
    sky_geo = SkyGeometry(n_pix=n_pix, l_max=l_max)

    for f in shard_files:
        data = torch.load(str(f), weights_only=True)
        signals = data['signals'].numpy()
        n = len(signals)
        sh_coeffs = np.empty((n, sky_geo.n_coeffs), dtype=np.float32)

        for i in tqdm(range(n), desc=f"  {f.name}"):
            sh_coeffs[i] = sky_geo.extract(signals[i])

        data['sh_coeffs'] = torch.tensor(sh_coeffs)
        torch.save(data, str(f))
        print(f"    {f.name}: {n} samples")
        del data, signals, sh_coeffs

    # update metadata
    meta_path = shard_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as fh:
            metadata = json.load(fh)
        metadata['sky_n_coeffs'] = (l_max + 1) ** 2
        metadata['sky_n_pix'] = n_pix
        metadata['sky_l_max'] = l_max
        with open(meta_path, 'w') as fh:
            json.dump(metadata, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create PyTorch tensor dataset from G2Net data")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to g2net-gravitational-wave-detection directory (for full rebuild)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save tensor files (for full rebuild)")
    parser.add_argument("--psd", type=str, default=None,
                        help="Path to precomputed avg_psd.npz (optional)")
    parser.add_argument("--add-sky", type=str, default=None,
                        help="Path to existing shard directory to add SH coefficients to")
    parser.add_argument("--l-max", type=int, default=10,
                        help="Maximum SH degree (used with --add-sky). Default 10.")

    args = parser.parse_args()

    if args.add_sky:
        add_sky_features(Path(args.add_sky), l_max=args.l_max)
    elif args.input and args.output:
        create_tensors(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            psd_path=Path(args.psd) if args.psd else None,
        )
    else:
        parser.error("Either --add-sky or both --input and --output are required")
