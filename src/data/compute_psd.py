"""
Compute and save the avreage Power Spectral Density (PSD) from noise samples.

This script must be run ONCE before training the 1D CNN model.
It computes the average noise spectrum which is used for whitening.

Usage:
    python src/data/compute_psd.py

The output is saved to the dataset directory as 'avg_psd.npz'.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from tqdm import tqdm

from data.g2net import find_dataset_dir, load_labels, load_sample, is_kaggle, get_output_dir
from data.preprocessing import compute_psd, save_psd, N_FREQ


def compute_and_save_average_psd(n_samples: int = 50000, output_path: Path = None, dataset_dir: Path = None) -> np.ndarray:
    """
    Compute average PSD from noise-only samples and save to disk.

    Parameters
    ----------
    n_samples : int
        Number of noise samples to use (more = better estimate, but slower)
    output_path : Path, optional
        Where to save the PSD. Defaults to dataset_dir/avg_psd.npz
    dataset_dir : Path, optional
        Path to G2Net dataset. If None, uses find_dataset_dir().

    Returns
    -------
    avg_psd : np.ndarray
        Average PSD, shape (3, N_FREQ)
    """
    print("=" * 60)
    print("COMPUTING AVERAGE PSD FROM NOISE SAMPLES")
    print("=" * 60)

    # find dataset
    if dataset_dir is None:
        dataset_dir = find_dataset_dir()
    output_dir = get_output_dir()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'}")

    if output_path is None:
        output_path = output_dir / "avg_psd.npz"

    # load labels
    df = load_labels(dataset_dir)
    print(f"Total samples: {len(df)}")

    # select just the noise-only samples
    noise_df = df[df['target'] == 0]
    print(f"Noise-only samples: {len(noise_df)}")

    # limit to n_samples
    if len(noise_df) > n_samples:
        noise_df = noise_df.sample(n=n_samples, random_state=42)
    actual_n = len(noise_df)
    print(f"Using {actual_n} samples for PSD estimation")

    # accumulate PSDs
    print("\nComputing PSDs...")
    psd_sum = np.zeros((3, N_FREQ), dtype=np.float64)   # 3 detectors
    count = 0

    for sample_id in tqdm(noise_df['id'].astype(str).values, desc="Processing"):
      sample = load_sample(sample_id, dataset_dir=dataset_dir)
      psd = compute_psd(sample)
      psd_sum += psd
      count += 1

    # compute average
    avg_psd = psd_sum / count
    print(f"\nComputed average from {count} samples")

    # apply smoothing to reduce variance
    print("Applying smoothing...")
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    for det in range(3):
        avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    # ensure no zeros
    avg_psd = np.maximum(avg_psd, 1e-50)

    # save
    save_psd(avg_psd, output_path)

    # print some stats
    print("\nPSD Statistics:")
    detector_names = ['H1 (Hanford)', 'L1 (Livingston)', 'V1 (Virgo)']
    for i, name in enumerate(detector_names):
        print(f"  {name}:")
        print(f"    Min:  {avg_psd[i].min():.2e}")
        print(f"    Max:  {avg_psd[i].max():.2e}")
        print(f"    Mean: {avg_psd[i].mean():.2e}")

    print("\n" + "=" * 60)
    print("PSD COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"\nSaved to: {output_path}")
    print("\nYou can now run the training pipeline.")

    return avg_psd


if __name__ == "__main__":
    compute_and_save_average_psd(n_samples=10000)
