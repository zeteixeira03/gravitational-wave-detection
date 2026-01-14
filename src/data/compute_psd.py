"""
Compute and save the average Power Spectral Density (PSD) from noise samples.

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

from data.g2net import find_dataset_dir, load_labels, load_sample
from data.preprocessing import (
    compute_psd,
    save_psd,
    N_FREQ,
    FS,
)


def compute_and_save_average_psd(
    n_samples: int = 50000,
    output_path: Path = None
) -> np.ndarray:
    """
    Compute average PSD from noise-only samples and save to disk.

    Parameters
    ----------
    n_samples : int
        Number of noise samples to use (more = better estimate, but slower)
    output_path : Path, optional
        Where to save the PSD. Defaults to dataset_dir/avg_psd.npz

    Returns
    -------
    avg_psd : np.ndarray
        Average PSD, shape (3, N_FREQ)
    """
    print("=" * 60)
    print("COMPUTING AVERAGE PSD FROM NOISE SAMPLES")
    print("=" * 60)

    # Find dataset
    dataset_dir = find_dataset_dir()
    print(f"Dataset directory: {dataset_dir}")

    if output_path is None:
        output_path = dataset_dir / "avg_psd.npz"

    # Load labels
    df = load_labels(dataset_dir)
    print(f"Total samples: {len(df)}")

    # Filter to noise-only samples (target = 0)
    noise_df = df[df['target'] == 0]
    print(f"Noise-only samples: {len(noise_df)}")

    # Limit to n_samples
    if len(noise_df) > n_samples:
        noise_df = noise_df.sample(n=n_samples, random_state=42)
    actual_n = len(noise_df)
    print(f"Using {actual_n} samples for PSD estimation")

    # Accumulate PSDs
    print("\nComputing PSDs...")
    psd_sum = np.zeros((3, N_FREQ), dtype=np.float64)   # 3 detectors
    count = 0
    errors = 0

    for sample_id in tqdm(noise_df['id'].astype(str).values, desc="Processing"):
        try:
            sample = load_sample(sample_id)
            psd = compute_psd(sample, FS)
            psd_sum += psd
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"\nError loading {sample_id}: {e}")
    # stop overloading the terminal
    if errors > 5:
        print(f"\n... and {errors - 5} more errors")

    # Compute average
    avg_psd = psd_sum / count
    print(f"\nComputed average from {count} samples ({errors} errors)")

    # Apply smoothing to reduce variance
    print("Applying smoothing...")
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    for det in range(3):
        avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    # Ensure no zeros
    avg_psd = np.maximum(avg_psd, 1e-50)

    # Save
    save_psd(avg_psd, output_path)

    # Print some statistics
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


def visualize_psd(avg_psd: np.ndarray, save_path: Path = None):
    """
    Visualize the average PSD for each detector.

    Parameters
    ----------
    avg_psd : np.ndarray
        Average PSD, shape (3, N_FREQ)
    save_path : Path, optional
        Where to save the plot
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        freqs = np.fft.rfftfreq(4096, d=1.0/FS)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        detector_names = ['H1 (Hanford)', 'L1 (Livingston)', 'V1 (Virgo)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i, (ax, name, color) in enumerate(zip(axes, detector_names, colors)):
            ax.semilogy(freqs, avg_psd[i], color=color, linewidth=0.5)
            ax.set_ylabel('PSD')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.axvline(20, color='red', linestyle='--', alpha=0.5, label='20 Hz')
            ax.axvline(500, color='red', linestyle='--', alpha=0.5, label='500 Hz')

        axes[-1].set_xlabel('Frequency (Hz)')
        axes[-1].set_xlim(0, 1024)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"PSD plot saved to: {save_path}")
        plt.close()

    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == "__main__":
    # Compute and save PSD
    avg_psd = compute_and_save_average_psd(n_samples=10000)

    # Try to visualize
    dataset_dir = find_dataset_dir()
    plot_path = dataset_dir / "avg_psd_plot.png"
    visualize_psd(avg_psd, save_path=plot_path)
