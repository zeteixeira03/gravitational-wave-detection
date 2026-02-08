"""
Signal Preprocessing for Gravitational Wave Detection

This module implements the critical preprocessing steps that make GW signals
detectable by neural networks:

1. Bandpass filtering - Remove frequencies outside the detector's sensitive range
2. Spectral whitening - Flatten the noise spectrum to make signals stand out
3. Tukey windowing - Reduce edge effects from filtering
4. Normalization - Scale signals for neural network input

"""

import numpy as np
from scipy import signal
from pathlib import Path 

# -----------------------------------------------------------------------------
#                                 Constants
# -----------------------------------------------------------------------------

FS: float = 2048.0          # Sampling rate (Hz)
N: int = 4096               # Samples per detector

# Frequency range where GW signals exist (and detectors are sensitive)
FREQ_LOW: float = 20.0      # Below this: seismic noise dominates
FREQ_HIGH: float = 500.0    # Above this: shot noise dominates, fewer GW signals

# Number of frequency bins in rfft output
N_FREQ: int = N // 2 + 1    # 2049 bins


# -----------------------------------------------------------------------------
#                               Bandpass Filter
# -----------------------------------------------------------------------------

# precompute filter coefficients (20-500 Hz bandpass, order 4)
_nyq = FS / 2
_b, _a = signal.butter(4, [FREQ_LOW / _nyq, FREQ_HIGH / _nyq], btype='band')


def bandpass_filter(data: np.ndarray) -> np.ndarray:
    """
    Apply 20-500 Hz bandpass filter to signal data.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N)

    Returns
    -------
    filtered : np.ndarray
        Filtered signal, same shape as input
    """
    return signal.filtfilt(_b, _a, data, axis=-1)


# -----------------------------------------------------------------------------
#                               Tukey Window
# -----------------------------------------------------------------------------

def apply_tukey_window(data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply Tukey window to reduce edge effects.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N) or (N,)
    alpha : float
        Shape parameter. alpha=0 is rectangular, alpha=1 is Hann window.
        alpha=0.5 is a good enough balance .

    Returns
    -------
    windowed : np.ndarray
        Windowed signal, same shape as input
    """
    n_samples = data.shape[-1]
    window = signal.windows.tukey(n_samples, alpha=alpha)

    return data * window


# -----------------------------------------------------------------------------
#                               PSD Computation
# -----------------------------------------------------------------------------

def compute_psd(data: np.ndarray) -> np.ndarray:
    """
    Compute Power Spectral Density of a signal.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N)

    Returns
    -------
    psd : np.ndarray
        Power spectral density, shape (3, N_FREQ)
    """
    windowed = apply_tukey_window(data, alpha=0.5)
    fft_vals = np.fft.rfft(windowed, axis=-1)
    return np.abs(fft_vals) ** 2 / (FS * N)


def compute_average_psd(noise_samples: np.ndarray) -> np.ndarray:
    """
    Compute average PSD from multiple noise-only samples.

    This is the key step for whitening: we estimate the detector noise
    spectrum by averaging over many noise-only samples (target=0).

    Parameters
    ----------
    noise_samples : np.ndarray
        Array of noise samples, shape (n_samples, 3, N)

    Returns
    -------
    avg_psd : np.ndarray
        Average PSD per detector, shape (3, N_FREQ)
    """
    n_samples = noise_samples.shape[0]

    psd_sum = np.zeros((3, N_FREQ), dtype=np.float64)
    for i in range(n_samples):
        psd = compute_psd(noise_samples[i])
        psd_sum += psd

    avg_psd = psd_sum / n_samples

    # smooth to reduce variance
    kernel = np.ones(5) / 5
    for det in range(3):
        avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    return np.maximum(avg_psd, 1e-50)


def compute_average_psd_from_generator(sample_generator, n_samples: int, verbose: bool = True) -> np.ndarray:
    """
    Compute average PSD from a generator of noise samples.

    Memory-efficient version that doesn't require loading all samples at once.

    Parameters
    ----------
    sample_generator : generator
        Yields noise samples of shape (3, N)
    n_samples : int
        Number of samples to process
    verbose : bool
        Whether to print progress

    Returns
    -------
    avg_psd : np.ndarray
        Average PSD per detector, shape (3, N_FREQ)
    """
    psd_sum = np.zeros((3, N_FREQ), dtype=np.float64)
    count = 0

    for sample in sample_generator:
        psd = compute_psd(sample)
        psd_sum += psd
        count += 1

        if verbose and count % 1000 == 0:
            print(f"  Processed {count}/{n_samples} samples...")

        if count >= n_samples:
            break

    avg_psd = psd_sum / count

    # smooth to reduce variance
    kernel = np.ones(5) / 5
    for det in range(3):
        avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    if verbose:
        print(f"  Computed average PSD from {count} samples")

    return np.maximum(avg_psd, 1e-50)


# -----------------------------------------------------------------------------
# Spectral Whitening
# -----------------------------------------------------------------------------

def whiten_signal(data: np.ndarray, avg_psd: np.ndarray) -> np.ndarray:
    """
    Whiten a signal using the average noise PSD.

    Whitening divides the signal's frequency content by the noise spectrum,
    effectively flattening the noise to be uniform across frequencies.
    This makes the GW signal's characteristic chirp pattern stand out.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N)
    avg_psd : np.ndarray
        Average noise PSD, shape (3, N_FREQ)

    Returns
    -------
    whitened : np.ndarray
        Whitened signal, shape (3, N)
    """
    windowed = apply_tukey_window(data, alpha=0.5)
    fft_vals = np.fft.rfft(windowed, axis=-1)
    whitening_filter = 1.0 / np.sqrt(avg_psd + 1e-50)
    whitened_fft = fft_vals * whitening_filter
    return np.fft.irfft(whitened_fft, n=N, axis=-1)


# -----------------------------------------------------------------------------
#                   Complete Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess_sample(sample: np.ndarray, avg_psd: np.ndarray) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to a sample.

    Pipeline: bandpass -> whiten -> window -> normalize

    Parameters
    ----------
    sample : np.ndarray
        Raw signal, shape (3, N)
    avg_psd : np.ndarray
        Average noise PSD for whitening, shape (3, N_FREQ)

    Returns
    -------
    processed : np.ndarray
        Preprocessed signal, shape (3, N)
    """
    processed = bandpass_filter(sample)
    processed = whiten_signal(processed, avg_psd)
    processed = apply_tukey_window(processed, alpha=0.2)

    # normalize each detector
    mean = processed.mean(axis=-1, keepdims=True)
    std = np.maximum(processed.std(axis=-1, keepdims=True), 1e-10)
    processed = (processed - mean) / std

    return processed.astype(np.float32)


# -----------------------------------------------------------------------------
#                       PSD Save/Load Utilities
# -----------------------------------------------------------------------------

def save_psd(avg_psd: np.ndarray, path: Path) -> None:
    """Save average PSD to disk."""
    np.savez_compressed(path, avg_psd=avg_psd)
    print(f"Average PSD saved to: {path}")


def load_psd(path: Path) -> np.ndarray:
    """Load average PSD from disk."""
    data = np.load(path)
    return data['avg_psd']
