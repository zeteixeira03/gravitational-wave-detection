"""
Signal Preprocessing for Gravitational Wave Detection

This module implements the critical preprocessing steps that make GW signals
detectable by neural networks:

1. Bandpass filtering - Remove frequencies outside the detector's sensitive range
2. Spectral whitening - Flatten the noise spectrum to make signals stand out
3. Tukey windowing - Reduce edge effects from filtering
4. Normalization - Scale signals for neural network input

The key insight from successful Kaggle solutions:
- Raw signals have colored noise (non-uniform power across frequencies)
- Whitening divides by the average noise PSD, making noise "white" (uniform)
- After whitening, the GW signal's characteristic chirp pattern becomes visible
"""

import numpy as np
from scipy import signal
from typing import Optional, Tuple
from pathlib import Path

# -----------------------------------------------------------------------------
#                                 Constants
# -----------------------------------------------------------------------------

FS: float = 2048.0          # Sampling rate (Hz)
N: int = 4096               # Samples per detector
NYQUIST: float = FS / 2     # Nyquist frequency (1024 Hz)

# Frequency range where GW signals exist (and detectors are sensitive)
FREQ_LOW: float = 20.0      # Below this: seismic noise dominates
FREQ_HIGH: float = 500.0    # Above this: shot noise dominates, fewer GW signals

# Number of frequency bins in rfft output
N_FREQ: int = N // 2 + 1    # 2049 bins


# -----------------------------------------------------------------------------
#                               Bandpass Filter
# -----------------------------------------------------------------------------

def create_bandpass_filter(
    low: float = FREQ_LOW,
    high: float = FREQ_HIGH,
    fs: float = FS,
    order: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create Butterworth bandpass filter coefficients.

    Parameters
    ----------
    low : float
        Low cutoff frequency (Hz)
    high : float
        High cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order (higher = sharper cutoff but more ringing)

    Returns
    -------
    b, a : np.ndarray
        Filter coefficients for use with scipy.signal.filtfilt
    """
    nyq = fs / 2
    low_norm = low / nyq
    high_norm = high / nyq

    # Ensure frequencies are valid
    low_norm = max(0.001, min(low_norm, 0.999))
    high_norm = max(low_norm + 0.001, min(high_norm, 0.999))

    b, a = signal.butter(order, [low_norm, high_norm], btype='band')
    return b, a


def bandpass_filter(
    data: np.ndarray,
    low: float = FREQ_LOW,
    high: float = FREQ_HIGH,
    fs: float = FS,
    order: int = 4
) -> np.ndarray:
    """
    Apply bandpass filter to signal data.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N) or (N,)
    low, high : float
        Cutoff frequencies (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order

    Returns
    -------
    filtered : np.ndarray
        Filtered signal, same shape as input
    """
    b, a = create_bandpass_filter(low, high, fs, order)

    # Handle both single detector and multi-detector input
    if data.ndim == 1:
        return signal.filtfilt(b, a, data)
    else:
        # Apply to each detector independently
        return np.array([signal.filtfilt(b, a, data[i]) for i in range(data.shape[0])])


# -----------------------------------------------------------------------------
#                               Tukey Window
# -----------------------------------------------------------------------------

def apply_tukey_window(data: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Apply Tukey (tapered cosine) window to reduce edge effects.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N) or (N,)
    alpha : float
        Shape parameter. alpha=0 is rectangular, alpha=1 is Hann window.
        alpha=0.5 is a good balance (default in many GW analyses).

    Returns
    -------
    windowed : np.ndarray
        Windowed signal, same shape as input
    """
    n_samples = data.shape[-1]
    window = signal.windows.tukey(n_samples, alpha=alpha)

    if data.ndim == 1:
        return data * window
    else:
        # Broadcast window across detectors
        return data * window[np.newaxis, :]


# -----------------------------------------------------------------------------
#                               PSD Computation
# -----------------------------------------------------------------------------

def compute_psd(data: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Compute Power Spectral Density of a signal.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N) for multi-detector or (N,) for single
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    psd : np.ndarray
        Power spectral density, shape (3, N_FREQ) or (N_FREQ,)
    """
    # Apply window before FFT to reduce spectral leakage
    windowed = apply_tukey_window(data, alpha=0.5)

    # Compute FFT
    fft_vals = np.fft.rfft(windowed, axis=-1)

    # PSD = |FFT|^2, normalized by number of samples and sampling freq
    psd = np.abs(fft_vals) ** 2
    psd = psd / (fs * data.shape[-1])

    return psd


def compute_average_psd(
    noise_samples: np.ndarray,
    fs: float = FS,
    smoothing: bool = True
) -> np.ndarray:
    """
    Compute average PSD from multiple noise-only samples.

    This is the KEY step for whitening: we estimate the detector noise
    spectrum by averaging over many noise-only samples (target=0).

    Parameters
    ----------
    noise_samples : np.ndarray
        Array of noise samples, shape (n_samples, 3, N)
    fs : float
        Sampling frequency (Hz)
    smoothing : bool
        Whether to apply smoothing to reduce variance
        
    Returns
    -------
    avg_psd : np.ndarray
        Average PSD per detector, shape (3, N_FREQ)
    """
    n_samples = noise_samples.shape[0]

    # Accumulate PSDs
    psd_sum = np.zeros((3, N_FREQ), dtype=np.float64)

    for i in range(n_samples):
        psd = compute_psd(noise_samples[i], fs)
        psd_sum += psd

    avg_psd = psd_sum / n_samples

    if smoothing:
        # Apply light smoothing to reduce variance in PSD estimate
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        for det in range(3):
            avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    # Ensure no zeros
    avg_psd = np.maximum(avg_psd, 1e-50)

    return avg_psd


def compute_average_psd_from_generator(
    sample_generator,
    n_samples: int,
    fs: float = FS,
    smoothing: bool = True,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute average PSD from a generator of noise samples.

    Memory-efficient version that doesn't require loading all samples at once.

    Parameters
    ----------
    sample_generator : generator
        Yields noise samples of shape (3, N)
    n_samples : int
        Number of samples to process
    fs : float
        Sampling frequency (Hz)
    smoothing : bool
        Whether to apply smoothing
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
        psd = compute_psd(sample, fs)
        psd_sum += psd
        count += 1

        if verbose and count % 1000 == 0:
            print(f"  Processed {count}/{n_samples} samples...")

        if count >= n_samples:
            break

    avg_psd = psd_sum / count

    if smoothing:
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        for det in range(3):
            avg_psd[det] = np.convolve(avg_psd[det], kernel, mode='same')

    avg_psd = np.maximum(avg_psd, 1e-50)

    if verbose:
        print(f"  Computed average PSD from {count} samples")

    return avg_psd


# -----------------------------------------------------------------------------
# Spectral Whitening
# -----------------------------------------------------------------------------

def whiten_signal(
    data: np.ndarray,
    avg_psd: np.ndarray,
    fs: float = FS
) -> np.ndarray:
    """
    Whiten a signal using the average noise PSD.

    Whitening divides the signal's frequency content by the noise spectrum,
    effectively "flattening" the noise to be uniform across frequencies.
    This makes the GW signal's characteristic chirp pattern stand out.

    Parameters
    ----------
    data : np.ndarray
        Signal data, shape (3, N)
    avg_psd : np.ndarray
        Average noise PSD, shape (3, N_FREQ)
    fs : float
        Sampling frequency (Hz)

    Returns
    -------
    whitened : np.ndarray
        Whitened signal, shape (3, N)
    """
    # Apply window before FFT
    windowed = apply_tukey_window(data, alpha=0.5)

    # Compute FFT
    fft_vals = np.fft.rfft(windowed, axis=-1)

    # Whiten: divide by sqrt(PSD)
    # Adding small epsilon for numerical stability
    whitening_filter = 1.0 / np.sqrt(avg_psd + 1e-50)

    # Apply whitening in frequency domain
    whitened_fft = fft_vals * whitening_filter

    # Transform back to time domain
    whitened = np.fft.irfft(whitened_fft, n=N, axis=-1)

    return whitened


# -----------------------------------------------------------------------------
#                   Complete Preprocessing Pipeline
# -----------------------------------------------------------------------------

def preprocess_sample(
    sample: np.ndarray,
    avg_psd: np.ndarray,
    apply_bandpass: bool = True,
    apply_whitening: bool = True,
    apply_window: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Apply complete preprocessing pipeline to a sample.

    Pipeline order:
    1. Bandpass filter (20-500 Hz) - remove out-of-band noise
    2. Spectral whitening - flatten noise spectrum
    3. Tukey window - reduce edge effects
    4. Normalize - scale for neural network input

    Parameters
    ----------
    sample : np.ndarray
        Raw signal, shape (3, N)
    avg_psd : np.ndarray
        Average noise PSD for whitening, shape (3, N_FREQ)
    apply_bandpass : bool
        Whether to apply bandpass filter
    apply_whitening : bool
        Whether to apply spectral whitening
    apply_window : bool
        Whether to apply Tukey window (after whitening)
    normalize : bool
        Whether to normalize to zero mean, unit variance

    Returns
    -------
    processed : np.ndarray
        Preprocessed signal, shape (3, N)
    """
    processed = sample.copy()

    # Step 1: Bandpass filter
    if apply_bandpass:
        processed = bandpass_filter(processed)

    # Step 2: Spectral whitening
    if apply_whitening:
        processed = whiten_signal(processed, avg_psd)

    # Step 3: Tukey window (reduces edge artifacts)
    if apply_window:
        processed = apply_tukey_window(processed, alpha=0.2)

    # Step 4: Normalize each detector independently
    if normalize:
        for det in range(processed.shape[0]):
            det_mean = processed[det].mean()
            det_std = processed[det].std()
            if det_std > 1e-10:
                processed[det] = (processed[det] - det_mean) / det_std
            else:
                processed[det] = processed[det] - det_mean

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


# -----------------------------------------------------------------------------
# Frequency Grid Utility
# -----------------------------------------------------------------------------

def get_frequency_grid(n: int = N, fs: float = FS) -> np.ndarray:
    """
    Get the frequency values corresponding to rfft output.

    Returns
    -------
    freqs : np.ndarray
        Frequency values in Hz, shape (N_FREQ,)
    """
    return np.fft.rfftfreq(n, d=1.0/fs)
