import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------

FS: float = 2048.0          # sampling rate (Hz)
N: int = 4096               # samples per detector
DETECTORS: list[str, str, str] = ["Hanford (H1)", "Livingston (L1)", "Virgo (V1)"]

# freq bands (Hz)
BANDS: list[Tuple[float, float]] = [
    (20, 64),
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
]

# precompute frequency grid and band index ranges
_FREQS = np.fft.rfftfreq(N, d=1.0 / FS)  # shape (N/2 + 1,)
_BAND_IDX: list[Tuple[int, int]] = [
    (
        int(np.searchsorted(_FREQS, f_low, side="left")),           # sort FFT results according to BANDS
        int(np.searchsorted(_FREQS, f_high, side="left")),
    )
    for (f_low, f_high) in BANDS
]


def compute_features(sample: np.ndarray) -> np.ndarray:
    """
    Compute engineered features for a single training sample.

    Uses a hybrid approach:
    part 1. Extract RAW amplitude features first (signal strength indicators)
    part 2. Then normalize and extract SHAPE features (correlations, frequency distribution)

    Parameters
    ----------
    sample : np.ndarray
        Waveform array of shape (3, N), one row per detector:
        [H1, L1, V1] (= [Hanford, Livingston, Virgo]).

    Returns
    -------
    feats : np.ndarray
        1D array of 39 engineered features
    """
    if sample.shape != (3, N):
        raise ValueError(f"Expected sample shape (3, {N}), got {sample.shape}")

    feats: list[float] = []

    # =====================================================================
    #                               part 1
    # =====================================================================

    # raw statistics per detector
    raw_stds = sample.std(axis=1)                      # shape (3,)
    raw_max_abs = np.max(np.abs(sample), axis=1)       # shape (3,)
    raw_rms = np.sqrt(np.mean(sample ** 2, axis=1))    # shape (3,)

    # log-scale raw features to handle tiny values (~1e-20)
    # log1p(x * 1e20) maps 1e-20 -> log1p(1) ~ 0.69, 1e-19 -> log1p(10) ~ 2.4
    SCALE = 1e20
    for i in range(3):
        feats.extend([
            float(np.log1p(raw_stds[i] * SCALE)),
            float(np.log1p(raw_max_abs[i] * SCALE)),
            float(np.log1p(raw_rms[i] * SCALE)),
        ])

    # cross-detector amplitude ratios (signal should affect all detectors similarly)
    # use log ratios for numerical stability
    def safe_log_ratio(a: float, b: float) -> float:
        eps = 1e-30
        return float(np.log((a + eps) / (b + eps)))

    feats.append(safe_log_ratio(raw_stds[0], raw_stds[1]))  # H1/L1 std ratio
    feats.append(safe_log_ratio(raw_stds[0], raw_stds[2]))  # H1/V1 std ratio
    feats.append(safe_log_ratio(raw_stds[1], raw_stds[2]))  # L1/V1 std ratio
    # 12 features so far

    # raw frequency domain features (before normalization)
    raw_fft = np.fft.rfft(sample, axis=1)
    raw_power = np.abs(raw_fft) ** 2  # shape (3, N_freq)

    # total power per detector (log-scaled)
    for i in range(3):
        total_power = raw_power[i].sum()
        feats.append(float(np.log1p(total_power * SCALE**2)))
    # 15 features so far

    # band power ratios (low freq vs high freq) - GW signals are typically in specific bands
    for det in range(3):
        p = raw_power[det]
        low_band = p[_BAND_IDX[0][0]:_BAND_IDX[1][1]].sum()   # 20-128 Hz
        high_band = p[_BAND_IDX[3][0]:_BAND_IDX[4][1]].sum()  # 256-1024 Hz
        feats.append(safe_log_ratio(low_band, high_band))
    # 18 features so far

    # =====================================================================
    #                           part 2
    # =====================================================================

    # normalize waveforms for shape-based features
    sample_norm = (sample - sample.mean(axis=1, keepdims=True)) / (raw_stds[:, np.newaxis] + 1e-30)

    # cross-detector correlations (on normalized data)
    # GW signals should show correlated patterns across detectors
    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        num = (x_centered * y_centered).mean()
        den = x.std() * y.std()
        if den == 0:
            return 0.0
        return float(num / den)

    feats.append(correlation(sample_norm[0], sample_norm[1]))  # H1-L1
    feats.append(correlation(sample_norm[0], sample_norm[2]))  # H1-V1
    feats.append(correlation(sample_norm[1], sample_norm[2]))  # L1-V1
    # 21 features so far

    # normalized frequency band distribution (shape of spectrum)
    norm_fft = np.fft.rfft(sample_norm, axis=1)
    norm_power = np.abs(norm_fft) ** 2

    for det in range(3):
        p = norm_power[det]
        total = p.sum() + 1e-30
        # fractional power in each band (sums to ~1)
        for start_idx, end_idx in _BAND_IDX:
            band_power = p[start_idx:end_idx].sum()
            feats.append(float(band_power / total))
    # 21 + 15 = 36 features so far

    # peak frequency per detector (where is the max power?)
    for det in range(3):
        peak_idx = np.argmax(norm_power[det])
        peak_freq_normalized = peak_idx / len(norm_power[det])  # 0 to 1
        feats.append(float(peak_freq_normalized))
    # 39 features total

    return np.asarray(feats, dtype=np.float32)