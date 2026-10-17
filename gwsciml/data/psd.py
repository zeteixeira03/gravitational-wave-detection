"""Power spectral density (PSD) estimation and whitening.

Conventions (used by every module in gwsciml)
---------------------------------------------
x[n]      real time series, n = 0..N-1, sample spacing dt = 1/fs.
w[n]      taper window (Tukey). mean(w^2) is its power loss.
X_k       rfft(w * x)_k, k = 0..N/2, frequency f_k = k fs / N.
S_k       one-sided PSD [1/Hz]. For a stationary process with PSD S,
          E|X_k|^2 = N mean(w^2) S_k / (2 dt). Hence the estimator
          S_k = 2 dt |X_k|^2 / (N mean(w^2)), averaged over noise samples.
          Check: white noise with variance s^2 has S = 2 s^2 dt.

Whitened frequency series:  Xw_k = X_k / sqrt(S_k fs / 2).
Whitened time series:        y = irfft(Xw).
For noise with PSD S, y has unit variance where w = 1 (the flat part of the
window). This is the normalisation the matched filter in
gwsciml.filtering.matched relies on.

Window choice (measured, see gwsciml/tests/test_psd.py): G2Net mergers sit at
1.5-1.85 s, so the window must stay flat to ~1.9 s; Tukey(0.1) does. Its short
taper leaks the 8-16 Hz noise bump into 20-35 Hz, but when the PSD is
estimated through the same window (periodogram_psd averaged over noise) the
whitening down-weights the leaked bins and the matched-filter SNR loss is
below 5 %. Tukey(0.5) leaks less but cuts ~30 % of the SNR of a merger at 1.85 s.

Noise-weighted inner product, <a|b> = 4 Re int a~(f) b~*(f) / S(f) df, in these
units is <a|b> = (2/N) Re sum_k Aw_k conj(Bw_k), with Aw, Bw whitened rffts.
"""

import numpy as np
from scipy import signal

from gwsciml.constants import DT, FS, N


def tukey(n: int = N, alpha: float = 0.1) -> np.ndarray:
    """Tukey window: flat in the middle, cosine tapers of total fraction alpha."""
    return signal.windows.tukey(n, alpha=alpha)


def periodogram_psd(x: np.ndarray, window: np.ndarray) -> np.ndarray:
    """One-sided PSD estimate from one windowed segment, along the last axis.

    Returns S_k = 2 dt |rfft(w x)|^2 / (N mean(w^2)) with shape (..., N/2+1).
    One segment gives a chi-squared(2) estimate per bin (100 % relative
    scatter), so average this over many noise samples.
    """
    n = x.shape[-1]
    X = np.fft.rfft(x * window, axis=-1)
    return 2.0 * DT * np.abs(X) ** 2 / (n * np.mean(window ** 2))


def welch_psd(x: np.ndarray, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample Welch PSD (Hann segments, 50 % overlap), along the last axis.

    For a 2 s sample and nperseg=1024 this averages 7 segments at 2 Hz
    resolution. Same one-sided density units as periodogram_psd.
    """
    return signal.welch(x, fs=FS, nperseg=nperseg, noverlap=nperseg // 2,
                        window="hann", detrend=False, axis=-1)


def whiten_fd(x: np.ndarray, psd: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Whitened rfft: rfft(w x) / sqrt(S fs / 2). psd broadcasts against (..., N/2+1)."""
    X = np.fft.rfft(x * window, axis=-1)
    return X / np.sqrt(psd * FS / 2.0)


def whiten(x: np.ndarray, psd: np.ndarray, window: np.ndarray,
           band: tuple[float, float] | None = None) -> np.ndarray:
    """Whitened time series (unit variance where w = 1 for noise with PSD psd).

    band: optional (f_lo, f_hi) in Hz; bins outside are set to zero. A band
    limit leaves a variance equal to the fraction of bins kept.
    """
    n = x.shape[-1]
    Xw = whiten_fd(x, psd, window)
    if band is not None:
        f = np.fft.rfftfreq(n, DT)
        Xw = Xw * ((f >= band[0]) & (f <= band[1]))
    return np.fft.irfft(Xw, n=n, axis=-1)


def colored_noise(psd: np.ndarray, n_samples: int, rng: np.random.Generator, n: int = N) -> np.ndarray:
    """Draw stationary Gaussian noise with one-sided PSD psd (length n/2+1).

    Each rfft bin k gets a complex Gaussian with E|X_k|^2 = n S_k / (2 dt)
    (the identity in the module docstring with w = 1). DC and Nyquist bins are
    real. Used by the tests and by synthetic checks; never as training data.
    """
    nf = n // 2 + 1
    var = n * psd / (2.0 * DT)
    X = (rng.standard_normal((n_samples, nf)) + 1j * rng.standard_normal((n_samples, nf))) * np.sqrt(var / 2.0)
    X[:, 0] = rng.standard_normal(n_samples) * np.sqrt(var[0])
    X[:, -1] = rng.standard_normal(n_samples) * np.sqrt(var[-1])
    return np.fft.irfft(X, n=n, axis=-1)
