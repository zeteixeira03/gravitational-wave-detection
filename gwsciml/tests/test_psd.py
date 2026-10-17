"""Analytic identities for PSD estimation and whitening."""

import numpy as np
import pytest

from gwsciml.constants import DT, FS, N
from gwsciml.data.psd import colored_noise, periodogram_psd, tukey, welch_psd, whiten, whiten_fd

from gwsciml.tests.helpers import FREQS, g2net_like_psd

RNG = np.random.default_rng(1234)


def test_white_noise_psd_level():
    # white noise with variance s^2 has one-sided PSD 2 s^2 dt
    s = 3.0
    x = RNG.standard_normal((2000, N)) * s
    for est in (periodogram_psd(x, tukey(N, 0.1)), welch_psd(x)[1]):
        level = est[:, 5:-5].mean()
        assert level == pytest.approx(2 * s ** 2 * DT, rel=0.01)


def test_colored_noise_recovers_psd():
    S = g2net_like_psd()
    x = colored_noise(S, 4000, RNG)
    est = periodogram_psd(x, np.ones(N)).mean(axis=0)
    ratio = est[1:-1] / S[1:-1]
    # 4000 chi2(2)/2 draws per bin: relative sd 1/sqrt(4000) = 1.6 %
    assert np.abs(ratio.mean() - 1) < 0.005
    assert np.abs(ratio - 1).max() < 0.1


def test_whitened_noise_unit_variance_and_flat():
    S = g2net_like_psd()
    x = colored_noise(S, 2000, RNG)
    w = tukey(N, 0.5)
    y = whiten(x, S, w, band=(20.0, 1024.0))
    frac = (FREQS >= 20).sum() / (N / 2)
    assert y[:, w == 1.0].var() / frac == pytest.approx(1.0, rel=0.01)
    # flat spectrum: whitened rfft has E|Xw|^2 = N mean(w^2) in every bin above 20 Hz
    p = (np.abs(whiten_fd(x, S, w)) ** 2).mean(axis=0) / (N * np.mean(w ** 2))
    groups = p[FREQS >= 20][:1000].reshape(-1, 10).mean(axis=1)  # 5 Hz groups, 20-520 Hz
    assert np.abs(groups - 1).max() < 0.06  # 4 % leakage at 25 Hz + 2.2 % sampling sd


def test_short_taper_leaks_low_frequency_power():
    # Documents why the default window is Tukey(0.5): with alpha = 0.1 the 12 Hz bump
    # leaks into 20-30 Hz and inflates the whitened power there by more than 20 %.
    S = g2net_like_psd()
    x = colored_noise(S, 1000, RNG)
    w = tukey(N, 0.1)
    p = (np.abs(whiten_fd(x, S, w)) ** 2).mean(axis=0) / (N * np.mean(w ** 2))
    assert p[(FREQS >= 20) & (FREQS < 30)].mean() > 1.2


def test_band_limited_variance_equals_bandwidth_fraction():
    x = RNG.standard_normal((1000, N))
    S = np.full(N // 2 + 1, 2 * DT)
    y = whiten(x, S, np.ones(N), band=(20.0, 500.0))
    f = np.fft.rfftfreq(N, DT)
    frac = ((f >= 20) & (f <= 500)).sum() / (N / 2)
    assert y.var() == pytest.approx(frac, rel=0.02)


def test_inner_product_matches_continuum_definition():
    # <a|b> = 4 Re sum a~ b~* / S df with a~ = rfft(a) / fs, against (2/N) Re sum Aw Bw*
    S = g2net_like_psd()
    a, b = RNG.standard_normal((2, N))
    w = np.ones(N)
    A, B = np.fft.rfft(a) / FS, np.fft.rfft(b) / FS
    cont = 4 * np.real(np.sum(A * np.conj(B) / S)) * (FS / N)
    Aw, Bw = whiten_fd(a, S, w), whiten_fd(b, S, w)
    disc = (2.0 / N) * np.real(np.sum(Aw * np.conj(Bw)))
    assert disc == pytest.approx(cont, rel=1e-12)
