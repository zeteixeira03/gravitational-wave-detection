"""End-to-end checks of the P1 detector on synthetic three-detector data."""

import numpy as np
import pytest
import torch

from gwsciml.constants import DT, FS, N
from gwsciml.data.psd import colored_noise, periodogram_psd, tukey
from gwsciml.detect.classical import ClassicalDetector, DetectorConfig
from gwsciml.filtering.matched import band_mask, inner, whiten_template
from gwsciml.filtering.network import Network
from gwsciml.templates.waveforms import taylorf2
from gwsciml.tests.helpers import FREQS, g2net_like_psd

RNG = np.random.default_rng(3)
MASS = [(40.0, 35.0), (30.0, 20.0), (60.0, 50.0)]


@pytest.fixture(scope="module")
def setup():
    S = g2net_like_psd()
    psds_true = np.stack([S, S * 1.1, S * 4.0])  # V1 ~2x less sensitive in amplitude
    w = tukey(N, 0.1)
    noise = np.stack([colored_noise(p, 300, RNG) for p in psds_true], axis=1)  # (300, 3, N)
    psds = periodogram_psd(noise, w).mean(axis=0)   # leakage-inclusive estimate, as on real data
    bank = np.stack([taylorf2(FREQS, *m) for m in MASS])
    det = ClassicalDetector(bank, psds, w, DetectorConfig(n_sky=500, top_k=3))
    return det, psds_true, bank


def inject(psds_true, h_fd, net, sky, psi, t_h1, phase, snr_h1):
    """Signal in each detector: F+ h + Fx (h * -i) with h = h_fd e^{i phase}, shifted by delays.
    Scaled so that the H1 optimal SNR equals snr_h1."""
    mask = band_mask(FREQS, (20.0, 1000.0))
    from gwsciml.filtering.network import antenna_patterns
    out = np.zeros((3, N))
    dt = net.delays[sky] - net.delays[sky, 0]
    for k in range(3):
        fp, fx = antenna_patterns(net.tensors[k], net.lon[sky], net.lat[sky], psi)
        hk = (fp - 1j * fx) * h_fd * np.exp(1j * phase) * mask
        out[k] = np.fft.irfft(hk * FS * np.exp(-2j * np.pi * FREQS * (t_h1 + dt[k])), n=N)
    hw = whiten_template(np.fft.rfft(out[0]) / FS, psds_true[0], mask, normalize=False)
    return out * snr_h1 / np.sqrt(inner(hw, hw))


def test_chisq_band_sum_equals_snr(setup):
    det, psds_true, bank = setup
    x = np.stack([colored_noise(p, 1, RNG)[0] for p in psds_true])
    dw = det.whiten(x)
    z = det.snr_series(dw, torch.as_tensor([1]))[:, 0]
    j = np.array([3000, 3005, 2990])
    cs = det.chisq(dw, 1, j)
    for k in range(3):
        assert cs[k][0] == pytest.approx(float(z[k, j[k]].abs()), rel=1e-4)


def test_noise_only_background(setup):
    det, psds_true, _ = setup
    stats = [det.statistics(np.stack([colored_noise(p, 1, RNG)[0] for p in psds_true])) for _ in range(20)]
    inc = np.array([s["inc_max"] for s in stats])
    coh = np.array([s["coh_max"] for s in stats])
    # incoherent max of ~chi2(6) over ~3 templates x 3700 times: typically 25-40
    assert 15 < np.median(inc) < 45
    # coherent projection keeps 4 of 6 real dof, so its background is lower
    assert np.median(coh) < np.median(inc)
    assert np.median([s["chi2r_H"] for s in stats]) == pytest.approx(1.0, abs=0.5)


def test_injection_recovery(setup):
    det, psds_true, bank = setup
    net = det.net
    sky, psi, t_h1 = 123, 0.4, 1.70
    sig = inject(psds_true, bank[0], net, sky, psi, t_h1, 0.9, snr_h1=12.0)
    noise = np.stack([colored_noise(p, 1, RNG)[0] for p in psds_true])
    s = det.statistics(noise + sig)
    assert s["tmpl"] == 0
    assert s["rho_H"] == pytest.approx(12.0, abs=2.5)
    assert s["coh_max"] > 0.8 * s["inc_max"]          # signal power sits in the coherent subspace
    assert s["null_at_coh"] < 8.0                     # null stream ~ chi2(2) noise
    assert s["chi2r_H"] < 2.0


def test_high_snr_timing(setup):
    # timing sd 1/(2 pi rho sigma_f): ~1 ms at rho 12 for this 20-70 Hz chirp, ~0.3 ms at rho 40
    det, psds_true, bank = setup
    net = det.net
    sky, t_h1 = 321, 1.62
    sig = inject(psds_true, bank[0], net, sky, 2.0, t_h1, -1.1, snr_h1=40.0)
    s = det.statistics(sig + np.stack([colored_noise(p, 1, RNG)[0] for p in psds_true]))
    true_dt = net.delays[sky] - net.delays[sky, 0]
    assert s["t_h1"] == pytest.approx(t_h1, abs=0.0015)
    assert s["dt_L"] == pytest.approx(true_dt[1], abs=0.0015)


def test_likelihood_map_consistent_with_lse(setup):
    # logsumexp over the map (bin widths as weights) reproduces the global inc_lse
    det, psds_true, _ = setup
    x = np.stack([colored_noise(p, 1, RNG)[0] for p in psds_true])
    st, lmap = det.statistics(x, return_map=True)
    w_b = int(round(det.cfg.time_bin * FS))
    n_t = det.j_hi - det.j_lo
    counts = np.full(det.n_time_bins, w_b, dtype=float)
    counts[-1] = n_t - w_b * (det.n_time_bins - 1)
    from scipy.special import logsumexp
    total = logsumexp(lmap.astype(float) + np.log(counts)) - np.log(lmap.shape[0] * n_t)
    assert total == pytest.approx(st["inc_lse"], abs=1e-3)
