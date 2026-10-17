"""Matched filter: normalisation, noise statistics and injection recovery."""

import numpy as np
import pytest
import torch

from gwsciml.constants import DT, FS, N
from gwsciml.data.psd import colored_noise, tukey, whiten_fd
from gwsciml.filtering.matched import band_mask, complex_snr, inner, match, whiten_template
from gwsciml.templates.waveforms import chirp_mass, newtonian_chirp_time, taylorf2
from gwsciml.tests.helpers import FREQS, g2net_like_psd

MASK = band_mask(FREQS, (20.0, 1000.0))


def inject(h_fd, psd, snr, j0):
    """Time-domain signal with merger at sample j0 and optimal SNR snr against psd."""
    hw = whiten_template(h_fd, psd, MASK, normalize=False)
    scale = snr / np.sqrt(inner(hw, hw))
    shifted = h_fd * MASK * np.exp(-2j * np.pi * FREQS * j0 * DT)
    return scale * np.fft.irfft(shifted * FS, n=N)


def test_template_normalisation():
    hw = whiten_template(taylorf2(FREQS, 20.0, 15.0), g2net_like_psd(), MASK)
    assert inner(hw, hw) == pytest.approx(1.0, rel=1e-12)


def test_noise_snr_statistics():
    S = g2net_like_psd()
    rng = np.random.default_rng(0)
    w = tukey(N, 0.5)
    x = colored_noise(S, 400, rng)
    dw = torch.as_tensor(whiten_fd(x, S, w) * MASK)
    hw = torch.as_tensor(whiten_template(taylorf2(FREQS, 40.0, 35.0), S, MASK)[None])
    z = complex_snr(dw, hw)[:, 0]
    # merger times whose template support (< 0.5 s here) lies inside the flat part [0.5, 1.5] s
    zr = z[:, 2048:3072]
    assert zr.real.var().item() == pytest.approx(1.0, rel=0.03)
    assert zr.imag.var().item() == pytest.approx(1.0, rel=0.03)
    assert (zr.abs() ** 2).mean().item() == pytest.approx(2.0, rel=0.03)


def timing_sd_samples(hw, rho):
    """Cramer-Rao timing error of |z|: sigma_t = 1 / (2 pi rho sigma_f), sigma_f the
    spread of frequency weighted by |Hw|^2 (the phase is maximised, so only the spread counts)."""
    p = np.abs(hw) ** 2 / np.sum(np.abs(hw) ** 2)
    sf = np.sqrt(np.sum(p * FREQS ** 2) - np.sum(p * FREQS) ** 2)
    return FS / (2 * np.pi * rho * sf)


@pytest.mark.parametrize("snr,j0", [(8.0, 2900), (15.0, 2600)])
def test_injection_recovered_at_time_and_snr(snr, j0):
    S = g2net_like_psd()
    h = taylorf2(FREQS, 40.0, 35.0)
    hw = whiten_template(h, S, MASK)
    rng = np.random.default_rng(1)
    w = tukey(N, 0.5)
    n_trials = 200
    d = colored_noise(S, n_trials, rng) + inject(h, S, snr, j0)
    z = complex_snr(torch.as_tensor(whiten_fd(d, S, w) * MASK), torch.as_tensor(hw[None]))[:, 0]
    # exact identity at the true time: E|z|^2 = rho^2 + 2 (sd of the mean ~ 2 rho / sqrt(n))
    m2 = (z[:, j0].abs() ** 2).mean().item()
    assert m2 - 2 == pytest.approx(snr ** 2, abs=4 * 2 * snr / np.sqrt(n_trials))
    # arrival time: median absolute error of a Gaussian is 0.67 sd; allow 1.5 sd + 1 sample
    err = np.abs(z.abs().argmax(dim=-1).numpy() - j0)
    assert np.median(err) <= 1.5 * timing_sd_samples(hw, snr) + 1


def test_noiseless_injection_exact():
    S = g2net_like_psd()
    h = taylorf2(FREQS, 30.0, 25.0)
    d = inject(h, S, 10.0, 2048)
    z = complex_snr(torch.as_tensor(whiten_fd(d, S, np.ones(N)) * MASK),
                    torch.as_tensor(whiten_template(h, S, MASK)[None]))[0].abs().numpy()
    assert np.argmax(z) == 2048
    assert z.max() == pytest.approx(10.0, rel=1e-3)


def test_match_identity_and_time_shift_invariance():
    S = g2net_like_psd()
    h = taylorf2(FREQS, 30.0, 25.0)
    hw = whiten_template(h, S, MASK)
    hs = hw * np.exp(-2j * np.pi * FREQS * 0.3) * np.exp(1j * 0.7)  # shift 0.3 s, phase 0.7
    m = match(hw[None], np.stack([hw, hs]))
    assert m[0, 0] == pytest.approx(1.0, abs=1e-9)
    assert m[0, 1] == pytest.approx(1.0, abs=1e-3)


def test_taylorf2_newtonian_stationary_phase_time():
    # Newtonian SPA: t(f) = (1/2pi) dPsi/df, so t_c - t(f) = tau0(f). Check d/df of the phase.
    from gwsciml.templates.waveforms import taylorf2_phase
    m1, m2 = 20.0, 10.0
    f = np.array([25.0, 40.0, 60.0])
    eps = 1e-4
    dpsi = (taylorf2_phase(f + eps, m1, m2, 0) - taylorf2_phase(f - eps, m1, m2, 0)) / (2 * eps)
    tau = -dpsi / (2 * np.pi)
    np.testing.assert_allclose(tau, newtonian_chirp_time(f, chirp_mass(m1, m2)), rtol=1e-6)


def test_taylorf2_amplitude_power_law():
    h = taylorf2(FREQS, 10.0, 10.0)
    nz = np.abs(h) > 0
    slope = np.polyfit(np.log(FREQS[nz]), np.log(np.abs(h[nz])), 1)[0]
    assert slope == pytest.approx(-7.0 / 6.0, abs=1e-9)


@pytest.mark.parametrize("pn_order", [0, 2, 4, 6, 7])
def test_taylorf2_matches_lalsimulation(pn_order):
    lal = pytest.importorskip("lal")
    ls = pytest.importorskip("lalsimulation")
    from gwsciml.templates.waveforms import f_isco
    S = g2net_like_psd()
    for m1, m2 in [(8.0, 6.0), (40.0, 10.0)]:
        params = lal.CreateDict()
        ls.SimInspiralWaveformParamsInsertPNPhaseOrder(params, pn_order)
        hp, _ = ls.SimInspiralChooseFDWaveform(
            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0, 1e8 * lal.PC_SI, 0, 0, 0, 0, 0,
            FREQS[1], 20.0, FREQS[-1], 20.0, params, ls.TaylorF2)
        ref = np.zeros(len(FREQS), complex)
        n = min(len(FREQS), hp.data.length)
        ref[:n] = hp.data.data[:n]
        ref[FREQS >= f_isco(m1, m2)] = 0.0  # LAL runs to f_max; ours stops at the ISCO
        mine = whiten_template(taylorf2(FREQS, m1, m2, pn_order=pn_order), S, MASK)
        assert match(mine[None], whiten_template(ref, S, MASK)[None])[0, 0] > 0.99999
