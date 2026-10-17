"""The batched front end reproduces the per-sample classical detector."""

import numpy as np
import torch

from gwsciml.constants import N
from gwsciml.data.psd import colored_noise, periodogram_psd, tukey
from gwsciml.detect.classical import ClassicalDetector, DetectorConfig
from gwsciml.models.frontend import MatchedFilterFrontEnd
from gwsciml.templates.waveforms import taylorf2
from gwsciml.tests.helpers import FREQS, g2net_like_psd


def test_frontend_matches_classical_detector():
    rng = np.random.default_rng(5)
    S = g2net_like_psd()
    psd_true = np.stack([S, 1.1 * S, 4 * S])
    w = tukey(N, 0.1)
    noise = np.stack([colored_noise(p, 200, rng) for p in psd_true], axis=1)
    c = 1e-20 / np.sqrt(S[400])           # bring the strain to the real G2Net scale ~1e-20
    noise = noise * c                      # (tests the float32 rescaling inside the front end)
    psds = periodogram_psd(noise, w).mean(axis=0)
    bank = np.stack([taylorf2(FREQS, 40.0, 35.0), taylorf2(FREQS, 25.0, 15.0)])
    det = ClassicalDetector(bank, psds, w, DetectorConfig(n_sky=200))
    fe = MatchedFilterFrontEnd(bank, psds, w)
    x = noise[:4]
    ref = torch.stack([det.incoherent(det.snr_series(det.whiten(xi))) for xi in x])
    out = fe(torch.as_tensor(x))[:, 3]
    torch.testing.assert_close(out, ref.float(), rtol=2e-3, atol=2e-3)
