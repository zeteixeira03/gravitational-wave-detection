"""Shared synthetic inputs for tests (never used as training data)."""

import numpy as np

from gwsciml.constants import DT, N

FREQS = np.fft.rfftfreq(N, DT)


def g2net_like_psd():
    """Bump at ~12 Hz, roll-off below, bucket ~200 Hz: the shape of the G2Net LIGO noise
    (ASD ~1.5e-21 at 12 Hz against ~3e-24 at 200 Hz, a power ratio of ~2.5e5)."""
    f = np.maximum(FREQS, 0.5)
    asd = 3e-24 * (1 + (f / 400.0) ** 2) ** 0.5 + 1.5e-21 * np.exp(-0.5 * (np.log(f / 12.0) / 0.25) ** 2)
    return asd ** 2
