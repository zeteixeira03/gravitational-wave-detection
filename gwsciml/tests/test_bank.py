"""Template bank: duration rule and coverage."""

import numpy as np
import pytest

from gwsciml.filtering.matched import match
from gwsciml.templates.bank import duration_limited_f_start, make_templates, sample_candidates, stochastic_bank
from gwsciml.templates.waveforms import chirp_mass, newtonian_chirp_time
from gwsciml.tests.helpers import g2net_like_psd


def test_start_frequency_inverts_chirp_time():
    m1, m2 = np.array([30.0, 8.0, 60.0]), np.array([25.0, 6.0, 10.0])
    f0 = duration_limited_f_start(m1, m2, 1.5, f_lower=1.0)
    np.testing.assert_allclose(newtonian_chirp_time(f0, chirp_mass(m1, m2)), 1.5, rtol=1e-10)
    assert np.all(duration_limited_f_start(m1, m2, 100.0, f_lower=20.0) == 20.0)


def test_bank_covers_new_points():
    rng = np.random.default_rng(0)
    S = g2net_like_psd()
    cand = sample_candidates(800, rng, 20.0, 60.0)
    temps = make_templates(cand, S, approximant="TaylorF2")
    acc = stochastic_bank(cand, temps, 0.95, verbose=False)
    test = sample_candidates(100, rng, 20.0, 60.0)
    ff = match(make_templates(test, S, approximant="TaylorF2"), temps[acc]).max(axis=1)
    # 800 candidates cover the space well but not perfectly: most new points reach MM
    assert np.mean(ff >= 0.95) > 0.9
    assert ff.min() > 0.85
