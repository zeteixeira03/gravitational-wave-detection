"""Network geometry against LAL and against analytic properties."""

import numpy as np
import pytest

from gwsciml.filtering.network import (C_SI, Network, antenna_patterns, detector_geometry,
                                       geocentre_delay, polarisation_basis, sky_direction)

RNG = np.random.default_rng(7)
LAL_NAMES = {"H1": "LHO_4K_DETECTOR", "L1": "LLO_4K_DETECTOR", "V1": "VIRGO_DETECTOR"}


def random_sky(n):
    return RNG.uniform(-np.pi, np.pi, n), np.arcsin(RNG.uniform(-1, 1, n)), RNG.uniform(0, np.pi, n)


@pytest.mark.parametrize("det", ["H1", "L1", "V1"])
def test_geometry_matches_lal(det):
    lal = pytest.importorskip("lal")
    ref = lal.CachedDetectors[getattr(lal, LAL_NAMES[det])]
    r, D = detector_geometry(det)
    np.testing.assert_allclose(r, np.array(ref.location), atol=1.0)  # metres
    np.testing.assert_allclose(D, np.array(ref.response, dtype=float), atol=2e-6)  # float32 in LAL


@pytest.mark.parametrize("det", ["H1", "L1", "V1"])
def test_antenna_patterns_and_delays_match_lal(det):
    lal = pytest.importorskip("lal")
    ref = lal.CachedDetectors[getattr(lal, LAL_NAMES[det])]
    r, D = detector_geometry(det)
    gps = 1e9
    gmst = lal.GreenwichMeanSiderealTime(gps)
    for lon, lat, psi in zip(*random_sky(20)):
        ra = lon + gmst
        fp_ref, fx_ref = lal.ComputeDetAMResponse(ref.response, ra, lat, psi, gmst)
        fp, fx = antenna_patterns(D, lon, lat, psi)
        assert fp == pytest.approx(fp_ref, abs=1e-5)
        assert fx == pytest.approx(fx_ref, abs=1e-5)
        dt_ref = lal.TimeDelayFromEarthCenter(ref.location, ra, lat, gps)
        assert geocentre_delay(r, lon, lat) == pytest.approx(dt_ref, abs=1e-9)


def test_polarisation_basis_is_orthonormal_and_transverse():
    lon, lat, psi = random_sky(50)
    X, Y = polarisation_basis(lon, lat, psi)
    n = sky_direction(lon, lat)
    for a, b, v in [(X, X, 1), (Y, Y, 1), (X, Y, 0), (X, n, 0), (Y, n, 0)]:
        np.testing.assert_allclose(np.sum(a * b, axis=-1), v, atol=1e-12)


def test_antenna_pattern_psi_rotation():
    # F+ + i Fx picks up exp(-2 i psi): spin-2 transformation of the polarisation tensors
    _, D = detector_geometry("H1")
    lon, lat, psi = random_sky(30)
    f0 = np.array(antenna_patterns(D, lon, lat, 0.0))
    f1 = np.array(antenna_patterns(D, lon, lat, psi))
    np.testing.assert_allclose(f1[0] + 1j * f1[1], (f0[0] + 1j * f0[1]) * np.exp(-2j * psi), atol=1e-12)


def test_light_travel_times():
    net = Network(n_sky=4000)
    # H1-L1 ~10.0 ms, H1-V1 ~27.3 ms, L1-V1 ~26.4 ms
    assert net.max_pair_delay(0, 1) == pytest.approx(0.01001, abs=1e-4)
    assert net.max_pair_delay(0, 2) == pytest.approx(0.02729, abs=2e-4)
    assert net.max_pair_delay(1, 2) == pytest.approx(0.02644, abs=2e-4)
    # pairwise delays on the grid never exceed the baseline light time
    d = net.delays
    assert np.abs(d[:, 0] - d[:, 1]).max() <= net.max_pair_delay(0, 1) + 1e-12


def test_projector_properties():
    net = Network(n_sky=500)
    sigma = np.array([1.0, 1.1, 0.5])
    P = net.projectors(sigma)
    np.testing.assert_allclose(P @ P, P, atol=1e-9)              # idempotent
    np.testing.assert_allclose(P, np.swapaxes(P, 1, 2), atol=1e-12)  # symmetric
    np.testing.assert_allclose(np.trace(P, axis1=1, axis2=2), 2.0, atol=1e-9)  # rank 2
    # a signal z = sigma (F+ a + Fx b) for complex a, b lies entirely in the projected space
    a, b = 1.3 - 0.4j, -0.2 + 0.9j
    z = sigma * (net.fplus * a + net.fcross * b)
    null = z - np.einsum("skl,sl->sk", P, z)
    assert np.abs(null).max() < 1e-9
