"""Frequency-domain GR waveform templates (fixed physics filters, ROADMAP D1).

Fourier convention: h~(f) = int h(t) exp(-2 pi i f t) dt, the same sign as
numpy.fft.rfft. A signal whose coalescence is at time t_c carries the factor
exp(-2 pi i f t_c). Templates here have t_c = 0, so after an inverse FFT the
merger sits at sample 0 and the inspiral wraps to the end of the array; the
matched filter (gwsciml.filtering.matched) relies on this layout.

Units: masses in solar masses, frequencies in Hz, times in s. The physical
amplitude scale (distance, orientation) is irrelevant for normalised
templates, so every function returns the shape up to a constant factor.

TaylorF2 (own implementation)
-----------------------------
Stationary-phase approximation of the post-Newtonian (PN) inspiral of two
non-spinning point masses on quasi-circular orbits. With
    M   = m1 + m2                 total mass, in seconds (times G Msun / c^3)
    eta = m1 m2 / M^2             symmetric mass ratio, 0 < eta <= 1/4
    Mc  = eta^(3/5) M             chirp mass
    v   = (pi M f)^(1/3)          PN expansion parameter (orbital velocity / c)
the template is
    h~(f) = f^(-7/6) exp(-i Psi(f)),
    Psi(f) = 2 pi f t_c - phi_c - pi/4 + 3 / (128 eta v^5) * sum_k phi_k v^k,
with the 3.5PN coefficients phi_k of Buonanno, Iyer, Ochsner, Pan &
Sathyaprakash, PRD 80, 084043 (2009), Eq. (3.18). The amplitude f^(-7/6) is the
leading (Newtonian) order. The template stops at the innermost stable circular
orbit of a test mass, f_isco = 1 / (6^(3/2) pi M), where the PN series fails.
"""

import numpy as np

MTSUN_S = 4.925490947641267e-06  # G Msun / c^3 [s]
EULER_GAMMA = 0.5772156649015329


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def symmetric_mass_ratio(m1, m2):
    return m1 * m2 / (m1 + m2) ** 2


def f_isco(m1, m2):
    """Gravitational-wave frequency at the test-mass ISCO, 1 / (6^1.5 pi M)."""
    return 1.0 / (6.0 ** 1.5 * np.pi * (m1 + m2) * MTSUN_S)


def newtonian_chirp_time(f, mc):
    """Leading-order time from frequency f to coalescence: (5/256) (pi f)^(-8/3) Mc^(-5/3)."""
    return 5.0 / 256.0 * (np.pi * f) ** (-8.0 / 3.0) * (mc * MTSUN_S) ** (-5.0 / 3.0)


def pn_phase_coefficients(eta: float, v: np.ndarray) -> list:
    """phi_k for k = 0..7 (non-spinning, 3.5PN). phi_5 and phi_6 depend on v through logs."""
    v_lso = 1.0 / np.sqrt(6.0)
    pi = np.pi
    return [
        1.0,
        0.0,
        3715.0 / 756.0 + 55.0 / 9.0 * eta,
        -16.0 * pi,
        15293365.0 / 508032.0 + 27145.0 / 504.0 * eta + 3085.0 / 72.0 * eta ** 2,
        pi * (38645.0 / 756.0 - 65.0 / 9.0 * eta) * (1.0 + 3.0 * np.log(v / v_lso)),
        (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * pi ** 2 - 6848.0 / 21.0 * EULER_GAMMA
         - 6848.0 / 21.0 * np.log(4.0 * v)
         + (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * pi ** 2) * eta
         + 76055.0 / 1728.0 * eta ** 2 - 127825.0 / 1296.0 * eta ** 3),
        pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta ** 2),
    ]


def taylorf2_phase(f, m1, m2, pn_order: int = 7):
    """PN part of Psi(f): 3/(128 eta v^5) sum_{k<=pn_order} phi_k v^k (t_c = phi_c = 0, no -pi/4)."""
    M = (m1 + m2) * MTSUN_S
    eta = symmetric_mass_ratio(m1, m2)
    v = (np.pi * M * f) ** (1.0 / 3.0)
    phis = pn_phase_coefficients(eta, v)
    series = sum(phis[k] * v ** k for k in range(pn_order + 1))
    return 3.0 / (128.0 * eta * v ** 5) * series


def taylorf2(freqs, m1, m2, f_lower: float = 20.0, pn_order: int = 7):
    """Complex TaylorF2 template on the grid freqs, zero outside [f_lower, f_isco)."""
    freqs = np.asarray(freqs, dtype=float)
    h = np.zeros(freqs.shape, dtype=complex)
    m = (freqs >= f_lower) & (freqs < f_isco(m1, m2))
    f = freqs[m]
    psi = taylorf2_phase(f, m1, m2, pn_order) - np.pi / 4.0
    h[m] = f ** (-7.0 / 6.0) * np.exp(-1j * psi)
    return h


def lal_fd_waveform(freqs, m1, m2, chi1z: float = 0.0, chi2z: float = 0.0,
                    approximant: str = "IMRPhenomD", f_lower: float = 20.0,
                    inclination: float = 0.0):
    """Plus polarisation from LALSimulation on the uniform grid freqs (needs lalsuite).

    For aligned-spin models at inclination 0 the cross polarisation is i times
    the plus one, so a phase-maximised matched filter needs only h+.
    The returned series uses the same Fourier sign as numpy (checked in tests).
    """
    import lal
    import lalsimulation as ls

    df = freqs[1] - freqs[0]
    f_max = freqs[-1]
    hp, _ = ls.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0.0, 0.0, chi1z, 0.0, 0.0, chi2z,
        1e6 * lal.PC_SI * 100.0, inclination, 0.0, 0.0, 0.0, 0.0,
        df, f_lower, f_max, f_lower, None, ls.GetApproximantFromString(approximant))
    data = hp.data.data
    out = np.zeros(len(freqs), dtype=complex)
    n = min(len(out), len(data))
    out[:n] = data[:n]
    out[freqs < f_lower] = 0.0
    return out
