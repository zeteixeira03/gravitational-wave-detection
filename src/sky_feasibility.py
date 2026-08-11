"""
Offline feasibility check for S2 geometric cross-correlation (Phase 3, Step 2).

Computes sky maps from detector cross-correlations, decomposes them into
spherical harmonic coefficients, and measures discriminative power as a
univariate classifier. Gate: if ROC-AUC < 0.55, sky maps are too noisy
at low SNR to be useful.

Usage:
    python -m src.sky_feasibility --n-samples 5000 --n-pix 192 --l-max 8
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# scipy.special.sph_harm was deprecated in scipy 1.15 and removed thereafter in
# favour of sph_harm_y, which takes (l, m, theta, phi) instead of (m, l, phi, theta).
# Kaggle images track a newer scipy than local, so bind a single sph_harm(m, l, phi, theta)
# callable that works on both.
try:
    from scipy.special import sph_harm as _sph_harm

    def sph_harm(m, l, phi, theta):
        return _sph_harm(m, l, phi, theta)
except ImportError:
    from scipy.special import sph_harm_y as _sph_harm_y

    def sph_harm(m, l, phi, theta):
        return _sph_harm_y(l, m, theta, phi)

from data import FS, N, load_psd, preprocess_sample
from data.g2net import find_dataset_dir, load_labels, load_sample


# ============================================================================================
#                                     detector geometry
# ============================================================================================

# ECEF coordinates (metres). source: LIGO-T980044, Virgo TDS VIR-0984A-15
DETECTOR_POSITIONS = {
    "H1": np.array([-2.16142e6, -3.83469e6,  4.60035e6]),
    "L1": np.array([-7.42760e4, -5.49628e6,  3.22425e6]),
    "V1": np.array([ 4.54638e6,  8.42990e5,  4.37849e6]),
}

C_LIGHT = 299_792_458.0  # m/s

PAIRS = [("H1", "L1"), ("H1", "V1"), ("L1", "V1")]

DET_INDEX = {"H1": 0, "L1": 1, "V1": 2}


# ============================================================================================
#                                        sky grid
# ============================================================================================

def make_sky_grid(n_pix: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a roughly uniform grid on S2.

    Parameters
    ----------
    n_pix
        Target number of sky pixels. Actual count may differ due to rounding.

    Returns
    -------
    theta
        Colatitude in [0, pi], shape (actual_n_pix,).
    phi
        Azimuth in [0, 2*pi), shape (actual_n_pix,).
    """
    n_theta = int(np.round(np.sqrt(n_pix / 2)))
    n_phi = int(np.round(2 * n_theta))

    # uniform in cos(theta) for equal-area bands
    theta = np.arccos(np.linspace(-1, 1, n_theta))
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    theta_grid, phi_grid = np.meshgrid(theta, phi)
    return theta_grid.ravel(), phi_grid.ravel()


# ============================================================================================
#                                     time delay model
# ============================================================================================

def compute_time_delays(
    theta: np.ndarray,
    phi: np.ndarray,
) -> dict[tuple[str, str], np.ndarray]:
    """
    Compute inter-detector time delays at each sky pixel (plane-wave approximation).

    Parameters
    ----------
    theta
        Colatitude array, shape (n_pix,).
    phi
        Azimuth array, shape (n_pix,).

    Returns
    -------
    delays
        Maps each detector pair to an array of delays in seconds, shape (n_pix,).
        Positive means signal arrives at the first detector first.
    """
    n_hat = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])

    delays = {}
    for det_i, det_j in PAIRS:
        baseline = DETECTOR_POSITIONS[det_i] - DETECTOR_POSITIONS[det_j]
        delays[(det_i, det_j)] = -n_hat @ baseline / C_LIGHT

    return delays


def delay_to_sample_index(delay_sec: np.ndarray) -> np.ndarray:
    """Convert time delays in seconds to nearest integer sample offsets at FS."""
    return np.rint(delay_sec * FS).astype(int)


# ============================================================================================
#                                    cross-correlation
# ============================================================================================

_XCORR_LEN = 2 * N - 1
_XCORR_FFT_LEN = 2 * N  # next power of 2 from 8191 -> 8192


def normalized_cross_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Full normalized cross-correlation between two 1-D signals via FFT.

    Parameters
    ----------
    x, y
        Detector time series, each shape (N,).

    Returns
    -------
    rho
        Normalized cross-correlation in [-1, 1], shape (2*N - 1,).
        Zero-lag at index N - 1.
    """
    if np.all(x == 0) or np.all(y == 0):
        return np.zeros(_XCORR_LEN)

    # zero-pad to next power of 2 for fast FFT (8192 vs 8191)
    X = np.fft.rfft(x, n=_XCORR_FFT_LEN)
    Y = np.fft.rfft(y, n=_XCORR_FFT_LEN)

    x_corr = np.fft.irfft(X * np.conj(Y), n=_XCORR_FFT_LEN)
    # extract valid correlation length and shift so zero-lag lands at index N-1
    x_corr = np.concatenate([x_corr[-N+1:], x_corr[:N]])

    norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
    if norm < 1e-30:
        return np.zeros(_XCORR_LEN)
    return x_corr / norm


def evaluate_xcorr_at_delay(rho: np.ndarray, delay_samples: np.ndarray) -> np.ndarray:
    """
    Look up cross-correlation values at specific lag offsets.

    Parameters
    ----------
    rho
        Output of normalized_cross_correlation, shape (2*N - 1,).
    delay_samples
        Integer lag offsets, shape (n_pix,). One per sky pixel.

    Returns
    -------
    values
        Cross-correlation at each requested lag, shape (n_pix,).
    """
    indices = np.clip(delay_samples + (N - 1), 0, len(rho) - 1)
    return rho[indices]


# ============================================================================================
#                                        sky map
# ============================================================================================

def build_sky_map(
    signal: np.ndarray,
    delay_indices: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    """
    Compute the consistency sky map for one preprocessed sample.

    For each sky pixel k, C(k) = sum over detector pairs of rho_ij(tau_ij(k))^2.

    Parameters
    ----------
    signal
        Whitened signal, shape (3, 4096). Channel order: H1, L1, V1.
    delay_indices
        Maps each detector pair to integer sample delays per pixel.

    Returns
    -------
    sky_map
        Consistency score per pixel, shape (n_pix,).
    """
    n_pix = len(next(iter(delay_indices.values())))
    sky_map = np.zeros(n_pix)

    for det_i, det_j in PAIRS:
        rho = normalized_cross_correlation(
            signal[DET_INDEX[det_i]], signal[DET_INDEX[det_j]]
        )
        values = evaluate_xcorr_at_delay(rho, delay_indices[(det_i, det_j)])
        sky_map += values ** 2

    return sky_map


# ============================================================================================
#                                spherical harmonic decomposition
# ============================================================================================

def compute_sh_matrix(
    theta: np.ndarray,
    phi: np.ndarray,
    l_max: int,
) -> np.ndarray:
    """
    Build the real spherical harmonic design matrix.

    Parameters
    ----------
    theta
        Colatitude array, shape (n_pix,).
    phi
        Azimuth array, shape (n_pix,).
    l_max
        Maximum degree. Produces (l_max + 1)^2 coefficients.

    Returns
    -------
    Y
        Design matrix, shape (n_pix, n_coeffs). Each column is one real SH
        basis function evaluated at all sky pixels. Only depends on the grid.
    """
    n_coeffs = (l_max + 1) ** 2
    n_pix = len(theta)
    Y = np.zeros((n_pix, n_coeffs))

    # column j = l^2 + l + m
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            j = l**2 + l + m
            ylm = sph_harm(m, l, phi, theta)

            # complex -> real conversion
            if m > 0:
                Y[:, j] = np.sqrt(2) * np.real(ylm)
            elif m < 0:
                Y[:, j] = np.sqrt(2) * np.imag(ylm)
            else:
                Y[:, j] = np.real(ylm)

    return Y


def decompose_sky_map(sky_map: np.ndarray, sh_matrix: np.ndarray) -> np.ndarray:
    """
    Project a sky map onto spherical harmonic coefficients via least-squares.

    Parameters
    ----------
    sky_map
        Consistency scores, shape (n_pix,).
    sh_matrix
        Design matrix from compute_sh_matrix, shape (n_pix, n_coeffs).

    Returns
    -------
    coeffs
        SH coefficients, shape (n_coeffs,).
    """
    return np.linalg.lstsq(sh_matrix, sky_map, rcond=None)[0]


# ============================================================================================
#                                       sky geometry
# ============================================================================================

class SkyGeometry:
    """
    Precomputed sky grid, time delays, and SH basis for on-the-fly
    spherical harmonic coefficient extraction from preprocessed signals.

    Precomputes the pseudoinverse of the SH design matrix so that
    decomposition is a single matrix-vector multiply (~0.006 ms) instead
    of a least-squares solve (~10 ms).
    """

    def __init__(self, n_pix: int = 192, l_max: int = 8):
        theta, phi = make_sky_grid(n_pix)
        delays = compute_time_delays(theta, phi)
        self.delay_indices = {pair: delay_to_sample_index(d) for pair, d in delays.items()}
        sh_matrix = compute_sh_matrix(theta, phi, l_max)
        self.sh_pinv = np.linalg.pinv(sh_matrix).astype(np.float32)
        self.n_coeffs = (l_max + 1) ** 2

    def extract(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute SH coefficients from a preprocessed signal.

        Parameters
        ----------
        signal
            Whitened signal, shape (3, 4096).

        Returns
        -------
        coeffs
            SH coefficients, shape (n_coeffs,).
        """
        sky_map = build_sky_map(signal, self.delay_indices)
        return self.sh_pinv @ sky_map


# ============================================================================================
#                        spherical harmonic algebra (readout variants + tests)
# ============================================================================================
#
# The real SH coefficients a_j (j = l^2 + l + m) transform covariantly under SO(3):
# each degree l rotates within its own (2l+1)-dimensional block by a real Wigner-D
# matrix. The angular power spectrum p_l = sum_m a_{lm}^2 is the squared norm of each
# block and is therefore SO(3)-invariant. These helpers give the block index (for the
# `power` readout) and a real-basis rotation (for the invariance unit tests).

import math


def sh_block_index(l_max: int) -> np.ndarray:
    """
    Map each real-SH coefficient index to its degree l.

    Returns
    -------
    block_index
        Array of length (l_max + 1)^2 where entry j holds the degree l of the
        coefficient at index j = l^2 + l + m.
    """
    idx = np.empty((l_max + 1) ** 2, dtype=np.int64)
    for l in range(l_max + 1):
        idx[l * l : (l + 1) * (l + 1)] = l
    return idx


def power_spectrum(coeffs: np.ndarray, l_max: int) -> np.ndarray:
    """
    Angular power spectrum p_l = sum_m a_{lm}^2 from real-SH coefficients.

    Parameters
    ----------
    coeffs
        Real-SH coefficients, shape (..., (l_max + 1)^2).
    l_max
        Maximum degree.

    Returns
    -------
    power
        Per-degree power, shape (..., l_max + 1).
    """
    block = sh_block_index(l_max)
    sq = coeffs ** 2
    out = np.zeros(coeffs.shape[:-1] + (l_max + 1,), dtype=coeffs.dtype)
    for l in range(l_max + 1):
        out[..., l] = sq[..., block == l].sum(axis=-1)
    return out


def _wigner_d_small(l: int, beta: float) -> np.ndarray:
    """
    Wigner (small) d-matrix d^l_{m'm}(beta) in the complex SH basis.

    Rows and columns are ordered m = -l, ..., +l.
    """
    dim = 2 * l + 1
    d = np.zeros((dim, dim))
    c = math.cos(beta / 2.0)
    s = math.sin(beta / 2.0)
    for a, mp in enumerate(range(-l, l + 1)):
        for b, m in enumerate(range(-l, l + 1)):
            pref = math.sqrt(
                math.factorial(l + mp) * math.factorial(l - mp)
                * math.factorial(l + m) * math.factorial(l - m)
            )
            s_min = max(0, m - mp)
            s_max = min(l + m, l - mp)
            total = 0.0
            for k in range(s_min, s_max + 1):
                denom = (
                    math.factorial(l + m - k) * math.factorial(k)
                    * math.factorial(mp - m + k) * math.factorial(l - mp - k)
                )
                cpow = 2 * l + m - mp - 2 * k
                spow = mp - m + 2 * k
                total += ((-1) ** (k)) * (c ** cpow) * (s ** spow) / denom
            d[a, b] = pref * total * ((-1) ** (mp - m))
    return d


def _wigner_D_complex(l: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """Complex Wigner-D matrix D^l_{m'm}(alpha, beta, gamma), m ordered -l..+l."""
    d = _wigner_d_small(l, beta)
    m = np.arange(-l, l + 1)
    left = np.exp(-1j * m * alpha)[:, None]
    right = np.exp(-1j * m * gamma)[None, :]
    return left * d * right


def _complex_to_real_block(l: int) -> np.ndarray:
    """
    Unitary matrix C (per degree l) mapping complex SH coefficients to the real
    convention used by ``compute_sh_matrix``: r = C c, with c and r ordered m=-l..+l.

    Convention (matching compute_sh_matrix, scipy CS phase absorbed):
        r_m =  sqrt(2) * Re(c_m)          for m > 0
        r_0 =  c_0
        r_m =  sqrt(2) * Im(c_m)          for m < 0
    """
    dim = 2 * l + 1
    C = np.zeros((dim, dim), dtype=complex)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for a, m in enumerate(range(-l, l + 1)):
        pos = m + l       # column of c_{+|m|}
        neg = -m + l      # column of c_{-|m|}
        if m > 0:
            # sqrt2 Re(c_m) = (1/sqrt2)(c_m + c_{-m}(-1)^m); conj(c_m)=(-1)^m c_{-m}
            C[a, pos] = inv_sqrt2
            C[a, neg] = inv_sqrt2 * ((-1) ** m)
        elif m < 0:
            # sqrt2 Im(c_m) = (1/(sqrt2 i))(c_m - (-1)^m c_{-m})
            C[a, pos] = inv_sqrt2 / 1j
            C[a, neg] = -inv_sqrt2 * ((-1) ** m) / 1j
        else:
            C[a, pos] = 1.0
    return C


def real_sh_rotation_matrix(l_max: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Block-diagonal real orthogonal matrix realising an SO(3) rotation (ZYZ Euler
    angles) on real-SH coefficient vectors of length (l_max + 1)^2.

    Applying this matrix to a coefficient vector is exactly the transform induced
    by rotating the underlying field on the sphere: each degree l rotates within
    its own (2l+1)-block, so the per-degree power p_l is preserved.
    """
    n = (l_max + 1) ** 2
    R = np.zeros((n, n))
    for l in range(l_max + 1):
        Dc = _wigner_D_complex(l, alpha, beta, gamma)
        C = _complex_to_real_block(l)
        Dr = C @ Dc @ C.conj().T
        blk = slice(l * l, (l + 1) * (l + 1))
        R[blk, blk] = Dr.real
    return R


def random_real_sh_rotation(l_max: int, rng: np.random.Generator) -> np.ndarray:
    """Real-SH rotation matrix for a uniformly random SO(3) rotation."""
    alpha = rng.uniform(0, 2 * np.pi)
    beta = math.acos(rng.uniform(-1, 1))
    gamma = rng.uniform(0, 2 * np.pi)
    return real_sh_rotation_matrix(l_max, alpha, beta, gamma)


def wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    """
    Wigner 3-j symbol via the Racah formula. Returns 0 outside the selection rules.
    """
    if m1 + m2 + m3 != 0:
        return 0.0
    if not (abs(j1 - j2) <= j3 <= j1 + j2):
        return 0.0
    if any(abs(m) > j for m, j in [(m1, j1), (m2, j2), (m3, j3)]):
        return 0.0

    def f(n):
        return math.factorial(n)

    tri = (f(j1 + j2 - j3) * f(j1 - j2 + j3) * f(-j1 + j2 + j3)) / f(j1 + j2 + j3 + 1)
    pref = math.sqrt(
        tri
        * f(j1 + m1) * f(j1 - m1) * f(j2 + m2) * f(j2 - m2)
        * f(j3 + m3) * f(j3 - m3)
    )
    t_min = max(0, j2 - j3 - m1, j1 - j3 + m2)
    t_max = min(j1 + j2 - j3, j1 - m1, j2 + m2)
    total = 0.0
    for t in range(t_min, t_max + 1):
        denom = (
            f(t) * f(j3 - j2 + t + m1) * f(j3 - j1 + t - m2)
            * f(j1 + j2 - j3 - t) * f(j1 - t - m1) * f(j2 - t + m2)
        )
        total += ((-1) ** t) / denom
    return ((-1) ** (j1 - j2 - m3)) * pref * total


def bispectrum_terms(l_max: int, l_bisp: int) -> dict:
    """
    Precompute the real-basis trilinear terms of the SH bispectrum subset.

    For each degree triple (l1 <= l2 <= l3 <= l_bisp) satisfying the triangle
    inequality, the rotation-invariant scalar
        B_{l1 l2 l3} = sum_{m1 m2 m3} (l1 l2 l3; m1 m2 m3) c_{l1 m1} c_{l2 m2} c_{l3 m3}
    is expanded into the real-SH basis (c = C^H r), giving a real symmetric
    trilinear form B = sum_k w_k r_{i_k} r_{j_k} r_{p_k}. Each feature is one
    triple; terms are flattened for a single index_add over the batch.

    Returns
    -------
    dict with arrays: feature (term -> feature index), i, j, p (real-coef global
    indices), w (real weight), and n_features (number of triples kept).
    """
    cinv = {l: _complex_to_real_block(l).conj().T for l in range(l_bisp + 1)}  # [m, a]

    feature, ii, jj, pp, ww = [], [], [], [], []
    fidx = 0
    for l1 in range(l_bisp + 1):
        for l2 in range(l1, l_bisp + 1):
            for l3 in range(l2, l_bisp + 1):
                if not (abs(l1 - l2) <= l3 <= l1 + l2):
                    continue
                dim1, dim2, dim3 = 2 * l1 + 1, 2 * l2 + 1, 2 * l3 + 1
                T = np.zeros((dim1, dim2, dim3), dtype=complex)
                # c_{l,m} = sum_a cinv[l][m, a] r_{l,a}; substitute into the 3j
                # contraction to get a real trilinear form in the real coefficients.
                for a1, m1 in enumerate(range(-l1, l1 + 1)):
                    for a2, m2 in enumerate(range(-l2, l2 + 1)):
                        m3 = -(m1 + m2)
                        if abs(m3) > l3:
                            continue
                        w3 = wigner_3j(l1, l2, l3, m1, m2, m3)
                        if w3 == 0.0:
                            continue
                        a3 = m3 + l3
                        T += w3 * np.multiply.outer(
                            np.multiply.outer(cinv[l1][m1 + l1], cinv[l2][m2 + l2]),
                            cinv[l3][a3],
                        )
                Tr = T.real
                base1, base2, base3 = l1 * l1, l2 * l2, l3 * l3
                nz = np.argwhere(np.abs(Tr) > 1e-8)
                if len(nz) == 0:
                    continue
                for b1, b2, b3 in nz:
                    feature.append(fidx)
                    ii.append(base1 + b1)
                    jj.append(base2 + b2)
                    pp.append(base3 + b3)
                    ww.append(float(Tr[b1, b2, b3]))
                fidx += 1

    return {
        "feature": np.array(feature, dtype=np.int64),
        "i": np.array(ii, dtype=np.int64),
        "j": np.array(jj, dtype=np.int64),
        "p": np.array(pp, dtype=np.int64),
        "w": np.array(ww, dtype=np.float32),
        "n_features": fidx,
    }


# ============================================================================================
#                                       visualization
# ============================================================================================

def plot_sky_map(
    theta: np.ndarray,
    phi: np.ndarray,
    sky_map: np.ndarray,
    title: str = "",
    figsize: tuple = (10, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a single sky map on a Mollweide projection.

    Parameters
    ----------
    theta
        Colatitude array, shape (n_pix,).
    phi
        Azimuth array, shape (n_pix,).
    sky_map
        Consistency score per pixel, shape (n_pix,).
    title
        Figure title.
    figsize
        Figure dimensions.
    save_path
        If provided, save figure to this path.
    """
    lat = np.pi / 2 - theta
    lon = phi - np.pi

    fig, ax = plt.subplots(subplot_kw={"projection": "mollweide"}, figsize=figsize)
    sc = ax.scatter(lon, lat, c=sky_map, cmap="inferno", s=12, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="C(k)", shrink=0.6)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sky_maps_grid(
    sample_ids: list[str],
    labels: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    delay_indices: dict,
    avg_psd: np.ndarray,
    dataset_dir,
    n_cols: int = 3,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot sky maps for multiple samples in a grid with shared color scale.

    Parameters
    ----------
    sample_ids
        Sample IDs to plot.
    labels
        Ground truth labels (0 or 1) for each sample.
    theta, phi
        Sky grid arrays from make_sky_grid.
    delay_indices
        Precomputed delay index dict.
    avg_psd
        Average PSD for preprocessing.
    dataset_dir
        Path to dataset root.
    n_cols
        Number of columns in the grid.
    save_path
        If provided, save figure to this path.
    """
    lat = np.pi / 2 - theta
    lon = phi - np.pi

    n = len(sample_ids)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        subplot_kw={"projection": "mollweide"},
        figsize=(5 * n_cols, 3.5 * n_rows),
    )
    axes = np.atleast_2d(axes)

    # compute all sky maps first for consistent color scale
    sky_maps = []
    for sid in sample_ids:
        raw = load_sample(sid, split="train", dataset_dir=dataset_dir)
        sig = preprocess_sample(raw, avg_psd)
        sky_maps.append(build_sky_map(sig, delay_indices))

    all_vals = np.concatenate(sky_maps)
    vmin, vmax = all_vals.min(), all_vals.max()

    for idx, (sid, label, sm) in enumerate(zip(sample_ids, labels, sky_maps)):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        sc = ax.scatter(lon, lat, c=sm, cmap="inferno", s=10, edgecolors="none",
                        vmin=vmin, vmax=vmax)
        tag = "signal" if label == 1 else "noise"
        ax.set_title(f"{sid[:8]}... ({tag})", fontsize=9)
        ax.grid(True, alpha=0.3)

    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.colorbar(sc, ax=axes, label="C(k)", shrink=0.6, location="bottom", pad=0.08)
    fig.suptitle("sky map consistency scores", fontsize=12)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ============================================================================================
#                                      feasibility gate
# ============================================================================================

def evaluate_feasibility(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Measure discriminative power of a scalar sky-map feature.

    Parameters
    ----------
    features
        Scalar feature per sample, shape (n_samples,).
    labels
        Ground truth labels, shape (n_samples,).

    Returns
    -------
    results
        Dict with keys: auc, mean_pos, mean_neg, separation.
    """
    auc = roc_auc_score(labels, features)
    mean_pos = features[labels == 1].mean()
    mean_neg = features[labels == 0].mean()
    separation = mean_pos - mean_neg

    print("PROCEED" if auc >= 0.55 else "SKIP")

    return {"auc": auc, "mean_pos": mean_pos, "mean_neg": mean_neg, "separation": separation}


# ============================================================================================
#                                        main loop
# ============================================================================================

def run_feasibility(n_samples: int = 5000, n_pix: int = 192, l_max: int = 8) -> None:
    """
    End-to-end feasibility check for S2 sky maps.

    Loads samples, builds sky maps, extracts scalar features and SH coefficients,
    and reports AUC for each as a univariate classifier.

    Parameters
    ----------
    n_samples
        Total samples (half positive, half negative).
    n_pix
        Target sky grid resolution.
    l_max
        Maximum SH degree.
    """
    dataset_dir = find_dataset_dir()
    df = load_labels(dataset_dir)
    avg_psd = load_psd(dataset_dir / "avg_psd.npz")

    # sky geometry (precompute once)
    theta, phi = make_sky_grid(n_pix)
    delays = compute_time_delays(theta, phi)
    delay_indices = {pair: delay_to_sample_index(d) for pair, d in delays.items()}
    sh_matrix = compute_sh_matrix(theta, phi, l_max)

    # balanced sample selection
    positives = df[df["target"] == 1]["id"].values
    negatives = df[df["target"] == 0]["id"].values
    rng = np.random.default_rng(42)
    half = n_samples // 2
    chosen_pos = rng.choice(positives, size=half, replace=False)
    chosen_neg = rng.choice(negatives, size=half, replace=False)
    sample_ids = np.concatenate([chosen_pos, chosen_neg])
    labels = np.array([1] * half + [0] * half)

    # main loop
    all_sh_coeffs = np.empty((n_samples, (l_max + 1) ** 2))
    sky_map_max = np.empty(n_samples)
    sky_map_std = np.empty(n_samples)

    for i, sid in enumerate(sample_ids):
        if i % 500 == 0:
            print(f"  {i}/{n_samples}")

        raw = load_sample(sid, split="train", dataset_dir=dataset_dir)
        sig = preprocess_sample(raw, avg_psd)

        sky_map = build_sky_map(sig, delay_indices)
        all_sh_coeffs[i] = decompose_sky_map(sky_map, sh_matrix)
        sky_map_max[i] = sky_map.max()
        sky_map_std[i] = sky_map.std()

    # gate test: compare scalar features
    monopole = all_sh_coeffs[:, 0]

    print("\nscalar feature AUCs:")
    for name, feat in [("l=0 monopole", monopole), ("max(C(k))", sky_map_max), ("std(C(k))", sky_map_std)]:
        auc = roc_auc_score(labels, feat)
        mp = feat[labels == 1].mean()
        mn = feat[labels == 0].mean()
        print(f"  {name:15s}  AUC = {auc:.4f}   mean(pos) = {mp:.2e}   mean(neg) = {mn:.2e}")

    best_auc = max(roc_auc_score(labels, f) for f in [monopole, sky_map_max, sky_map_std])
    print(f"\nbest AUC = {best_auc:.4f} -- {'PROCEED' if best_auc >= 0.55 else 'SKIP'}")

    # per-coefficient AUC
    n_coeffs = all_sh_coeffs.shape[1]
    print(f"\nSH coefficient AUCs (l_max={l_max}, {n_coeffs} coefficients):")
    for j in range(n_coeffs):
        auc_j = roc_auc_score(labels, np.abs(all_sh_coeffs[:, j]))
        if auc_j > 0.53:
            l = int(np.floor(np.sqrt(j)))
            m = j - l * l - l
            print(f"  (l={l}, m={m:+d})  AUC = {auc_j:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S2 sky map feasibility check")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-pix", type=int, default=192)
    parser.add_argument("--l-max", type=int, default=8)
    args = parser.parse_args()
    run_feasibility(args.n_samples, args.n_pix, args.l_max)
