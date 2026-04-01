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
from scipy.special import sph_harm
from sklearn.metrics import roc_auc_score

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

    print(f"\nscalar feature AUCs:")
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
