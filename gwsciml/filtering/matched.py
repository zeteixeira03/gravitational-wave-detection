"""Matched filtering in whitened units (conventions in gwsciml.data.psd).

Whitened rffts: Dw = rfft(w d) / sqrt(S fs/2) for data, Hw = h~ fs / sqrt(S fs/2)
for a template h~ given as a continuous Fourier transform (fs h~ is the DFT of
the sampled waveform). The inner product is <a|b> = (2/N) Re sum_k Aw_k conj(Bw_k).

Complex SNR time series for a unit-norm template (<h|h> = 1):
    z_j = (2/N) sum_{k=0}^{N/2} Dw_k conj(Hw_k) exp(2 pi i k j / N)
        = 2 * ifft(Z)_j,   Z_k = Dw_k conj(Hw_k) for k <= N/2, 0 above.
z_j is the template with its merger at sample j, maximised over phase by |z_j|.
In Gaussian noise (flat part of the window): Re z, Im z ~ N(0, 1), |z|^2 ~ chi2(2).
For data = noise + s with s = rho h shifted to j0: z_j0 has mean rho e^{i phi}.
"""

import numpy as np
import torch

from gwsciml.constants import FS, N


def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def whiten_template(h_fd: np.ndarray, psd: np.ndarray, mask: np.ndarray | None = None,
                    normalize: bool = True) -> np.ndarray:
    """Whitened template Hw = h~ fs / sqrt(S fs / 2), zeroed outside mask, scaled to <h|h> = 1."""
    hw = h_fd * FS / np.sqrt(psd * FS / 2.0)
    if mask is not None:
        hw = hw * mask
    if normalize:
        hw = hw / np.sqrt(inner(hw, hw))[..., None]
    return hw


def inner(aw: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """<a|b> = (2/N) Re sum_k aw_k conj(bw_k) over the last axis."""
    return (2.0 / N) * np.real(np.sum(aw * np.conj(bw), axis=-1))


def complex_snr(dw: torch.Tensor, hw: torch.Tensor) -> torch.Tensor:
    """Complex SNR series z for every (data, template) pair.

    dw: (..., F) complex whitened data, F = N/2 + 1.
    hw: (T, F) complex unit-norm whitened templates.
    Returns (..., T, N) complex: z[..., t, j] for template t, merger at sample j.
    """
    prod = dw[..., None, :] * torch.conj(hw)
    full = torch.zeros(prod.shape[:-1] + (N,), dtype=prod.dtype, device=prod.device)
    full[..., : prod.shape[-1]] = prod
    return 2.0 * torch.fft.ifft(full, dim=-1)


def match(aw: np.ndarray, bw: np.ndarray) -> np.ndarray:
    """Overlap maximised over time and phase, max_j |<a|b shifted by j>| / norms.

    aw: (A, F), bw: (B, F) whitened templates. Returns (A, B).
    """
    a = torch.as_tensor(aw / np.sqrt(inner(aw, aw))[:, None])
    b = torch.as_tensor(bw / np.sqrt(inner(bw, bw))[:, None])
    return complex_snr(a, b).abs().amax(dim=-1).numpy()
