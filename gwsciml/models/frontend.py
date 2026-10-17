"""Fixed matched-filter front end as a batched torch module (P1 steps 1-3, P3 input).

Same mathematics as gwsciml.detect.classical, vectorised over a batch so it
runs on a GPU inside a training loop. No trainable parameters: the whitening
filter and the whitened unit-norm templates are buffers.

Scale. Raw strain is ~1e-20; its square underflows float32 normal range. The
module multiplies the input by STRAIN_SCALE and the PSD by STRAIN_SCALE^2 first,
which leaves every whitened quantity unchanged.

forward(x) with x (B, 3, N) raw strain returns rho2 maps (B, 4, T, J):
channel 0 |z_H(t)|^2, 1 max_{|d|<=D_HL} |z_L(t+d)|^2, 2 max_{|d|<=D_HV} |z_V(t+d)|^2,
channel 3 their sum (the coincident incoherent statistic), for templates T and
H1 merger times t in t_range, optionally max-pooled in time by `decimate`.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gwsciml.constants import DT, FS, N
from gwsciml.filtering.matched import band_mask, inner, whiten_template
from gwsciml.filtering.network import Network

STRAIN_SCALE = 1e20
FREQS = np.fft.rfftfreq(N, DT)


class MatchedFilterFrontEnd(nn.Module):
    def __init__(self, bank_fd: np.ndarray, psds: np.ndarray, window: np.ndarray, band=(20.0, 1000.0),
                 t_range=(0.1, 1.9), timing_margin: float = 0.002, decimate: int = 1):
        super().__init__()
        mask = band_mask(FREQS, band)
        hw = np.stack([whiten_template(bank_fd, psds[k][None], mask, normalize=False) for k in range(3)])
        self.register_buffer("sigma", torch.as_tensor(np.sqrt(inner(hw, hw)), dtype=torch.float32))
        hw = hw / np.sqrt(inner(hw, hw))[..., None]
        self.register_buffer("hw_conj", torch.as_tensor(np.conj(hw)).to(torch.complex64))      # (3,T,F)
        inv = mask / np.sqrt(psds * STRAIN_SCALE ** 2 * FS / 2.0)
        self.register_buffer("inv_asd", torch.as_tensor(inv, dtype=torch.float32))            # (3,F)
        self.register_buffer("window", torch.as_tensor(window, dtype=torch.float32))
        net = Network(n_sky=200)
        self.max_hl = int(np.ceil((net.max_pair_delay(0, 1) + timing_margin) * FS))
        self.max_hv = int(np.ceil((net.max_pair_delay(0, 2) + timing_margin) * FS))
        self.j_lo, self.j_hi = int(t_range[0] * FS), int(t_range[1] * FS)
        self.decimate = decimate

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.fft.rfft(x.float() * STRAIN_SCALE * self.window, dim=-1)
        return X * self.inv_asd

    def snr(self, dw: torch.Tensor, templates=slice(None)) -> torch.Tensor:
        """(B, 3, T', N) complex SNR."""
        prod = dw[:, :, None, :] * self.hw_conj[:, templates]
        return 2.0 * torch.fft.ifft(prod, n=N, dim=-1)   # zero-pads F -> N: positive frequencies only

    def forward(self, x: torch.Tensor, templates=slice(None)) -> torch.Tensor:
        p = self.snr(self.whiten(x), templates).abs() ** 2                       # (B,3,T,N)
        B, _, T, n = p.shape
        pl = F.max_pool1d(p[:, 1].reshape(B * T, 1, n), 2 * self.max_hl + 1, 1, self.max_hl).reshape(B, T, n)
        pv = F.max_pool1d(p[:, 2].reshape(B * T, 1, n), 2 * self.max_hv + 1, 1, self.max_hv).reshape(B, T, n)
        out = torch.stack([p[:, 0], pl, pv, p[:, 0] + pl + pv], dim=1)[..., self.j_lo:self.j_hi]
        if self.decimate > 1:
            C, J = out.shape[1], out.shape[-1]
            out = F.max_pool1d(out.reshape(B * C * T, 1, J), self.decimate, self.decimate, ceil_mode=True)
            out = out.reshape(B, C, T, -1)
        return out
