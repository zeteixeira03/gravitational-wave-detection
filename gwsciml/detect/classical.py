"""P1 classical detector: whitened template bank -> per-detector complex SNR ->
time-coincident and coherent network statistics, plus a chi-squared test.

Inputs per sample: raw strain x (3, N) for detectors (H1, L1, V1).

Step 1  Whiten each detector with its own average PSD (gwsciml.data.psd; the
        PSD is estimated through the same window, see the window note there).
Step 2  For every template j and detector k: complex SNR z_kj(t), unit-norm
        template in detector k's PSD (gwsciml.filtering.matched). t is the
        template's merger time.
Step 3  Coincident incoherent statistic (time-shift-only network model):
            rho2_inc_j(t) = |z_Hj(t)|^2 + max_{|d| <= D_HL} |z_Lj(t+d)|^2
                                        + max_{|d| <= D_HV} |z_Vj(t+d)|^2,
        D_ab = light travel time between sites (+ a timing margin). Noise
        background ~ max of chi2(6)-like variables. t restricted to t_range.
Step 4  Coherent statistic for the top-K templates of step 3: for sky points s
        and geocentre times t, z_k(t + dt_k(s)) (linear interpolation of the
        complex series) and rho2_coh = z^H P_j(s) z, with P_j the projector of
        gwsciml.filtering.network built from sigma_kj = template norm in
        detector k (relative sensitivity). Null energy z^H (I - P) z.
Step 5  Allen chi-squared per detector at its peak: split the template into p
        bands of equal <h|h>, z_i = band contributions, chi2 = p sum |z_i - z/p|^2
        (2p - 2 dof in Gaussian noise); reweighted SNR (Allen 2005,
        Babak et al. 2013): rho / [(1 + (chi2_r)^3) / 2]^(1/6) when chi2_r > 1.

Returned per sample: a dict of scalar statistics and the maximum-likelihood
point estimate (template index, H1 merger time, sky index) for P2.
Optionally also a likelihood map L[j, b] = log mean_{t in bin b} exp(rho2_inc_j(t) / 2)
over templates j and H1-merger-time bins b; P2 fits prior weights on it.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gwsciml.constants import DT, FS, N
from gwsciml.data.psd import whiten_fd
from gwsciml.filtering.matched import band_mask, inner, whiten_template
from gwsciml.filtering.network import Network

FREQS = np.fft.rfftfreq(N, DT)


@dataclass
class DetectorConfig:
    band: tuple = (20.0, 1000.0)
    t_range: tuple = (0.1, 1.9)     # allowed H1 merger times [s]
    timing_margin: float = 0.002    # added to light travel times [s]
    top_k: int = 16                 # templates passed to the coherent stage
    coh_half_window: int = 24       # +- samples around the step-3 peak (geocentre time)
    n_sky: int = 1000
    chisq_bins: int = 8
    template_chunk: int = 128
    time_bin: float = 0.05          # width of the time bins of the likelihood map [s]


class ClassicalDetector:
    def __init__(self, bank_fd: np.ndarray, psds: np.ndarray, window: np.ndarray, cfg: DetectorConfig = DetectorConfig()):
        """bank_fd: (T, F) unwhitened FD templates (continuous-FT units); psds: (3, F)."""
        self.cfg = cfg
        self.window = window
        self.psds = psds
        mask = band_mask(FREQS, cfg.band)
        self.mask = mask
        hw = np.stack([whiten_template(bank_fd, psds[k][None], mask, normalize=False) for k in range(3)])  # (3,T,F)
        self.sigma = np.sqrt(inner(hw, hw))                                   # (3, T)
        self.hw = torch.as_tensor(hw / self.sigma[..., None]).to(torch.complex64)
        self.net = Network(n_sky=cfg.n_sky)
        d = self.net
        self.max_hl = int(np.ceil((d.max_pair_delay(0, 1) + cfg.timing_margin) * FS))
        self.max_hv = int(np.ceil((d.max_pair_delay(0, 2) + cfg.timing_margin) * FS))
        self.j_lo, self.j_hi = int(cfg.t_range[0] * FS), int(cfg.t_range[1] * FS)
        # chi-squared band edges per detector and template: equal <h|h> per band
        p = cfg.chisq_bins
        cum = np.cumsum(np.abs(hw) ** 2, axis=-1)
        cum /= cum[..., -1:]
        edges = np.stack([np.argmax(cum >= q, axis=-1) for q in np.arange(1, p) / p], axis=-1)  # (3,T,p-1)
        bins = np.zeros(hw.shape, dtype=np.int64)
        for b in range(p - 1):
            bins += (np.arange(hw.shape[-1]) > edges[..., b:b + 1])
        self.chisq_bin = torch.as_tensor(bins)

    # ------------------------------------------------------------------ steps 1-2
    def whiten(self, x: np.ndarray) -> torch.Tensor:
        dw = whiten_fd(x, self.psds, self.window) * self.mask
        return torch.as_tensor(dw).to(torch.complex64)

    def snr_series(self, dw: torch.Tensor, templates=slice(None)) -> torch.Tensor:
        """(3, T', N) complex SNR for the selected templates."""
        hw = self.hw[:, templates]
        prod = dw[:, None, :] * torch.conj(hw)
        full = torch.zeros(prod.shape[:-1] + (N,), dtype=prod.dtype)
        full[..., : prod.shape[-1]] = prod
        return 2.0 * torch.fft.ifft(full, dim=-1)

    # ------------------------------------------------------------------ step 3
    def incoherent(self, z: torch.Tensor):
        """rho2_inc (T', J) over allowed H1 merger times, plus the L1/V1 lag of the max."""
        p = z.abs() ** 2                                                # (3, T', N)
        pl = F.max_pool1d(p[1:2].transpose(0, 1), 2 * self.max_hl + 1, stride=1, padding=self.max_hl).transpose(0, 1)
        pv = F.max_pool1d(p[2:3].transpose(0, 1), 2 * self.max_hv + 1, stride=1, padding=self.max_hv).transpose(0, 1)
        s = p[0] + pl[0] + pv[0]
        return s[:, self.j_lo:self.j_hi]

    # ------------------------------------------------------------------ step 4
    def coherent(self, z: torch.Tensor, tmpl: int, j_peak: int):
        """max over sky and geocentre time near j_peak of z^H P z; returns (rho2_coh, null2, sky, j)."""
        net, hw_ = self.net, self.cfg.coh_half_window
        delays = (net.delays - net.delays[:, :1]) * FS                    # (S, 3) samples, relative to H1
        jg = j_peak + np.arange(-hw_, hw_ + 1)                             # H1 times
        pos = jg[None, None, :] + delays[:, :, None]                       # (S, 3, W)
        lo = np.floor(pos).astype(np.int64)
        frac = torch.as_tensor(pos - lo, dtype=torch.float32)
        lo = np.clip(lo, 0, N - 2)
        zs = torch.stack([z[k] for k in range(3)])                         # (3, N)
        idx = torch.as_tensor(lo)
        k_idx = torch.arange(3)[None, :, None].expand_as(idx)
        a, b = zs[k_idx, idx], zs[k_idx, idx + 1]
        zi = a * (1 - frac) + b * frac                                     # (S, 3, W)
        P = torch.as_tensor(net.projectors(self.sigma[:, tmpl]), dtype=torch.complex64)  # (S,3,3)
        pz = torch.einsum("skl,slw->skw", P, zi)
        coh = torch.einsum("skw,skw->sw", torch.conj(zi), pz).real
        tot = (zi.abs() ** 2).sum(dim=1)
        s, w = np.unravel_index(int(coh.argmax()), coh.shape)
        return float(coh[s, w]), float(tot[s, w] - coh[s, w]), int(s), int(jg[w])

    # ------------------------------------------------------------------ step 5
    def chisq(self, dw: torch.Tensor, tmpl: int, j: np.ndarray):
        """Allen chi2 and reweighted SNR per detector at merger samples j (3,)."""
        p = self.cfg.chisq_bins
        out = []
        for k in range(3):
            phase = torch.exp(2j * np.pi * torch.arange(dw.shape[-1]) * float(j[k]) / N).to(torch.complex64)
            contrib = (2.0 / N) * dw[k] * torch.conj(self.hw[k, tmpl]) * phase  # complex, sums to z
            zb = torch.zeros(p, dtype=torch.complex64).index_add_(0, self.chisq_bin[k, tmpl], contrib)
            zt = zb.sum()
            chi2 = float(p * ((zb - zt / p).abs() ** 2).sum())
            rho = float(zt.abs())
            chi2r = chi2 / (2 * p - 2)
            rw = rho / ((1 + chi2r ** 3) / 2) ** (1 / 6) if chi2r > 1 else rho
            out.append((rho, chi2r, rw))
        return out

    # ------------------------------------------------------------------ per sample
    @property
    def n_time_bins(self) -> int:
        return int(np.ceil((self.j_hi - self.j_lo) / (self.cfg.time_bin * FS)))

    @torch.no_grad()
    def statistics(self, x: np.ndarray, return_map: bool = False):
        dw = self.whiten(x)
        T = self.hw.shape[1]
        best_inc = torch.empty(T)
        best_j = torch.empty(T, dtype=torch.long)
        lse_parts = []
        n_b, w_b = self.n_time_bins, int(round(self.cfg.time_bin * FS))
        lmap = torch.empty(T, n_b, dtype=torch.float32) if return_map else None
        for s in range(0, T, self.cfg.template_chunk):
            sl = slice(s, min(T, s + self.cfg.template_chunk))
            inc = self.incoherent(self.snr_series(dw, sl))                  # (t, J)
            m, j = inc.max(dim=-1)
            best_inc[sl], best_j[sl] = m, j + self.j_lo
            half = inc.double() / 2
            lse_parts.append(torch.logsumexp(half, dim=-1))
            if return_map:
                pad = n_b * w_b - half.shape[-1]
                hp = F.pad(half, (0, pad), value=-np.inf).reshape(half.shape[0], n_b, w_b)
                counts = torch.full((n_b,), float(w_b), dtype=torch.float64)
                counts[-1] = w_b - pad
                lmap[sl] = (torch.logsumexp(hp, dim=-1) - torch.log(counts)).float()
        lse_t = torch.cat(lse_parts)                                        # per template, over time
        out = {
            "inc_max": float(best_inc.max()),
            "inc_lse": float(torch.logsumexp(lse_t, 0) - np.log(lse_t.numel() * (self.j_hi - self.j_lo))),
        }
        top = torch.topk(best_inc, min(self.cfg.top_k, T)).indices.numpy()
        z_top = self.snr_series(dw, torch.as_tensor(top))
        best = None
        for i, t in enumerate(top):
            coh, null, sky, j = self.coherent(z_top[:, i], int(t), int(best_j[t]))
            if best is None or coh > best[0]:
                best = (coh, null, sky, j, int(t), i)
        coh, null, sky, j, tmpl, i = best
        out.update(coh_max=coh, null_at_coh=null, tmpl=tmpl, t_h1=j / FS, sky=sky)
        # per-detector peaks of the best template (for chi2 and P2): search near the coherent time
        z = z_top[:, i].abs()
        jk = []
        for k, m in zip(range(3), (0, self.max_hl, self.max_hv)):
            lo, hi = max(j - m, 0), min(j + m + 1, N)
            jk.append(lo + int(z[k, lo:hi].argmax()))
        out.update({f"rho_{d}": float(z[k, jk[k]]) for k, d in enumerate("HLV")})
        out.update({f"dt_{d}": (jk[k] - jk[0]) / FS for k, d in enumerate("HLV") if k})
        cs = self.chisq(dw, tmpl, np.array(jk))
        out.update({f"chi2r_{d}": cs[k][1] for k, d in enumerate("HLV")})
        out["rw_net2"] = sum(c[2] ** 2 for c in cs)
        out["inc_best_tmpl"] = float(best_inc[tmpl])
        return (out, lmap.numpy()) if return_map else out
