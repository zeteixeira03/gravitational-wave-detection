"""Template bank: start frequencies, whitened templates, stochastic placement.

Duration limit. The data are T = 2 s long, so a template longer than the data
wraps around in the circular matched filter and its early part correlates with
unrelated data. Each template therefore starts at the frequency where the
leading-order (Newtonian) time to coalescence equals max_duration:
    tau0(f) = (5/256) (pi f)^(-8/3) Mc^(-5/3) = max_duration
    =>  f_start = (5 / (256 max_duration))^(3/8) / pi * Mc^(-5/8),
and f_start = max(f_start, f_lower). The higher PN terms make the true duration
slightly longer; max_duration is chosen with margin for that.

Placement. With unit-norm whitened templates a, b, the match is
m(a, b) = max over time shift and phase of |<a|b>| (gwsciml.filtering.matched).
Greedy stochastic placement: walk through candidates in random order, accept a
candidate when its best match with every accepted template is below the
minimal match MM. Every candidate then has a template with match >= MM, so the
SNR lost to the discreteness of the bank is at most 1 - MM for points drawn
from the candidate distribution.
"""

import numpy as np
import torch

from gwsciml.constants import DT, N
from gwsciml.filtering.matched import band_mask, complex_snr, whiten_template
from gwsciml.templates.waveforms import MTSUN_S, chirp_mass, lal_fd_waveform, taylorf2

FREQS = np.fft.rfftfreq(N, DT)


def duration_limited_f_start(m1, m2, max_duration: float, f_lower: float = 20.0):
    mc = chirp_mass(m1, m2) * MTSUN_S
    f = (5.0 / (256.0 * max_duration)) ** 0.375 / np.pi * mc ** -0.625
    return np.maximum(f, f_lower)


def make_templates(params: np.ndarray, psd: np.ndarray, band=(20.0, 1000.0), max_duration: float = 1.5,
                   approximant: str = "IMRPhenomD") -> np.ndarray:
    """Unit-norm whitened FD templates for rows (m1, m2, chi1z, chi2z). Returns (T, N/2+1) complex128.

    The time origin is the waveform model's own (peak of the (2,2) mode for
    IMRPhenomD, coalescence for TaylorF2). psd is the PSD the bank is normalised in.
    """
    mask = band_mask(FREQS, band)
    out = np.zeros((len(params), len(FREQS)), dtype=complex)
    for i, (m1, m2, s1, s2) in enumerate(params):
        f0 = float(duration_limited_f_start(m1, m2, max_duration, band[0]))
        if approximant == "TaylorF2":
            h = taylorf2(FREQS, m1, m2, f_lower=f0)
        else:
            h = lal_fd_waveform(FREQS, m1, m2, s1, s2, approximant=approximant, f_lower=f0)
        out[i] = whiten_template(h, psd, mask)
    return out


def sample_candidates(n: int, rng: np.random.Generator, m_min=5.0, m_max=100.0, chi_max=0.0) -> np.ndarray:
    """Candidate (m1, m2, chi1z, chi2z), m1 >= m2, uniform in component masses; spins uniform in [-chi_max, chi_max]."""
    m = rng.uniform(m_min, m_max, (n, 2))
    m1, m2 = m.max(axis=1), m.min(axis=1)
    chi = rng.uniform(-chi_max, chi_max, (n, 2)) if chi_max > 0 else np.zeros((n, 2))
    return np.column_stack([m1, m2, chi])


def stochastic_bank(cand_params: np.ndarray, cand_templates: np.ndarray, min_match: float = 0.97,
                    batch: int = 64, verbose: bool = True):
    """Greedy placement over candidates (in the given order). Returns indices of accepted candidates.

    Candidates are processed in batches: each batch is matched against the
    accepted bank at once, then accepted one by one within the batch (a
    candidate is also checked against earlier acceptances from its own batch).
    """
    # complex64 and blocks of (64 x 256 x N) keep the working set near 0.5 GB
    temps = torch.as_tensor(cand_templates).to(torch.complex64)
    accepted: list[int] = []
    bank = temps[:0]
    for start in range(0, len(temps), batch):
        cb = temps[start:start + batch]
        best = torch.zeros(len(cb))
        if len(bank):
            for s in range(0, len(bank), 256):
                m = complex_snr(cb, bank[s:s + 256]).abs().amax(dim=-1)
                best = torch.maximum(best, m.amax(dim=-1))
        new_local: list[int] = []
        for i in range(len(cb)):
            if best[i] >= min_match:
                continue
            if new_local:
                m = complex_snr(cb[i:i + 1], cb[new_local]).abs().amax(dim=-1).max()
                if m >= min_match:
                    continue
            new_local.append(i)
        accepted.extend(start + i for i in new_local)
        bank = temps[accepted]
        if verbose and (start // batch) % 20 == 0:
            print(f"  {start + len(cb)}/{len(temps)} candidates, bank size {len(accepted)}", flush=True)
    return np.array(accepted)

