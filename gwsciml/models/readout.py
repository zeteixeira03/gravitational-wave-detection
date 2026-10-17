"""Learned readout on the likelihood map (P3 prototype).

The P2 statistic is log Lambda = LSE_{j,b} (L_jb + log pi_j + log tau_b), with
L_jb = log <exp(rho2_inc / 2)> per template j and time bin b. exp(rho^2 / 2) is the
likelihood ratio maximised over amplitude, i.e. a flat prior on the SNR. With a
proper SNR prior p(A) the per-cell term becomes a different increasing function
g(L). This module learns g (monotone, piecewise linear) and the log-priors a_j, c_b
discriminatively:
    logit = w0 + w1 * LSE_{j,b}( g(L_jb) + a_j + c_b ).
With g = identity, a = log pi, c = log tau it equals the P2 statistic up to an
affine map, so initialising there starts from the EM solution.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonePL(nn.Module):
    """Piecewise-linear g with slope 1 below the first knot and slope 1 + sum_{k' <= k} d_k'
    after knot k, clamped at >= 0.05 so g is strictly increasing. d = 0 gives g(x) = x."""

    def __init__(self, knots):
        super().__init__()
        self.register_buffer("knots", torch.as_tensor(knots, dtype=torch.float32))
        self.dslope = nn.Parameter(torch.zeros(len(knots)))

    def forward(self, x):
        slopes = 1.0 + torch.cumsum(self.dslope, 0)                      # slope after each knot
        slopes = torch.clamp(slopes, min=0.05)
        inc = torch.diff(torch.cat([torch.ones(1, device=x.device), slopes]))  # slope increments
        return x + (inc * F.relu(x[..., None] - self.knots)).sum(-1)


class LikelihoodMapReadout(nn.Module):
    def __init__(self, log_pi: np.ndarray, log_tau: np.ndarray, knots=(2, 4, 6, 8, 12, 16, 24, 32, 48)):
        super().__init__()
        self.a = nn.Parameter(torch.as_tensor(log_pi, dtype=torch.float32))
        self.c = nn.Parameter(torch.as_tensor(log_tau, dtype=torch.float32))
        self.g = MonotonePL(knots)
        self.w = nn.Parameter(torch.tensor([0.0, 1.0]))

    def score(self, L):
        """log Lambda-like score for maps L (B, J, Bt)."""
        return torch.logsumexp((self.g(L) + self.a[:, None] + self.c[None, :]).flatten(1), dim=1)

    def forward(self, L):
        return self.w[0] + self.w[1] * self.score(L)
