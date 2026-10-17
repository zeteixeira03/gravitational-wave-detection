"""Empirical-Bayes prior over the template bank and merger time (P2 a).

Model. For a sample d and a bank template j with merger time in bin b, the
detector gives the likelihood map L[j, b] = log <exp(rho2_inc_j(t) / 2)>_{t in b}
(gwsciml.detect.classical). With prior weights pi_j (templates) and tau_b (time
bins), the marginalised likelihood ratio of a sample is
    Lambda(d) = sum_{j,b} pi_j tau_b exp(L[j, b]),
which is the Neyman-Pearson statistic for this signal family (up to the
amplitude prior implied by exp(rho^2 / 2)).

Fit. G2Net gives no parameter labels, so (pi, tau) are fitted by maximum
marginal likelihood on train positives: maximise sum_i log Lambda(d_i) over the
two probability simplices. EM: responsibilities r_ijb = pi_j tau_b exp(L_ijb) /
Lambda(d_i); pi_j <- mean_i sum_b r_ijb, tau_b <- mean_i sum_j r_ijb. Each
step cannot decrease the objective (standard EM for mixture weights; the
log-prior splits into log pi_j + log tau_b, so the M-step is exact).
A Dirichlet pseudo-count alpha keeps every weight positive (MAP-EM).
"""

import numpy as np
from scipy.special import logsumexp


def log_lambda(lmap: np.ndarray, log_pi: np.ndarray, log_tau: np.ndarray) -> np.ndarray:
    """log Lambda for maps (n, J, B)."""
    return logsumexp(lmap + log_pi[None, :, None] + log_tau[None, None, :], axis=(1, 2))


def fit_prior(lmap: np.ndarray, n_iter: int = 200, alpha: float = 0.1, tol: float = 1e-7, verbose: bool = False):
    """MAP-EM for (pi, tau) on maps of positives (n, J, B). Returns log_pi, log_tau, objective trace."""
    n, J, B = lmap.shape
    log_pi = np.full(J, -np.log(J))
    log_tau = np.full(B, -np.log(B))
    trace = []
    for it in range(n_iter):
        lr = lmap + log_pi[None, :, None] + log_tau[None, None, :]
        ll = logsumexp(lr, axis=(1, 2))                                   # (n,)
        r = np.exp(lr - ll[:, None, None])                                # responsibilities
        trace.append(float(ll.mean()))
        pi = r.sum(axis=(0, 2)) + alpha
        tau = r.sum(axis=(0, 1)) + alpha
        log_pi, log_tau = np.log(pi / pi.sum()), np.log(tau / tau.sum())
        if verbose and it % 20 == 0:
            print(f"  EM {it}: mean log Lambda {trace[-1]:.5f}")
        if it > 2 and abs(trace[-1] - trace[-2]) < tol:
            break
    return log_pi, log_tau, trace
