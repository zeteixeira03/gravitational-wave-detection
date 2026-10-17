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


def log_lambda(lmap: np.ndarray, log_pi: np.ndarray, log_tau: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """log Lambda for maps (n, J, B), computed in chunks of samples."""
    out = []
    for s in range(0, len(lmap), chunk):
        L = np.asarray(lmap[s:s + chunk], dtype=np.float64)
        out.append(logsumexp(L + log_pi[None, :, None] + log_tau[None, None, :], axis=(1, 2)))
    return np.concatenate(out)


def fit_prior(lmap: np.ndarray, n_iter: int = 500, alpha: float = 0.1, tol: float = 1e-7, verbose: bool = False):
    """MAP-EM for (pi, tau) on maps of positives (n, J, B). Returns log_pi, log_tau, objective trace.

    Implementation: A_ijb = exp(L_ijb - m_i) with m_i = max_jb L_ijb is formed once
    (float32, values in [0, 1]). Then Lambda_i = exp(m_i) pi^T A_i tau and the EM
    sufficient statistics are contractions of A:
        sum_b r_ijb = pi_j (A_i tau)_j / (pi^T A_i tau),
        sum_j r_ijb = tau_b (pi^T A_i)_b / (pi^T A_i tau).
    """
    n, J, B = lmap.shape
    m = np.empty(n)
    A = np.empty((n, J, B), dtype=np.float32)
    for s in range(0, n, 500):
        L = np.asarray(lmap[s:s + 500], dtype=np.float64)
        m[s:s + 500] = L.max(axis=(1, 2))
        A[s:s + 500] = np.exp(L - m[s:s + 500, None, None])
    pi = np.full(J, 1.0 / J)
    tau = np.full(B, 1.0 / B)
    trace = []
    for it in range(n_iter):
        At = A @ tau.astype(np.float32)                                   # (n, J)
        lam = At @ pi                                                     # (n,)
        trace.append(float(np.mean(np.log(lam) + m)))
        pA = np.einsum("j,njb->nb", pi.astype(np.float32), A)            # (n, B)
        new_pi = pi * (At / lam[:, None]).sum(axis=0) + alpha
        new_tau = tau * (pA / lam[:, None]).sum(axis=0) + alpha
        pi, tau = new_pi / new_pi.sum(), new_tau / new_tau.sum()
        if verbose and it % 50 == 0:
            print(f"  EM {it}: mean log Lambda {trace[-1]:.5f}", flush=True)
        if it > 2 and abs(trace[-1] - trace[-2]) < tol:
            break
    return np.log(pi), np.log(tau), trace


def fit_joint_prior(lmap: np.ndarray, n_iter: int = 500, alpha: float = 0.1, tol: float = 1e-7):
    """MAP-EM for a joint weight pi_jb (no product assumption). Returns log_pi (J, B), trace.

    Same one-time tensor A_ijb = exp(L_ijb - m_i) as fit_prior; the M-step is
    pi_jb <- pi_jb sum_i A_ijb / Lambda_i + alpha, normalised. J x B parameters, so
    alpha matters more here than for the product prior.
    """
    n, J, B = lmap.shape
    m = np.empty(n)
    A = np.empty((n, J * B), dtype=np.float32)
    for s in range(0, n, 500):
        L = np.asarray(lmap[s:s + 500], dtype=np.float64).reshape(-1, J * B)
        m[s:s + 500] = L.max(axis=1)
        A[s:s + 500] = np.exp(L - m[s:s + 500, None])
    pi = np.full(J * B, 1.0 / (J * B))
    trace = []
    for _ in range(n_iter):
        lam = A @ pi.astype(np.float32)
        trace.append(float(np.mean(np.log(lam) + m)))
        new = pi * ((1.0 / lam) @ A) + alpha
        pi = new / new.sum()
        if len(trace) > 2 and abs(trace[-1] - trace[-2]) < tol:
            break
    return np.log(pi).reshape(J, B), trace


def log_lambda_joint(lmap: np.ndarray, log_pi: np.ndarray, chunk: int = 1000) -> np.ndarray:
    out = []
    for s in range(0, len(lmap), chunk):
        L = np.asarray(lmap[s:s + chunk], dtype=np.float64)
        out.append(logsumexp(L + log_pi[None], axis=(1, 2)))
    return np.concatenate(out)
