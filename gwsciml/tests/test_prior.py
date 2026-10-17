"""EM prior fit on synthetic likelihood maps with known weights."""

import numpy as np

from gwsciml.pe.prior import fit_prior, log_lambda


def test_em_recovers_weights_and_increases_objective():
    rng = np.random.default_rng(0)
    J, B, n = 6, 4, 4000
    pi = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    tau = np.array([0.1, 0.1, 0.2, 0.6])
    # each sample: noise map ~ log-mean-exp of chi2-like values, plus a strong peak at its true (j, b)
    lmap = rng.normal(0, 1, (n, J, B))
    j = rng.choice(J, n, p=pi)
    b = rng.choice(B, n, p=tau)
    lmap[np.arange(n), j, b] += 12.0
    log_pi, log_tau, trace = fit_prior(lmap, alpha=0.0)
    assert np.all(np.diff(trace) > -1e-9)                      # EM never decreases the objective
    np.testing.assert_allclose(np.exp(log_pi), pi, atol=0.03)
    np.testing.assert_allclose(np.exp(log_tau), tau, atol=0.03)
    uniform = log_lambda(lmap, np.full(J, -np.log(J)), np.full(B, -np.log(B))).mean()
    assert log_lambda(lmap, log_pi, log_tau).mean() > uniform
