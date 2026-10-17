"""
Probe logic on synthetic data with a known answer.

Two worlds, same shapes as the real probe (l_max=2, 9 coefficients):

- redundant: the label depends only on a, and a is a linear function of z plus
  small noise. z must decode a (high R^2) and appending a must add ~no AUC.
- informative: the label depends on a component of a that z does not contain.
  Appending a must raise the AUC, with a CI above zero.
"""
import numpy as np

from probe import probe_features, r2_by_degree

L_MAX = 2
N_COEF = (L_MAX + 1) ** 2


def _world(informative: bool, n=4000, d_z=32, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, d_z))
    W = rng.standard_normal((d_z, N_COEF)) / np.sqrt(d_z)
    A = Z @ W + 0.05 * rng.standard_normal((n, N_COEF))
    hidden = rng.standard_normal(n)
    if informative:
        A[:, 0] = hidden            # monopole carries a component z cannot see
        logit = 2.0 * hidden
    else:
        logit = 2.0 * A[:, 1]       # label is a function of a that z already has
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model_proba = 1 / (1 + np.exp(-logit))
    return Z.astype(np.float32), A.astype(np.float32), y, model_proba


def test_r2_by_degree_blocks():
    r2 = np.arange(N_COEF, dtype=float)
    # l=0 -> [0], l=1 -> [1,2,3], l=2 -> [4..8]
    assert r2_by_degree(r2, L_MAX) == [0.0, 2.0, 6.0]


def test_redundant_world():
    Z, A, y, mp = _world(informative=False)
    out = probe_features(Z, A, y, mp, L_MAX)
    assert out['encoding_r2']['trained']['mean'] > 0.9
    lo, hi = out['delta_z_a']['ci95']
    assert lo <= 0.005 and abs(out['delta_z_a']['delta']) < 0.01


def test_informative_world():
    Z, A, y, mp = _world(informative=True)
    out = probe_features(Z, A, y, mp, L_MAX)
    # z cannot decode the hidden monopole
    assert out['encoding_r2']['trained']['per_degree'][0] < 0.05
    assert out['delta_z_a']['ci95'][0] > 0.05
