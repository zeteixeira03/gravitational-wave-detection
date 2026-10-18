"""
Tests for the representation probe (src/probe.py, FOLLOWUP_PLAN.md P1).

The probe turns a trained model into a handful of numbers that decide whether
the redundancy interpretation of SWEEP_FINDINGS F2 survives. Each statistic is
checked here against synthetic data where the right answer is known in advance,
so a wrong verdict on Kaggle cannot come from a broken estimator.
"""

import json

import numpy as np
import torch

from models.diy_model import DIYModel
from probe import (
    odd_variance_fraction, paired_bootstrap_delta, r2_by_degree, read_outcome,
    ridge_r2, run_probe, train_mlp_probe, verify_stored_sh,
)
from sky_feasibility import SkyGeometry, sh_block_index

L_MAX = 10
N_SKY = (L_MAX + 1) ** 2


def test_ridge_recovers_known_linear_map():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((4000, 20))
    Y = Z @ rng.standard_normal((20, 5)) + 0.5 * rng.standard_normal((4000, 5))
    fit = ridge_r2(Z[:2000], Y[:2000], Z[2000:], Y[2000:])
    # signal variance per target ~20 against noise 0.25 -> true R^2 ~0.99
    assert fit['r2'] > 0.95, fit['r2']
    assert len(fit['r2_per_target']) == 5


def test_ridge_on_independent_targets_is_near_zero():
    rng = np.random.default_rng(1)
    Z, Y = rng.standard_normal((4000, 50)), rng.standard_normal((4000, 8))
    fit = ridge_r2(Z[:2000], Y[:2000], Z[2000:], Y[2000:])
    assert abs(fit['r2']) < 0.02, fit['r2']


def test_r2_by_degree_isolates_the_predictable_block():
    # only the l=0 coefficient depends on Z; every l>=1 coefficient is noise
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((4000, 10))
    Y = rng.standard_normal((4000, N_SKY))
    Y[:, 0] = Z @ rng.standard_normal(10)
    fit = ridge_r2(Z[:2000], Y[:2000], Z[2000:], Y[2000:])
    by_l = r2_by_degree(fit, L_MAX)
    assert len(by_l) == L_MAX + 1
    assert by_l[0] > 0.99
    assert max(abs(v) for v in by_l[1:]) < 0.02


def test_odd_variance_fraction_limits():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((500, 6))
    assert odd_variance_fraction(a, a) == 0.0
    assert np.isclose(odd_variance_fraction(a, -a), 1.0)
    b = rng.standard_normal((500, 6))
    assert 0.3 < odd_variance_fraction(a, b) < 0.7  # independent halves: ~1/2


def test_paired_bootstrap_delta():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 4000).astype(float)
    weak = y + 2.0 * rng.standard_normal(4000)
    strong = y + 0.5 * rng.standard_normal(4000)
    same = paired_bootstrap_delta(y, weak, weak, n_boot=100)
    assert same['delta'] == 0.0 and same['ci_lo'] == 0.0 and same['ci_hi'] == 0.0
    better = paired_bootstrap_delta(y, weak, strong, n_boot=100)
    assert better['delta'] > 0 and better['ci_lo'] > 0


def test_read_outcome_rule():
    assert read_outcome({'ci_lo': 0.0002, 'ci_hi': 0.0015}, 0.60) == 'complementary'
    assert read_outcome({'ci_lo': -0.0004, 'ci_hi': 0.0006}, 0.60) == 'redundant'
    assert read_outcome({'ci_lo': -0.0004, 'ci_hi': 0.0020}, 0.60) == 'inconclusive'
    # a CI entirely above zero wins even if it is narrow and below the margin
    assert read_outcome({'ci_lo': 0.0001, 'ci_hi': 0.0005}, 0.60) == 'complementary'
    # "adds nothing" is vacuous when the sky features carry nothing on their own
    assert read_outcome({'ci_lo': -0.0004, 'ci_hi': 0.0006}, 0.52) == 'sky_uninformative'


def test_mlp_probe_detects_complementary_features_and_keeps_rng():
    rng = np.random.default_rng(5)
    n = 6000
    y = rng.integers(0, 2, n).astype(np.float32)
    z = (y[:, None] * 0.6 + rng.standard_normal((n, 8))).astype(np.float32)
    extra = (y[:, None] * 0.6 + rng.standard_normal((n, 4))).astype(np.float32)  # independent evidence
    zx = np.concatenate([z, extra], 1)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    p_z = train_mlp_probe(z[:3000], y[:3000], z[3000:], 'cpu', max_epochs=10)
    p_zx = train_mlp_probe(zx[:3000], y[:3000], zx[3000:], 'cpu', max_epochs=10)
    assert torch.equal(before, torch.get_rng_state()), "probe training moved the caller's RNG"
    d = paired_bootstrap_delta(y[3000:], p_z, p_zx, n_boot=100)
    assert d['ci_lo'] > 0, d


def test_verify_stored_sh():
    rng = np.random.default_rng(6)
    s = rng.standard_normal((10, N_SKY)).astype(np.float32)
    out = verify_stored_sh(s, s.copy())
    assert out['l_max_inferred'] == L_MAX and out['l_max_consistent']
    assert out['max_abs_diff'] == 0.0


def test_run_probe_end_to_end_smoke():
    # tiny model and random signals: checks wiring, shapes and that the record
    # serializes, not any number in it
    torch.manual_seed(0)
    n = 160
    signals = torch.randn(n, 3, 4096)
    labels = torch.tensor([i % 2 for i in range(n)])
    geo = SkyGeometry(n_pix=192, l_max=L_MAX)
    stored = torch.tensor(np.stack([geo.extract(s.numpy()) for s in signals]), dtype=torch.float32)
    model = DIYModel(n_channels=4, n_sky_features=N_SKY, sky_readout="none").eval()
    rand_model = DIYModel(n_channels=4, n_sky_features=N_SKY, sky_readout="none").eval()
    rec = run_probe(model, rand_model, signals, labels, stored, geo, 'cpu',
                    l_max=L_MAX, batch_size=32, n_boot=20, verbose=False)
    json.dumps(rec)
    assert rec['sh_check']['max_rel_diff'] < 1e-5
    # random signals carry no label information, so the gate must fire
    assert rec['outcome'] == 'sky_uninformative', rec['label_auc']
    assert len(rec['ridge']['trained']['alm']['r2_by_degree']) == L_MAX + 1
    assert len(rec['ridge']['trained']['log_power']['r2_per_degree']) == L_MAX + 1
    assert set(rec['label_auc']) >= {'cnn', 'sky', 'cnn+sky', 'model_on_probe_test'}
    # the swap decomposition of the SH vector is exact: even + odd recombine
    assert 0.0 <= rec['swap']['sh_odd_var_frac'] <= 1.0
    assert len(sh_block_index(L_MAX)) == N_SKY
