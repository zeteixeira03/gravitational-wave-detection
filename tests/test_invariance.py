"""
Invariance unit tests for the sky readout variants and the H1<->L1 merge.

These are the correctness gate for the whole sweep: if a readout advertised as
SO(3)-invariant is not invariant, or the "symmetric" merge is not swap-invariant,
the experiment measures nothing. Run these before any training.

Claims tested:
- ``power`` and ``bispectrum`` readouts are unchanged by an SO(3) rotation of the
  sky map (applied as the exact real-SH block rotation), to numerical tolerance.
- ``mlp121`` and ``scramble`` readouts DO change under the same rotation.
- ``scramble`` preserves each sample's coefficient multiset (marginals).
- The ``symmetric`` merge is exactly invariant under swapping the H1 and L1 input
  channels; the ``concat`` merge is not.
- ``match_params`` equalizes the conditioning-path parameter count across readouts.
"""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from models.diy_model import DIYModel, SkyReadout, sky_conditioning_dim
from sky_feasibility import random_real_sh_rotation, real_sh_rotation_matrix, power_spectrum, bispectrum_terms

L_MAX = 10
N_SKY = (L_MAX + 1) ** 2
FEAT_DIM = 32
BATCH = 8


def _readout(mode, **kw):
    """Build a readout in double precision with a randomized (non-identity) output
    projection so the modulation actually depends on the sky input. The RNG is
    seeded so the drawn weights (and thus the invariance residual) don't depend on
    test execution order."""
    torch.manual_seed(0)
    m = SkyReadout(mode, N_SKY, FEAT_DIM, L_MAX, **kw).double().eval()
    with torch.no_grad():
        m.fc2.weight.normal_()
        m.fc2.bias.normal_()
    return m


@pytest.fixture
def rotated_pair():
    rng = np.random.default_rng(0)
    sky = torch.tensor(rng.standard_normal((BATCH, N_SKY)), dtype=torch.float64)
    R = torch.tensor(random_real_sh_rotation(L_MAX, rng), dtype=torch.float64)
    sky_rot = sky @ R.T                      # per-sample: sky_rot_i = R @ sky_i
    feats = torch.tensor(rng.standard_normal((BATCH, FEAT_DIM)), dtype=torch.float64)
    return sky, sky_rot, feats


# ------------------------------------------------------------------ rotation is a genuine SO(3) rep

def test_rotation_matrix_is_so3_representation():
    # R(g1 . g2) == R(g1) @ R(g2) to numerical tolerance: this is what makes the
    # rotation a genuine SO(3) representation rather than an arbitrary block-orthogonal
    # matrix (which would preserve power trivially). Without this, the invariance
    # tests below would prove nothing about SO(3).
    l_max = 6
    rng = np.random.default_rng(3)

    def angles():
        return (rng.uniform(0, 2 * np.pi), np.arccos(rng.uniform(-1, 1)), rng.uniform(0, 2 * np.pi))

    g1, g2 = angles(), angles()
    R1 = real_sh_rotation_matrix(l_max, *g1)
    R2 = real_sh_rotation_matrix(l_max, *g2)
    composed = (Rotation.from_euler("ZYZ", g1) * Rotation.from_euler("ZYZ", g2)).as_euler("ZYZ")
    R12 = real_sh_rotation_matrix(l_max, *composed)
    assert np.allclose(R12, R1 @ R2, atol=1e-10)
    # l=1 block is a proper rotation (det +1)
    blk = real_sh_rotation_matrix(1, *g1)[1:4, 1:4]
    assert abs(np.linalg.det(blk) - 1.0) < 1e-9


# ------------------------------------------------------------------ SO(3) invariance

# power reduces to sums of squares in double precision (exact); bispectrum weights
# are stored float32 to match the signal dtype in the real model, so its invariance
# residual is float32-limited -- still ~7 orders below the feature scale.
@pytest.mark.parametrize("mode,atol", [("power", 1e-8), ("bispectrum", 1e-4)])
def test_invariant_readout_unchanged_by_rotation(mode, atol, rotated_pair):
    sky, sky_rot, feats = rotated_pair
    m = _readout(mode)
    with torch.no_grad():
        out = m(feats, sky)
        out_rot = m(feats, sky_rot)
    assert torch.allclose(out, out_rot, atol=atol), \
        f"{mode} readout changed under SO(3) rotation (max {abs(out - out_rot).max():.2e})"


@pytest.mark.parametrize("mode", ["mlp121", "scramble"])
def test_noninvariant_readout_changes_under_rotation(mode, rotated_pair):
    sky, sky_rot, feats = rotated_pair
    m = _readout(mode, scramble_seed=0)
    with torch.no_grad():
        out = m(feats, sky)
        out_rot = m(feats, sky_rot)
    assert not torch.allclose(out, out_rot, atol=1e-6), \
        f"{mode} readout was unexpectedly invariant to rotation"


def test_power_reduce_matches_reference(rotated_pair):
    sky, _, _ = rotated_pair
    m = _readout("power")
    with torch.no_grad():
        p = m._power(sky)
    ref = power_spectrum(sky.numpy(), L_MAX)
    assert np.allclose(p.numpy(), ref, atol=1e-10)


# ------------------------------------------------------------------ scramble control

def test_scramble_preserves_marginals():
    rng = np.random.default_rng(1)
    sky = torch.tensor(rng.standard_normal((BATCH, N_SKY)), dtype=torch.float64)
    m = _readout("scramble", scramble_seed=0)
    with torch.no_grad():
        scr = m._scramble(sky)
    # every row is a permutation of the original row: sorted values match, order does not
    assert torch.allclose(scr.sort(dim=1).values, sky.sort(dim=1).values, atol=1e-12)
    assert not torch.allclose(scr, sky)


def test_scramble_is_fixed_across_calls_and_per_sample():
    # the control is a per-sample permutation fixed across epochs (not a
    # per-forward re-draw, and not a single global permutation the linear
    # readout could absorb). two calls on the same input must match exactly;
    # different samples must get different permutations; a different
    # scramble_seed must give a different permutation.
    rng = np.random.default_rng(2)
    sky = torch.tensor(rng.standard_normal((BATCH, N_SKY)), dtype=torch.float64)
    m = _readout("scramble", scramble_seed=0)
    with torch.no_grad():
        a = m._scramble(sky)
        b = m._scramble(sky)
    assert torch.equal(a, b), "scramble is not deterministic across calls"

    # per-sample: at least two rows must carry different permutations, so the
    # scramble is not one global permutation shared by every sample
    order = a.argsort(dim=1)
    assert not torch.equal(order[0], order[1]), "same permutation reused across samples"

    m2 = _readout("scramble", scramble_seed=1)
    with torch.no_grad():
        c = m2._scramble(sky)
    assert not torch.equal(a, c), "scramble_seed does not change the permutation"


# ------------------------------------------------------------------ H1<->L1 swap (Z2)

def _pooled(model, X):
    with torch.no_grad():
        pooled, _ = model._extract_pooled(X)
    return pooled


# GeM forces float32 inside its forward (an AMP NaN guard), so the full model
# runs in float32; the swap-invariance is nonetheless exact because H1/L1 share
# extractor and branch weights, making the swapped path bit-identical.

def test_symmetric_merge_invariant_to_h1l1_swap():
    torch.manual_seed(0)
    X = torch.randn(BATCH, 3, 4096)
    X_swap = X[:, [1, 0, 2], :]
    for pool in ("mean", "max"):
        model = DIYModel(
            n_channels=4, n_sky_features=N_SKY, sky_readout="none",
            h1l1_merge="symmetric", h1l1_pool=pool,
        ).eval()
        p, ps = _pooled(model, X), _pooled(model, X_swap)
        assert torch.allclose(p, ps, atol=1e-5), \
            f"symmetric/{pool} pooled features changed under H1<->L1 swap (max {abs(p - ps).max():.2e})"


def test_concat_merge_changes_under_h1l1_swap():
    torch.manual_seed(0)
    X = torch.randn(BATCH, 3, 4096)
    X_swap = X[:, [1, 0, 2], :]
    model = DIYModel(
        n_channels=4, n_sky_features=N_SKY, sky_readout="none", h1l1_merge="concat",
    ).eval()
    p, ps = _pooled(model, X), _pooled(model, X_swap)
    assert not torch.allclose(p, ps, atol=1e-6), \
        "concat pooled features were unexpectedly invariant to H1<->L1 swap"


# ------------------------------------------------------------------ parameter matching

@pytest.mark.parametrize("mode", ["power", "bispectrum", "scramble"])
def test_match_params_equalizes_conditioning_count(mode):
    ref = DIYModel(n_channels=16, n_sky_features=N_SKY, sky_readout="mlp121").conditioning_param_count()
    matched = DIYModel(
        n_channels=16, n_sky_features=N_SKY, sky_readout=mode, match_params=True,
    ).conditioning_param_count()
    # integer hidden width can't hit the target exactly; require within 0.5% and one
    # hidden-unit's worth of the feat-dim projection.
    assert abs(matched - ref) / ref < 0.005, \
        f"{mode} matched conditioning params {matched} vs ref {ref} differ by >0.5%"


def test_match_params_changes_hidden_for_power():
    unmatched = DIYModel(n_channels=16, n_sky_features=N_SKY, sky_readout="power").conditioning_param_count()
    matched = DIYModel(n_channels=16, n_sky_features=N_SKY, sky_readout="power", match_params=True).conditioning_param_count()
    assert matched > unmatched, "match_params should widen the power readout's hidden layer"
