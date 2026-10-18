"""
Representation probe for the sky-readout sweep (FOLLOWUP_PLAN.md, P1).

The sweep found that the sky pathway adds nothing (SWEEP_FINDINGS F2). Two
explanations predict that outcome and disagree on the cause:

- redundancy: the CNN's pooled features already carry the label information
  the SH coefficients carry, so a second pathway has nothing to add;
- fusion failure: the information is complementary, but the FiLM pathway
  never learns to use it.

They make different predictions about a frozen, trained CNN. Under
redundancy, a classifier on (CNN features + SH) does no better than one on
CNN features alone. Under fusion failure it does better. This module measures
that difference, and two descriptive quantities around it:

- how much of the SH vector is linearly decodable from the CNN features
  (ridge R^2, per degree l), against a random-init network of the same
  architecture as the floor;
- how the H1<->L1 channel-swap augmentation shapes both representations
  (the fraction of variance that is odd under the swap).

Everything runs on the validation split, which the CNN never trained on. The
split is halved again into probe-train and probe-test.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from sky_feasibility import sh_block_index

SWAP = [1, 0, 2]  # H1 <-> L1, Virgo fixed

# pre-registered in FOLLOWUP_PLAN.md section P1. Do not change after a probe has run.
DELTA_MARGIN = 0.001
# the feasibility gate from sky_feasibility.py: below this the sky features carry
# too little label information for "adds nothing" to mean anything
SKY_GATE_AUC = 0.55


# ============================================================================================
#                                        features
# ============================================================================================

def calibrate_bn(model: nn.Module, signals: torch.Tensor, device, batch_size: int = 256) -> None:
    """
    Recompute BatchNorm running statistics on the given signals.

    Used for the random-init baseline only: its running stats are still at the
    (0, 1) defaults, which makes eval-mode activations meaningless. Only the BN
    layers switch to train mode, so dropout and drop-path stay off and the pass
    is deterministic.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in bn_layers]
    model.eval()
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            model._extract_pooled(signals[i:i + batch_size].to(device))
    for m, mom in zip(bn_layers, momenta):
        m.momentum = mom
    model.eval()


def extract_pooled(model: nn.Module, signals: torch.Tensor, device, batch_size: int = 256,
                   swap: bool = False) -> np.ndarray:
    """
    Pooled CNN features (the input to the sky readout / classifier head).

    Parameters
    ----------
    model
        A DIYModel. Its pooled features do not depend on the sky input.
    signals
        Shape (N, 3, 4096).
    swap
        Exchange the H1 and L1 channels before the forward pass.

    Returns
    -------
    features
        Shape (N, feat_dim), float32.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            x = signals[i:i + batch_size]
            if swap:
                x = x[:, SWAP, :]
            pooled, _ = model._extract_pooled(x.to(device))
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out)


def model_probabilities(model: nn.Module, signals: torch.Tensor, sky: torch.Tensor, device,
                        batch_size: int = 256) -> np.ndarray:
    """Sigmoid outputs of the full model, for the reference AUC on probe-test."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            logits = model(signals[i:i + batch_size].to(device), sky_features=sky[i:i + batch_size].to(device))
            out.append(torch.sigmoid(logits.float()).cpu().numpy().ravel())
    return np.concatenate(out)


def recompute_sh(signals: torch.Tensor, sky_geo, swap: bool = False) -> np.ndarray:
    """SH coefficients recomputed from the signals with the current SkyGeometry."""
    out = np.empty((len(signals), sky_geo.n_coeffs), dtype=np.float32)
    for i in range(len(signals)):
        x = signals[i].numpy()
        if swap:
            x = x[SWAP]
        out[i] = sky_geo.extract(x)
    return out


# ============================================================================================
#                                      verification
# ============================================================================================

def verify_stored_sh(stored: np.ndarray, recomputed: np.ndarray) -> dict:
    """
    Compare the SH coefficients stored in the shards against a recomputation.

    This settles SWEEP_FINDINGS F9 (l_max was inferred from file sizes) and
    checks that the stored features are what the current SkyGeometry produces.
    """
    n_coeffs = stored.shape[1]
    l_max = int(round(np.sqrt(n_coeffs))) - 1
    diff = np.abs(stored - recomputed)
    scale = np.abs(stored).max()
    return {
        'stored_shape': list(stored.shape),
        'l_max_inferred': l_max,
        'l_max_consistent': (l_max + 1) ** 2 == n_coeffs,
        'max_abs_diff': float(diff.max()),
        'max_rel_diff': float(diff.max() / scale) if scale > 0 else float('nan'),
        'n_checked': int(len(stored)),
    }


# ============================================================================================
#                                    swap decomposition
# ============================================================================================

def odd_variance_fraction(a: np.ndarray, a_swap: np.ndarray) -> float:
    """
    Fraction of total variance in ``a`` that is odd under the H1<->L1 swap.

    Any representation splits as a = even + odd with even = (a + a_swap)/2 and
    odd = (a - a_swap)/2. A swap-invariant representation has odd == 0. Returns
    sum_j Var(odd_j) / sum_j Var(a_j).
    """
    odd = 0.5 * (a - a_swap)
    total = a.var(axis=0).sum()
    return float(odd.var(axis=0).sum() / total) if total > 0 else float('nan')


# ============================================================================================
#                                         ridge
# ============================================================================================

def _standardize(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(axis=0)
    sd = train.std(axis=0)
    sd[sd == 0] = 1.0
    return [(x - mu) / sd for x in (train,) + others]


def _ridge_predict_all(Z_tr, Y_tr, Z_ev, alphas):
    """Yield ridge predictions on Z_ev for each alpha from one SVD of Z_tr (centred inputs)."""
    U, s, Vt = np.linalg.svd(Z_tr, full_matrices=False)
    A = Z_ev @ Vt.T
    B = U.T @ Y_tr
    for a in alphas:
        yield A @ ((s / (s * s + a))[:, None] * B)


def ridge_r2(Z_tr: np.ndarray, Y_tr: np.ndarray, Z_te: np.ndarray, Y_te: np.ndarray,
             n_folds: int = 5, seed: int = 0) -> dict:
    """
    Held-out R^2 of a ridge map Z -> Y, with the penalty chosen per target by
    k-fold CV on train.

    Inputs and targets are standardized with probe-train statistics, so every
    target coefficient carries equal weight. R^2 per target is
    1 - SSE_j / SST_j on probe-test. The penalty is per target because one
    shared penalty lets the many unpredictable coefficients drive CV toward
    heavy shrinkage, which biases R^2 down for the few predictable ones.

    Returns
    -------
    dict
        ``r2`` (pooled over targets: 1 - sum SSE / sum SST), ``r2_per_target``,
        ``sse``/``sst`` per target, ``alpha_median`` (chosen penalty divided by
        n_train) and ``alpha_edge_frac`` (fraction of targets whose penalty sits
        at an end of the grid).
    """
    Z_tr, Z_te = _standardize(Z_tr, Z_te)
    Y_tr, Y_te = _standardize(Y_tr, Y_te)
    n = len(Z_tr)
    rel = np.logspace(-6, 2, 17)
    alphas = rel * n

    folds = np.array_split(np.random.default_rng(seed).permutation(n), n_folds)
    cv_sse = np.zeros((len(alphas), Y_tr.shape[1]))
    for k in range(n_folds):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        zm, ym = Z_tr[trn].mean(0), Y_tr[trn].mean(0)
        preds = _ridge_predict_all(Z_tr[trn] - zm, Y_tr[trn] - ym, Z_tr[val] - zm, alphas)
        for i, p in enumerate(preds):
            cv_sse[i] += ((p + ym - Y_tr[val]) ** 2).sum(axis=0)
    best = cv_sse.argmin(axis=0)  # per target

    # stream the alphas: stacking every prediction is ~1.2 GB at 75k x 121 x 17
    zm, ym = Z_tr.mean(0), Y_tr.mean(0)
    pred = np.empty_like(Y_te)
    for i, p in enumerate(_ridge_predict_all(Z_tr - zm, Y_tr - ym, Z_te - zm, alphas)):
        cols = best == i
        pred[:, cols] = p[:, cols] + ym[cols]
    sse = ((pred - Y_te) ** 2).sum(axis=0)
    sst = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0)
    return {
        'r2': float(1.0 - sse.sum() / sst.sum()),
        'r2_per_target': (1.0 - sse / sst).tolist(),
        'sse': sse.tolist(),
        'sst': sst.tolist(),
        'alpha_median': float(np.median(rel[best])),
        'alpha_edge_frac': float(np.isin(best, (0, len(alphas) - 1)).mean()),
    }


def r2_lge1(fit: dict, l_max: int) -> float:
    """Pool a ridge fit over the l >= 1 (direction-dependent) a_lm targets."""
    block = sh_block_index(l_max)
    sse, sst = np.asarray(fit['sse']), np.asarray(fit['sst'])
    return float(1.0 - sse[block >= 1].sum() / sst[block >= 1].sum())


def r2_by_degree(fit: dict, l_max: int) -> list[float]:
    """Pool a ridge fit over a_lm targets into one R^2 per degree l."""
    block = sh_block_index(l_max)
    sse, sst = np.asarray(fit['sse']), np.asarray(fit['sst'])
    return [float(1.0 - sse[block == l].sum() / sst[block == l].sum()) for l in range(l_max + 1)]


# ============================================================================================
#                                      label probes
# ============================================================================================

def train_mlp_probe(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, device, seed: int = 0,
                    hidden: tuple[int, int] = (256, 64), dropout: float = 0.2, lr: float = 1e-3,
                    weight_decay: float = 1e-4, batch_size: int = 512, max_epochs: int = 40,
                    patience: int = 4, inner_val: float = 0.1) -> np.ndarray:
    """
    Train a fixed-capacity MLP classifier on frozen features; return P(signal) on X_te.

    Every feature set gets the same head, schedule and stopping rule, so AUC
    differences between feature sets come from the features. Early stopping
    uses an inner split of probe-train; probe-test is never seen in training.
    The caller's global RNG state is left untouched.
    """
    X_tr, X_te = _standardize(X_tr, X_te)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_tr))
    n_in = int(len(X_tr) * inner_val)
    iv, it = perm[:n_in], perm[n_in:]

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    Xt, yt = t(X_tr[it]), t(y_tr[it]).unsqueeze(1)
    Xv, yv = t(X_tr[iv]), t(y_tr[iv]).unsqueeze(1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(
            nn.Linear(X_tr.shape[1], hidden[0]), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        ).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator().manual_seed(seed)

        best_loss, best_state, bad = float('inf'), None, 0
        for _ in range(max_epochs):
            net.train()
            order = torch.randperm(len(Xt), generator=gen).to(device)
            for i in range(0, len(Xt), batch_size):
                b = order[i:i + batch_size]
                opt.zero_grad()
                loss_fn(net(Xt[b]), yt[b]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = loss_fn(net(Xv), yv).item()
            if v < best_loss:
                best_loss, bad = v, 0
                best_state = {k: p.detach().clone() for k, p in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(t(X_te))).cpu().numpy().ravel()


def paired_bootstrap_delta(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
                           n_boot: int = 1000, seed: int = 0) -> dict:
    """
    AUC(p_b) - AUC(p_a) on the same samples, with a paired-bootstrap 95% interval.

    Pairing matters: both classifiers score the same test samples, so most of
    the sampling noise is shared and cancels in the difference.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx])
    return {
        'delta': float(roc_auc_score(y, p_b) - roc_auc_score(y, p_a)),
        'ci_lo': float(np.percentile(deltas, 2.5)),
        'ci_hi': float(np.percentile(deltas, 97.5)),
        'n_boot': n_boot,
    }


def read_outcome(delta: dict, sky_auc: float, margin: float = DELTA_MARGIN,
                 gate: float = SKY_GATE_AUC) -> str:
    """
    Apply the pre-registered reading rule (FOLLOWUP_PLAN.md P1) to the primary delta.

    - ``sky_uninformative``: the SH-only probe is below the feasibility gate.
      Nothing to be redundant with; the probe cannot discriminate.
    - ``complementary``: the CI lies above 0. Frozen CNN features plus SH beat
      CNN features alone, so the information was there and the end-to-end
      sky pathway failed to use it. Rejects the redundancy interpretation.
    - ``redundant``: the CI lies below +margin. SH carries label information
      on its own but adds none on top of the CNN features.
    - ``inconclusive``: neither.
    """
    if sky_auc < gate:
        return 'sky_uninformative'
    if delta['ci_lo'] > 0:
        return 'complementary'
    if delta['ci_hi'] < margin:
        return 'redundant'
    return 'inconclusive'


# ============================================================================================
#                                      orchestration
# ============================================================================================

def run_probe(model: nn.Module, rand_model: nn.Module, signals: torch.Tensor, labels: torch.Tensor,
              stored_sh: torch.Tensor, sky_geo, device, l_max: int = 10, batch_size: int = 256,
              seed: int = 0, n_boot: int = 1000, verbose: bool = True) -> dict:
    """
    Run probe P1 on the validation split and return one JSON-serializable record.

    Parameters
    ----------
    model
        The trained DIYModel.
    rand_model
        The same architecture at its untrained init (same seed). BN statistics
        are recalibrated on probe-train before use.
    signals, labels, stored_sh
        The validation split, exactly as the training run evaluated it.
    sky_geo
        SkyGeometry used to recompute (and swap-recompute) the SH coefficients.
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    n = len(labels)
    perm = np.random.default_rng(seed).permutation(n)
    tr, te = np.sort(perm[: n // 2]), np.sort(perm[n // 2:])
    y = labels.numpy().astype(np.float32)
    S = stored_sh.numpy().astype(np.float32)
    block = sh_block_index(l_max)

    log(f"[probe] {n} val samples -> {len(tr)} probe-train / {len(te)} probe-test")

    log("[probe] pooled features: trained, trained+swap")
    Z = extract_pooled(model, signals, device, batch_size)
    Z_swap = extract_pooled(model, signals, device, batch_size, swap=True)

    # BN statistics converge after ~10k samples under a cumulative average; 20k of
    # probe-train is enough and keeps the indexed copy near 1 GB, not 3.7 GB.
    log("[probe] pooled features: random init (BN calibrated on probe-train)")
    calibrate_bn(rand_model, signals[torch.as_tensor(tr[:20000])], device, batch_size)
    Z_rand = extract_pooled(rand_model, signals, device, batch_size)
    Z_rand_swap = extract_pooled(rand_model, signals, device, batch_size, swap=True)

    log("[probe] recomputing SH (unswapped and swapped) on CPU")
    S_rec = recompute_sh(signals, sky_geo)
    S_rec_swap = recompute_sh(signals, sky_geo, swap=True)
    sh_check = verify_stored_sh(S, S_rec)
    log(f"[probe] SH check: {sh_check}")

    S_even = 0.5 * (S_rec + S_rec_swap)
    S_odd = 0.5 * (S_rec - S_rec_swap)
    P = np.log(np.maximum(
        np.stack([(S ** 2)[:, block == l].sum(1) for l in range(l_max + 1)], 1), 1e-12))

    swap = {
        'z_odd_var_frac': odd_variance_fraction(Z, Z_swap),
        'z_rand_odd_var_frac': odd_variance_fraction(Z_rand, Z_rand_swap),
        'sh_odd_var_frac': odd_variance_fraction(S_rec, S_rec_swap),
    }
    log(f"[probe] swap: {swap}")

    log("[probe] ridge R^2")
    ridge = {}
    for name, feats in (('trained', Z), ('random', Z_rand)):
        r = {}
        for tgt_name, tgt in (('alm', S), ('log_power', P), ('alm_even', S_even), ('alm_odd', S_odd)):
            fit = ridge_r2(feats[tr], tgt[tr], feats[te], tgt[te], seed=seed)
            entry = {'r2': fit['r2'], 'alpha_median': fit['alpha_median'],
                     'alpha_edge_frac': fit['alpha_edge_frac']}
            if tgt_name == 'log_power':
                entry['r2_per_degree'] = fit['r2_per_target']
            else:
                entry['r2_by_degree'] = r2_by_degree(fit, l_max)
                entry['r2_lge1'] = r2_lge1(fit, l_max)
            r[tgt_name] = entry
        for cls_name, cls in (('signal', 1.0), ('noise', 0.0)):
            tr_c, te_c = tr[y[tr] == cls], te[y[te] == cls]
            fit = ridge_r2(feats[tr_c], S[tr_c], feats[te_c], S[te_c], seed=seed)
            r[f'alm_{cls_name}_only'] = {'r2': fit['r2'], 'r2_by_degree': r2_by_degree(fit, l_max),
                                        'r2_lge1': r2_lge1(fit, l_max),
                                        'alpha_median': fit['alpha_median'],
                                        'alpha_edge_frac': fit['alpha_edge_frac']}
        ridge[name] = r
        log(f"[probe] ridge {name}: alm {r['alm']['r2']:.3f}  log_power {r['log_power']['r2']:.3f}  "
            f"even {r['alm_even']['r2']:.3f}  odd {r['alm_odd']['r2']:.3f}")

    log("[probe] label probes")
    l0, lge1 = block == 0, block >= 1
    feature_sets = {
        'cnn': Z,
        'sky': S,
        'cnn+sky': np.concatenate([Z, S], 1),
        'sky_l0': S[:, l0],
        'sky_lge1': S[:, lge1],
        'cnn+sky_lge1': np.concatenate([Z, S[:, lge1]], 1),
        'sky_even': S_even,
        'sky_odd': S_odd,
        'cnn_random': Z_rand,
    }
    probs = {}
    for name, X in feature_sets.items():
        probs[name] = train_mlp_probe(X[tr], y[tr], X[te], device, seed=seed)
        log(f"[probe]   {name:14s} AUC {roc_auc_score(y[te], probs[name]):.4f}")
    auc = {k: float(roc_auc_score(y[te], p)) for k, p in probs.items()}
    # score every sample and index afterwards: indexing signals first would copy 3.7 GB
    p_model = model_probabilities(model, signals, stored_sh, device, batch_size)
    auc['model_on_probe_test'] = float(roc_auc_score(y[te], p_model[te]))

    primary = paired_bootstrap_delta(y[te], probs['cnn'], probs['cnn+sky'], n_boot=n_boot, seed=seed)
    secondary = paired_bootstrap_delta(y[te], probs['cnn'], probs['cnn+sky_lge1'], n_boot=n_boot, seed=seed)
    outcome = read_outcome(primary, auc['sky'])
    log(f"[probe] primary delta (cnn+sky - cnn): {primary}  -> {outcome}")

    return {
        'n_probe_train': int(len(tr)),
        'n_probe_test': int(len(te)),
        'sh_check': sh_check,
        'swap': swap,
        'ridge': ridge,
        'label_auc': auc,
        'delta_cnn_sky': primary,
        'delta_cnn_sky_lge1': secondary,
        'delta_margin': DELTA_MARGIN,
        'sky_gate_auc': SKY_GATE_AUC,
        'outcome': outcome,
    }
