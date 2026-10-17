"""
Redundancy probe for SWEEP_FINDINGS F2, and the F9 shard check.

F2 records that the sky pathway adds no AUC in any readout form, and offers an
untested interpretation: the CNN already extracts the inter-detector structure
the SH coefficients encode, so the sky features are redundant. This module
tests that on a trained config-1 model (no sky pathway), on the validation
split the model never trained on. The split is halved: readouts are fit on one
half and scored on the other.

Symbols: z is the pooled CNN feature vector (the classifier input), a the SH
coefficient vector, p its angular power spectrum, y the label.

Two measurements (pre-registered in ANALYSIS_PLAN section 8):

- Redundancy for the label (decisive). Logistic readouts on z, and on z with a
  and p appended. dAUC = AUC(z+a+p) - AUC(z) with a paired bootstrap CI. If a
  carries label information z lacks, dAUC > 0.
- Encoding (mechanism). Ridge R^2 of each a_lm from z, summarized per degree l,
  against two baselines: z from the same network at its random init, and the
  label alone.
"""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from sky_feasibility import power_spectrum

RIDGE_ALPHAS = np.logspace(-2, 4, 7)
LOGISTIC_CS = [0.01, 0.1, 1.0]


def environment_report() -> dict:
    """Software and hardware stack, so a stack change between runs is visible in the log."""
    report = {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        report.update({
            'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'device': torch.cuda.get_device_name(0),
            'capability': list(torch.cuda.get_device_capability(0)),
            'arch_list': torch.cuda.get_arch_list(),
        })
    return report


def inspect_shard(data_dir) -> dict:
    """
    F9: read one shard's stored fields directly instead of inferring them from
    file sizes. mmap keeps this from pulling the whole ~2.5 GB file into memory.
    """
    shard = sorted(Path(data_dir).glob('shard_*.pt'))[0]
    data = torch.load(str(shard), weights_only=True, mmap=True)
    return {
        'shard': shard.name,
        'fields': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in data.items()},
    }


@torch.no_grad()
def extract(model, loader, device):
    """
    Pooled features z (before any FiLM), SH coefficients a, labels y, and the
    model's own probabilities, over one pass of the loader in eval mode.
    """
    model.eval()
    zs, as_, ys, ps = [], [], [], []
    for X, y, sky in loader:
        X, sky = X.to(device), sky.to(device)
        pooled, _ = model._extract_pooled(X)
        logits = model.classify_from_pooled(pooled, sky)
        zs.append(pooled.float().cpu())
        as_.append(sky.float().cpu())
        ys.append(y.cpu())
        ps.append(torch.sigmoid(logits.float()).squeeze(-1).cpu())
    return (torch.cat(zs).numpy(), torch.cat(as_).numpy(),
            torch.cat(ys).numpy().astype(int), torch.cat(ps).numpy())


def split_halves(n: int, seed: int):
    """Fixed random halves: readouts are fit on the first, scored on the second."""
    order = np.random.default_rng(seed).permutation(n)
    return order[: n // 2], order[n // 2:]


def ridge_predict(Z_fit, A_fit, Z_eval, A_eval) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear prediction of each SH coefficient from features Z. Both sides are
    standardized on the fit half, so every coefficient counts equally and the
    ridge penalty sees features on one scale. Returns (standardized eval truth,
    eval prediction); R^2 is taken from these on any subset of the eval rows.
    """
    zs, as_ = StandardScaler().fit(Z_fit), StandardScaler().fit(A_fit)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
    ridge.fit(zs.transform(Z_fit), as_.transform(A_fit))
    return as_.transform(A_eval), ridge.predict(zs.transform(Z_eval))


def r2_per_coeff(truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return r2_score(truth, pred, multioutput='raw_values')


def r2_by_degree(r2: np.ndarray, l_max: int) -> list[float]:
    """Mean R^2 over the 2l+1 coefficients of each degree l (index l^2 .. (l+1)^2 - 1)."""
    return [float(r2[l * l:(l + 1) * (l + 1)].mean()) for l in range(l_max + 1)]


def logistic_proba(X_fit, y_fit, X_eval) -> np.ndarray:
    """
    Standardized L2 logistic readout, C chosen by 3-fold CV on the fit half.
    The same C grid is used for every feature set so no set gets more tuning.
    """
    sc = StandardScaler().fit(X_fit)
    clf = LogisticRegressionCV(Cs=LOGISTIC_CS, cv=3, scoring='roc_auc', max_iter=2000, n_jobs=-1)
    clf.fit(sc.transform(X_fit), y_fit)
    return clf.predict_proba(sc.transform(X_eval))[:, 1]


def paired_bootstrap_delta(y, p_base, p_aug, n_boot=1000, seed=0) -> dict:
    """
    AUC(p_aug) - AUC(p_base) on the same samples, with a percentile 95% CI from
    resampling samples (both scores move together, so the CI is on the paired
    difference, not on two independent AUCs).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[b] = roc_auc_score(y[idx], p_aug[idx]) - roc_auc_score(y[idx], p_base[idx])
    return {
        'delta': float(roc_auc_score(y, p_aug) - roc_auc_score(y, p_base)),
        'ci95': [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
        'n_boot': n_boot,
    }


def probe_features(Z, A, y, model_proba, l_max, Z_rand=None, split_seed=0) -> dict:
    """Run both measurements on extracted arrays. Split into fit/eval halves first."""
    fit, ev = split_halves(len(y), split_seed)
    P = power_spectrum(A.astype(np.float64), l_max)
    t0 = time.time()

    # encoding: R^2 of a from z, from random-init z, and from y alone
    truth, pred = ridge_predict(Z[fit], A[fit], Z[ev], A[ev])
    r2_z = r2_per_coeff(truth, pred)
    enc = {'trained': {'per_degree': r2_by_degree(r2_z, l_max), 'mean': float(r2_z.mean())}}
    # the same fit, scored separately on the noise-only and signal eval rows:
    # geometry in noise samples comes from noise correlations, which the label
    # gives the CNN no reason to encode.
    for cls in (0, 1):
        mask = y[ev] == cls
        enc['trained'][f'per_degree_y{cls}'] = r2_by_degree(r2_per_coeff(truth[mask], pred[mask]), l_max)
    yf = y[:, None].astype(float)
    r2_y = r2_per_coeff(*ridge_predict(yf[fit], A[fit], yf[ev], A[ev]))
    enc['label_only'] = {'per_degree': r2_by_degree(r2_y, l_max), 'mean': float(r2_y.mean())}
    enc['random_init'] = None
    if Z_rand is not None:
        r2_r = r2_per_coeff(*ridge_predict(Z_rand[fit], A[fit], Z_rand[ev], A[ev]))
        enc['random_init'] = {'per_degree': r2_by_degree(r2_r, l_max), 'mean': float(r2_r.mean())}

    # redundancy for the label: does appending a (and p) to z raise the AUC?
    feature_sets = {
        'z': Z,
        'z_a': np.hstack([Z, A]),
        'z_a_p': np.hstack([Z, A, P]),
        'a_p': np.hstack([A, P]),
    }
    proba = {k: logistic_proba(X[fit], y[fit], X[ev]) for k, X in feature_sets.items()}
    auc = {k: float(roc_auc_score(y[ev], p)) for k, p in proba.items()}
    auc['model'] = float(roc_auc_score(y[ev], model_proba[ev]))

    return {
        'n_fit': int(len(fit)),
        'n_eval': int(len(ev)),
        'split_seed': split_seed,
        'l_max': l_max,
        'encoding_r2': enc,
        'auc': auc,
        'delta_z_a': paired_bootstrap_delta(y[ev], proba['z'], proba['z_a']),
        'delta_z_a_p': paired_bootstrap_delta(y[ev], proba['z'], proba['z_a_p']),
        'probe_seconds': round(time.time() - t0, 1),
    }


def run_probe(output_dir=None, config_id=1, seed=0):
    """
    Kaggle entry point. Logs the stack and the F9 shard check first (so a later
    failure still leaves them), trains one sweep run with the standard logger,
    then probes its final model against the same network at random init.
    """
    from data.g2net import get_output_dir
    from model_runs import (
        build_model, check_gpu, find_tensor_data_dir, load_n_samples, read_git_hash,
        run_one, set_seed, sweep_hyperparameters,
    )

    has_gpu = check_gpu()
    output_dir = Path(output_dir) if output_dir else get_output_dir()
    data_dir = find_tensor_data_dir(output_dir)
    out_path = output_dir / "probe_results.json"
    report = {
        'config_id': config_id,
        'seed': seed,
        'git_hash': read_git_hash(),
        'environment': environment_report(),
        'shard_check': inspect_shard(data_dir),
    }
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))

    row, results = run_one(
        config_id, seed, data_dir, load_n_samples(data_dir),
        output_dir / "sweep_results.jsonl", output_dir / "models" / "saved",
        report['git_hash'], has_gpu,
    )
    report['run'] = {k: row[k] for k in ('config_id', 'auc', 'best_auc', 'best_epoch', 'epochs_run')}

    device, loader = results['device'], results['val_loader']
    Z, A, y, model_proba = extract(results['model'], loader, device)

    # the same network at its starting point: build_model after set_seed(seed)
    # draws the init the trained run started from
    hp, _ = sweep_hyperparameters(config_id, seed, has_gpu)
    set_seed(seed)
    rand_model = build_model(hp, A.shape[1], device)
    Z_rand, _, _, _ = extract(rand_model, loader, device)

    l_max = int(round(np.sqrt(A.shape[1]))) - 1
    report['probe'] = probe_features(Z, A, y, model_proba, l_max, Z_rand=Z_rand, split_seed=seed)
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(report['probe'], indent=1))
    print(f"\nProbe complete. Results: {out_path}")
    return report
