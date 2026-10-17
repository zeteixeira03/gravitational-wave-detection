"""Build and cache a stochastic template bank.

Usage: python -m gwsciml.templates.build_bank --psd data/g2net/packed/psd_avg_train_neg_tukey0.1.npy
Writes data/bank/<name>.npz with the accepted (m1, m2, chi1z, chi2z) rows and the config.
The bank is placed with the geometric mean of the H1 and L1 average PSDs.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gwsciml.templates.bank import make_templates, sample_candidates, stochastic_bank


def candidates(n, rng, m_min, m_max, chi_max):
    """Half uniform in component masses, half log-uniform: the bank is densest at low mass."""
    a = sample_candidates(n // 2, rng, m_min, m_max, chi_max)
    lm = np.exp(rng.uniform(np.log(m_min), np.log(m_max), (n - n // 2, 2)))
    b = np.column_stack([lm.max(1), lm.min(1), a[: n - n // 2, 2:] if chi_max > 0 else np.zeros((n - n // 2, 2))])
    c = np.concatenate([a, b])
    return c[rng.permutation(len(c))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psd", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/bank"))
    ap.add_argument("--n-cand", type=int, default=20000)
    ap.add_argument("--m-min", type=float, default=5.0)
    ap.add_argument("--m-max", type=float, default=100.0)
    ap.add_argument("--chi-max", type=float, default=0.0)
    ap.add_argument("--min-match", type=float, default=0.97)
    ap.add_argument("--max-duration", type=float, default=1.5)
    ap.add_argument("--approximant", default="IMRPhenomD")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    psd = np.load(args.psd)
    ref = np.sqrt(psd[0] * psd[1])
    rng = np.random.default_rng(args.seed)
    cand = candidates(args.n_cand, rng, args.m_min, args.m_max, args.chi_max)
    t0 = time.time()
    temps = make_templates(cand, ref, max_duration=args.max_duration, approximant=args.approximant)
    acc = stochastic_bank(cand, temps, args.min_match)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg["n_templates"] = int(len(acc))
    cfg["build_seconds"] = round(time.time() - t0, 1)
    name = f"{args.approximant}_m{args.m_min:g}-{args.m_max:g}_chi{args.chi_max:g}_mm{args.min_match:g}"
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / f"{name}.npz", params=cand[acc], config=json.dumps(cfg))
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
