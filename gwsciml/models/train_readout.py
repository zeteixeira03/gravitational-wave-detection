"""Train the likelihood-map readout on the train subset, evaluate once on val.

Usage: python -m gwsciml.models.train_readout --tag <P1 tag>
Early stopping uses a 20 % holdout of the *train* subset; val is scored once.
Appends one row to results/runs.jsonl.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from gwsciml.detect.run_p1 import auc_with_ci
from gwsciml.models.readout import LikelihoodMapReadout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--results", type=Path, default=Path("data/results"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--prior-l2", type=float, default=1e-3, help="L2 pull of a, c towards the EM prior")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    prior = np.load(args.results / f"p2_prior_{args.tag}.npz")
    tr = pd.read_csv(args.results / f"p1_train_{args.tag}.csv")
    va = pd.read_csv(args.results / f"p1_val_{args.tag}.csv")
    Ltr = torch.as_tensor(np.load(args.results / f"p1_train_{args.tag}_lmap.npy"))
    Lva = torch.as_tensor(np.load(args.results / f"p1_val_{args.tag}_lmap.npy"))
    ytr = torch.as_tensor(tr.target.values, dtype=torch.float32)

    # train / holdout split inside the train subset (the EM prior saw all train positives; the
    # holdout therefore measures overfitting of the readout, not of the prior)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr))
    hold, fit = perm[: len(tr) // 5], perm[len(tr) // 5:]
    model = LikelihoodMapReadout(prior["log_pi"], prior["log_tau"])
    a0, c0 = model.a.detach().clone(), model.c.detach().clone()
    # affine init of the logit from the EM score (logistic fit of w0, w1 only)
    def chunked(fn, L, n=1000):
        with torch.no_grad():
            return torch.cat([fn(L[i:i + n]) for i in range(0, len(L), n)])

    with torch.no_grad():
        s = chunked(model.score, Ltr[fit])
        model.w[1] = 1.0 / s.std()
        model.w[0] = -s.mean() / s.std()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = (-1.0, None, -1)
    hist = []
    t0 = time.time()
    for ep in range(args.epochs):
        losses = []
        for bi in np.array_split(rng.permutation(fit), max(1, len(fit) // args.batch)):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(Ltr[bi]), ytr[bi])
            loss = loss + args.prior_l2 * (((model.a - a0) ** 2).sum() + ((model.c - c0) ** 2).sum())
            loss.backward()
            opt.step()
            losses.append(float(loss))
        loss = np.mean(losses)
        h_auc = roc_auc_score(tr.target.values[hold], chunked(model, Ltr[hold]).numpy())
        hist.append((ep, float(loss), h_auc))
        if h_auc > best[0]:
            best = (h_auc, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        if ep % 2 == 0:
            print(f"epoch {ep}: loss {float(loss):.4f}  holdout AUC {h_auc:.4f}", flush=True)
    model.load_state_dict(best[1])
    s_va = chunked(model, Lva).numpy()
    h0 = roc_auc_score(tr.target.values[hold],
                       chunked(LikelihoodMapReadout(prior["log_pi"], prior["log_tau"]).score, Ltr[hold]).numpy())
    auc, sd = auc_with_ci(va.target.values, s_va)
    print(f"holdout AUC: EM init {h0:.4f} -> best {best[0]:.4f} (epoch {best[2]})")
    print(f"val AUC {auc:.4f} +- {sd:.4f}")
    knots = model.g.knots.numpy()
    slopes = torch.clamp(1.0 + torch.cumsum(model.g.dslope, 0), min=0.05).detach().numpy()
    print("g slopes after knots", dict(zip(knots.tolist(), np.round(slopes, 3).tolist())))
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("results/runs.jsonl", "a") as fh:
        fh.write(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "phase": "P3-proto", "git": git, "split": "val",
            "n": int(len(va)), "tag": args.tag, "model": "LikelihoodMapReadout (monotone g, a_j, c_b), EM init",
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "holdout_auc_em_init": h0, "holdout_auc_best": best[0], "best_epoch": best[2],
            "auc": auc, "auc_bootstrap_sd": sd, "g_slopes": dict(zip(knots.tolist(), slopes.tolist())),
            "curve": [{"epoch": e, "loss": l, "holdout_auc": h} for e, l, h in hist],
            "seconds": time.time() - t0,
        }) + "\n")


if __name__ == "__main__":
    main()
