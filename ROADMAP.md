# SciML Roadmap: G2Net detection from first principles

Status file for the physics-structured rebuild. Every session and every agent reads
this file first. It holds the decisions, not the reasoning chat. Update it at the end
of each phase. The old pipeline (`src/`, `kaggle/`) stays frozen as the baseline.

## 0. Scope (decided by the user, 2026-10-17)

- Data: **G2Net only**. No synthetic training set, no simulator study.
- Goal: the best validation AUC on G2Net, built from physics, geometry and SciML.
- Architecture: free. The old CNN is a baseline, not a constraint.
- Old docs (`CLAUDE.md`, `ANALYSIS_PLAN.md`, `PHASE2_HANDOFF.md`) are context only.
- Compute: development and tests on CPU in the cloud session; training on Kaggle
  free tier (2 concurrent GPU sessions, per-session and weekly time limits).
- D1 (resolved, permitted): analytic GR waveform templates are allowed as *fixed
  physics filters*. They are never training samples. Every model trains and is
  evaluated only on G2Net data.
- Source parameter estimation is in scope as *unsupervised* maximum-likelihood
  estimation (G2Net has no parameter labels). See section 2, P2.

## Start here (new session)

1. Read this file. Then read `THE_SCIENCE.md` and `SWEEP_FINDINGS.md` for context only.
2. Environment setup (the container is fresh each session):
   ```bash
   pip install torch numpy scipy scikit-learn pandas matplotlib tqdm pytest kaggle
   ```
   Install torch from PyPI. `download.pytorch.org` is blocked by the network policy.
   The PyPI wheel (CUDA build) runs on CPU.
3. Kaggle access check:
   ```bash
   test -n "$KAGGLE_USERNAME" && test -n "$KAGGLE_KEY" && echo "key present"
   kaggle competitions files g2net-gravitational-wave-detection | head
   ```
   If this fails: the environment must allow `www.kaggle.com` and `api.kaggle.com`
   (Network access, Allowed domains) and set `KAGGLE_USERNAME` / `KAGGLE_KEY`.
   Never ask the user to paste the key in chat.
4. Disk: about 22 GB is free; the raw dataset is about 70 GB. Download a subset only
   (single files with `kaggle competitions download -f <path>`, or one preprocessed
   shard from the user's Kaggle dataset). Full-scale work runs in Kaggle kernels.
5. Begin P0 (section 2). The first concrete task list is in section 5.

## 1. Foundation (why the design is what it is)

1. Data model per detector k: `d_k(t) = n_k(t) + F+_k h+(t - tau_k) + Fx_k hx(t - tau_k)`.
2. Detection is a hypothesis test. Neyman-Pearson: the optimal statistic is the
   likelihood ratio `Lambda(d) = int p(d|theta) p(theta) dtheta / p(d|H0)`.
3. Gaussian noise: `ln Lambda(d|theta) = <d|s> - <s|s>/2`, with the noise-weighted
   inner product `<a|b> = 4 Re int a~(f) b~*(f) / S(f) df`. `<d|s>` is the matched filter.
4. BCE on balanced classes converges to `Lambda / (1 + Lambda)`, a monotonic map of
   Lambda. So the ideal network *is* the marginalised likelihood ratio, and AUC has a
   ceiling set by the SNR distribution of the data, not by the architecture.
5. Consequence: build the known parts of Lambda into the model (whitening, matched
   filtering, network coherence) and learn only the unknown parts (the prior
   p(theta), the marginalisation, non-Gaussian noise).

## 2. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| P0 | Data access and audit: per-sample PSD variation, Gaussianity (kurtosis, glitches), class balance, the frozen old-model baseline numbers. | Audit report committed. |
| P1 | Classical baseline: whitened template bank (chirp mass, mass ratio), per-detector complex SNR time series, coherent network statistic with light-travel-time windows, chi-squared consistency. | AUC of the classical detector on the G2Net val split. |
| P2 | Unsupervised parameter estimation: per-sample maximum-likelihood theta-hat = argmax over (template j, time t0) of `rho_j(t0) = abs(<d \| h_j(. - t0)>) / sqrt(<h_j \| h_j>)` (abs removes the phase). Network geometry: peak-time differences within light travel time -> sky ring; amplitude/phase ratios narrow it. Uses: (a) empirical prior p(theta) from the train positives, used to weight the bank; (b) chi-squared band test for glitch rejection; (c) coarse-to-fine search; (d) self-supervised auxiliary target for P3. | theta-hat distribution of positives vs negatives; template-bank match on high-SNR positives (detects missing physics such as precession or higher modes). |
| P3 | Physics-structured network: matched-filter front end -> SNR "image" over (template manifold x time) per detector -> equivariant network (time translation, Z2 H1<->L1, coherence at allowed delays) -> calibrated ratio head. Convolution over the template axis uses the Fisher metric `g_ij = <d_i h \| d_j h>` as the manifold distance (GDL on the parameter space). Auxiliary self-supervised theta-hat head. | Beats the frozen baseline (0.874) by more than seed SD. |
| P4 | Controlled ablation of each component, 3+ seeds, fixed recipe. | Results log + findings. |

## 3. Working rules

- New code lives in `gwsciml/`. Do not edit `src/`.
- Every module has unit tests against analytic identities before it touches data.
- One results log (JSONL), one row per run, never hand-transcribed, no row dropped.
- Orchestrator (Opus) writes specs and reviews. Writers (Sonnet/Haiku) implement from
  self-contained specs. Review before every commit.

## 4. Decision log

| Date | Decision | Reason |
|---|---|---|
| 2026-10-17 | G2Net only; simulator out of scope. | User. |
| 2026-10-17 | Old pipeline frozen as baseline; new package `gwsciml/`. | Clean comparison, no regressions. |
| 2026-10-17 | D1: fixed GR templates permitted as filters, not as data. | User. |
| 2026-10-17 | Unsupervised PE (maximum likelihood over the bank) in scope. | User asked; no labels exist. |
| 2026-10-17 | Old `CLAUDE.md` conventions not binding. | User: outdated. |

## 5. P0 task list (next session)

1. Confirm Kaggle access (section "Start here", step 3).
2. Download a small, class-balanced subset of raw G2Net train files (a few thousand
   `.npy` files plus `training_labels.csv`) to `data/` (git-ignored).
3. `gwsciml/data/`: loader for raw samples, per-sample and average PSD (Welch),
   whitening. Unit tests: whitened Gaussian noise has unit variance and a flat PSD.
4. Audit notebook or script, results to `reports/P0_audit.md`:
   per-sample PSD spread per detector; kurtosis and outlier counts (glitches);
   class balance; signal visibility after whitening for a few positives.
5. Record the frozen baseline: old best AUC 0.874 (no sky), sweep band 0.860-0.864
   at the 100k recipe (`SWEEP_FINDINGS.md`). Use the same val split definition for
   all later comparisons; write it down in this file once fixed.
6. Then P1: template bank (`gwsciml/templates/`), with tests against analytic
   identities (normalisation `<h|h>` = 1 after scaling; the matched-filter peak
   recovers a template injected into white noise at the correct time and SNR).
