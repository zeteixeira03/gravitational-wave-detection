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
- Open decision D1: analytic waveform templates as *fixed physics filters* (not as
  training data). Section 2 depends on it.

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
| P2 | Unsupervised parameter estimation: per-sample maximum-likelihood theta-hat (template, time, phase, amplitude ratios, sky ring). Empirical prior p(theta) from the train positives. | theta-hat distribution of positives vs negatives. |
| P3 | Physics-structured network: matched-filter front end -> SNR "image" over (template manifold x time) per detector -> equivariant network (time translation, Z2 H1<->L1, coherence at allowed delays) -> calibrated ratio head. Auxiliary self-supervised theta-hat head. | Beats the frozen baseline (0.874) by more than seed SD. |
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
