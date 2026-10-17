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
   pip install torch numpy scipy scikit-learn pandas matplotlib tqdm pytest kaggle lalsuite
   ```
   `lalsuite` (PyPI wheel) gives the LAL waveform models (IMRPhenomD etc.) and the
   reference geometry that the tests compare against. Run tests with
   `python -m pytest gwsciml/tests -q` from the repo root.
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
   The key is a new-style token (`KGAT...`). kaggle>=2 reads it only from
   `KAGGLE_API_TOKEN`, so run `export KAGGLE_API_TOKEN="$KAGGLE_KEY"` first.
4. Disk: about 22 GB is free; the raw dataset is about 70 GB. Full-scale work runs in
   Kaggle kernels. **Single-file competition downloads are rate-limited** (HTTP 429
   after a few hundred files, still blocked 30+ min later). Use the pack kernel
   instead: `gwsciml/kaggle/pack_subset` reads a class-balanced subset inside Kaggle
   (5,000 + 5,000 per split, seed 0, raw float64) and writes `val_<c>.npy/.csv`,
   `train_<c>.npy/.csv`. Fetch with
   `kaggle kernels output zeteixeira/gwsciml-pack-subset -p data/g2net/packed`.
   Load with `gwsciml.data.io.load_packed`.
5. Read section 5 (status and results), then continue with section 6 (next tasks).

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
| 2026-10-17 | Val split fixed: rows 0-149,999 of `training_labels.csv` (sorted by id); train = rows 150,000-559,999. | Same split as the frozen baseline (`src/model_runs.py` shard split: first 3 shards of 50k). |
| 2026-10-17 | Whitening: Tukey(0.1) window, PSD = average periodogram of train negatives through the same window (leakage-inclusive). | Mergers sit at 1.5-1.85 s. Tukey(0.5) loses ~30 % SNR at 1.85 s; Tukey(0.1) loses < 5 % (measured, `test_window_choice_snr_efficiency`). Planck taper and high-pass prefilter gave no gain. |
| 2026-10-17 | Production waveform model IMRPhenomD (LAL); own TaylorF2 kept as the analytic reference (matches LAL to 1e-5 at every PN order). | Heavy BBH merge in band; TaylorF2 stops at the ISCO. |
| 2026-10-17 | Template duration capped at 1.5 s (start frequency from the Newtonian chirp time). | A template longer than the 2 s segment wraps in the circular filter. |
| 2026-10-17 | Orchestrator/writer split not used this session; code written and reviewed in one context. | Tool policy: subagents only on explicit user request. |

## 5. Status (end of session 2026-10-17)

P0 done, P1 done, P2 (a) done; P2 (b-d) and P3 open. All numbers below come from
`results/runs.jsonl` and the reports; nothing is hand-transcribed from elsewhere.

Evaluation set for this session: the packed **val subset**, 10,000 samples
(5,000 + 5,000, seed 0) from the fixed val split. AUC bootstrap sd ~0.004. The
frozen baseline 0.874 was measured on the full 150k val split, so differences
below ~0.008 are not resolved.

### P0 (reports/P0_audit.md)
- G2Net noise is **stationary Gaussian with one fixed PSD per detector**: per-bin
  periodogram cv^2 = 1 within sampling error in all 1,961 bins (20-1000 Hz), no
  glitches (0 of 15,000 detector-samples above 6 sigma), Gaussian kurtosis and
  peaks, uncorrelated detectors, train and val share the noise model.
- Lines (stationary too): 15 Hz (all), 60 Hz and 306 Hz (H1, L1), 438 Hz (V1).
- V1 is ~1.5x less sensitive in ASD than H1/L1 above 50 Hz.
- Consequence: the Gaussian likelihood of section 1 is *exact* for the noise; the
  only unknowns are the signal population and waveform family.

### P1 (classical detector, non-spinning IMRPhenomD bank, 944 templates, MM 0.97)
| statistic | val AUC |
|---|---|
| max over (template, time) of coincident incoherent SNR^2 | 0.8167 |
| log-sum-exp of the same (flat prior over bank and time) | 0.8224 |
| coherent SNR^2 (sky search on top-16 templates) | 0.8215 |
| chi-squared reweighted network SNR^2 | 0.8004 |

Coherence and chi-squared do not help here: the noise is Gaussian (nothing for
chi-squared to reject), and H1/L1 are nearly co-aligned with V1 weak, so the
coherent projection removes little noise.

### P2 (a): empirical-Bayes prior (reports/P2_report.md)
MAP-EM on 5,000 train positives fits weights over templates (pi_j) and H1 merger-time
bins of 50 ms (tau_b); val is scored with log sum pi_j tau_b exp(L_jb).

| prior | val AUC |
|---|---|
| uniform | 0.8228 |
| template only | 0.8364 |
| time only | 0.8532 |
| template x time | **0.8727 +- 0.0036** |
| joint (template, time), see results/runs.jsonl | run in progress at commit time |

- **A fully classical, interpretable detector (fixed GR filters + ~1k prior weights
  fitted without labels) matches the frozen CNN (0.874).**
- Time prior: 97.6 % of the weight at H1 merger times 1.45-1.80 s (the injection
  window, smeared by inter-detector delays). Template prior: chirp mass ~10-35 Msun,
  weighted median 17.3 vs 10.7 for the bank.
- Missing-physics signal: Allen chi2_r of positives grows with SNR (1.0 at rho 8-12,
  1.5 at 16-24, 4.2 at rho > 40), null energy too. Implied mismatch ~2-3 %, the same
  size as the bank spacing, so discreteness and missing physics (spin, precession)
  are not yet separated.

## 6. Next tasks (in order)

1. **Bigger prior fit.** The prior uses 5k train positives from a 10k subset. Run
   P1 over a large train slice in a Kaggle CPU kernel (0.2 s/sample/2 threads; GPU
   front end `gwsciml/models/frontend.py` is far faster) and refit pi, tau. Store
   likelihood maps (T x 36 float32 per sample) as a private Kaggle dataset; dataset
   downloads work through `api.kaggle.com`.
2. **Full-val evaluation** of the P2 statistic on all 150k val samples (same kernel),
   so the comparison with 0.874 is on the identical set.
3. **Separate discreteness from missing physics** (P2 d): refine templates around
   loud positives (local MM 0.995 sub-bank) and repeat the chi2-vs-rho table. If
   chi2 stays high, add aligned spins (bank ~3,000 templates, measured) and test
   IMRPhenomPv2/XPHM on the loudest positives.
4. **P3 network on the likelihood map.** Input: L[j, b] (and per-detector / coherent
   channels) from the fixed front end. The EM prior is the linear-in-exp special case
   of the readout, so initialise the network to reproduce log sum pi tau exp(L) and
   let it learn (i) the amplitude nonlinearity (exp(rho^2/2) assumes a flat SNR
   prior), (ii) smooth priors over the template manifold (graph convolution on the
   Fisher-metric kNN graph), (iii) finer time structure. Target: beat 0.874 by more
   than the seed sd. Compute: full-bank front end ~1 s per 64-sample batch on a P100
   (1.8 h per 410k epoch); precomputed maps avoid that cost.
5. Kernel outputs cannot be fetched with `kaggle kernels output` from the cloud
   session (`www.kaggleusercontent.com` is not on the network allowlist). This
   session used the SDK's `DownloadKernelOutputZip` on the allowed `api.kaggle.com`
   host with the kernel session id. Adding the host to the allowlist is cleaner.
