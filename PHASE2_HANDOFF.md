# NeurReps sky-readout sweep — session handoff (Phases 1–2 complete)

Working state for the NeurReps 2026 workshop paper experiment (deadline 23 Aug 2026,
fallback RPS 30 Aug). Read this + `ANALYSIS_PLAN.md` before continuing. The task is a
controlled sweep testing whether an SO(3)-invariant readout of the S² sky features
recovers performance the current MLP readout fails to deliver. Outcome B (clean negative
result + diagnosis) is the expected and acceptable paper.

Runs execute on **Kaggle free tier, P100, 2 concurrent sessions**. Local machine is
**CPU-only** (`torch 2.10.0+cpu`) — dev + unit tests only, no local training. Data lives
under `D:/Programming` (raw + preprocessed tensors) per `CLAUDE.md` paths.

## Phase status
- **Phase 1 (audit): done.** Findings below.
- **Phase 2 (implement + tests): done.** 13/13 tests pass. Code not yet committed
  (waiting on user go-ahead).
- **Phase 3 (run sweep): not started.** Needs the results logger + git-hash stamping
  (see Open items).
- **Phase 4 (report): not started.**

## Pre-registered decisions (locked in ANALYSIS_PLAN.md §0 "Locked values")
- **Primary metric:** validation AUC. **Seeds:** {0, 1, 2}, identical across all configs.
- **Primary decision comparator:** `power` (match_params) vs `mlp121` (match_params),
  i.e. Tier-1 config 3 vs config 4. `power` helps iff mean AUC exceeds mlp121-matched by
  more than pooled SD = sqrt((s_power² + s_mlp121²)/2). Matched-only isolates invariance
  from capacity; `power` vs unmatched `mlp121` (3 vs 2) is a supporting comparison, does
  not trigger the rule.

## What Phase 2 implemented (behind config flags; nothing tuned per-config)

**`src/sky_feasibility.py`** (SH algebra; no data/preprocessing/PSD/whitening touched):
- `sph_harm` compat shim: works on scipy <1.15 (`sph_harm`) and ≥1.15 (`sph_harm_y`,
  swapped args). Kaggle ships newer scipy than local 1.14.1 — this prevents a hard break
  at `SkyGeometry` construction on Kaggle.
- `sh_block_index`, `power_spectrum` — ℓ-block reduction for the invariant readout.
- `_wigner_d_small`, `_wigner_D_complex`, `_complex_to_real_block`,
  `real_sh_rotation_matrix`, `random_real_sh_rotation` — real-basis Wigner-D rotation.
- `wigner_3j`, `bispectrum_terms` — real trilinear CG/3j terms for the bispectrum readout.

**`src/models/diy_model.py`**:
- `SkyFiLM` → unified **`SkyReadout`** (flag `sky_readout`):
  `none` (FiLM bypassed) · `mlp121` (current; numerically identical by construction) ·
  `power` (pℓ, SO(3)-invariant) · `bispectrum` (power + CG subset, `l_bisp=4` → 14
  invariants) · `scramble` (control). Modules: `_power`, `_bispectrum`, `_scramble`,
  `_reduce`, `conditioning_param_count`.
- `DIYModel` new args: `sky_readout`, `h1l1_merge` (`concat`|`symmetric`), `h1l1_pool`
  (`mean`|`max`), `match_params`, `l_max`, `l_bisp`, `scramble_seed`. Fusion channel
  counts and `feat_dim` now depend on merge (concat 16n/feat 32n; symmetric 12n/feat 24n).
  `_sym_pool` does the swap-invariant pooling. `conditioning_param_count()` exposed.
- `SkyHeadModel` — config 5 (SH → classifier, no CNN). Same forward signature as DIYModel
  (accepts & ignores X) so the training loop needs no special-casing.
- `sky_conditioning_dim`, `hidden_for_param_count` — helpers; match_params targets
  mlp121@hidden=128 and widens the invariant readout's hidden layer to match.

**`src/model_runs.py`**:
- `set_seed` (python/numpy/torch/cuda + cuDNN deterministic; NOT
  `use_deterministic_algorithms(True)` — several 1D pool/conv backward kernels lack
  deterministic CUDA impls and would crash the P100).
- `_seed_worker` + per-epoch/shard loader `generator` → reproducible augmentation.
- `build_model` dispatch (SkyHeadModel vs DIYModel from flags). `fit` takes `seed`.
- New hyperparameter keys read in `train_from_tensors`; guards `sky_head_only` +
  `use_manifold_mixup` (incompatible; raises).

**`tests/test_invariance.py`** (13 tests) + `tests/conftest.py` (adds `src` to path).

## Validation (numbers)
- Wigner-D orthogonal 1e-14; **homomorphism** R(g₁∘g₂)=R(g₁)R(g₂) to 1e-15 and ℓ=1 block
  det +1 → genuine SO(3) representation (not an arbitrary orthogonal matrix).
- `power` invariant to rotation 1e-14 (double); `bispectrum` ~1e-6 (float32 CG weights —
  see limitation C); `mlp121`/`scramble` change under rotation.
- Symmetric merge exact under H1↔L1 swap; concat changes.
- `match_params` (n=16): mlp121 48882, power 48786, bispectrum 48810 (<0.2%,
  integer-hidden rounding; log exact counts).
- Same seed → identical AUC end-to-end (CPU); all 7 variants run end-to-end.

## Review findings / caveats (READ before Phase 3)
- **(A, fixed)** Test flakiness: `_readout` drew fc2 weights from unseeded global torch
  RNG → order-dependent; now `torch.manual_seed(0)` inside `_readout`.
- **(B)** `bispectrum` invariance is float32-limited (~1e-6), inherent — CG weight buffer
  must match the float32 signal dtype in the real model. `power` is exact. Not a defect.
- **(C, paper limitation)** `power` is exact SO(3)-invariant at the *readout-algebra*
  level (proven), but the *stored* SH coefficients carry grid-discretization + lstsq
  aliasing from the fixed 192-pixel non-iso grid, so the *physical-field* covariance is
  approximate. The mlp121-vs-power comparison is still fair (identical stored coeffs).
  State this in §6 Limitations.
- **(D)** GPU determinism untested (CPU-only local). cuDNN-deterministic + full seeding →
  same-seed near-identical, not bit-exact (atomics). Documented in `set_seed`.
- **(E)** `main()`'s combined-run default now runs at `seed=0` (was module `SEED=426425`)
  with deterministic cuDNN. Sweep overrides all of this; just don't expect the old default
  launch to reproduce byte-for-byte.
- **(F, blocker for logging)** Git hash on Kaggle: runs execute from the uploaded
  `gw-src-code` dataset, no `.git`. Commit hash must be stamped into the bundle before
  `kaggle datasets version` (e.g. write `src/_version.txt` at upload). Resolve in Phase 3.
- **(G)** Val/train class balance unverified (data on Kaggle). Disjoint by shard, no
  cross-sample sky leakage (per-sample coeffs). Constant across configs → no bias; verify
  for the dataset paragraph.
- **(H, pre-existing)** `avg_psd` computed across the dataset (incl. val noise) → mild
  train/val coupling, pre-existing, constant across configs. Not introduced here.
- **(I)** mlp121 == old SkyFiLM by construction (structure identical), not asserted by a
  test (old class removed). Submodule names changed `sky_film.*`→`sky_readout.*`; old
  checkpoints load `strict=False` only.
- **(J)** Symmetric merge has fewer params than concat (expected per plan; Z2 axis not
  param-matched — match_params only touches the sky conditioning path).
- **(K, needs sign-off)** `scramble` = independent per-sample permutation re-drawn each
  forward (marginals preserved, index→(ℓ,m) map destroyed). Stronger control than a fixed
  permutation (which would be a no-op). User to confirm this is the intended control.
- **(L)** `run_lr_range_test` still builds a default mlp121/concat model — fine for LR
  tuning (one LR for all configs per the plan), but it is not per-config.
- **(M)** Docs (`README`, `THE_SCIENCE`, `PIPELINE`, `developer-notes`, notebooks) still
  reference `SkyFiLM` / the single-readout architecture. Default architecture unchanged,
  but CLAUDE.md mandates updates when the model changes. Update in Phase 3/4.

## Sweep to run (Phase 3) — order matters, highest-value first
Tier 1: (1) none/concat  (2) mlp121/concat  (3) power/concat/match  (4) mlp121/concat/match
(5) sky_head_only.  **Check in with user after Tier 1.**
Tier 2: (6) scramble/concat  (7) power/symmetric/match.
Tier 3: (8) bispectrum/concat/match.
Every config × seeds {0,1,2}. Measure wall-clock on the FIRST Kaggle run (no faithful
local proxy) → drives the 13-day go/no-go in ANALYSIS_PLAN §6.

## Open items for Phase 3
1. Structured results logger: append one row per run (config ID, all flags, seed, total +
   conditioning param counts, AUC/acc/prec/rec/F1, wall-clock, git hash) to CSV/JSONL.
   Tables/figures generate from this file, never hand-transcribed. Never drop/select rows;
   flag diverged/anomalous runs, keep them.
2. Git-hash stamping into the Kaggle bundle (item F).
3. A sweep runner / config list driving `main()` per config (currently one config per
   `main()` launch via `kaggle/train.py`).
4. Confirm scramble design (K) and get commit go-ahead.

## Integrity guardrails (from the user, do not soften)
Same hyperparameters for every config or none. Never drop/select runs. Don't optimise
toward a conclusion — a clean negative result is fine. Don't change ANALYSIS_PLAN. No
paper prose. Flag anything that changes interpretation. State uncertainty when an effect
is inside seed variance.
