# Follow-up plan: probe and capacity control

**Status: registered before any run.** This file is committed before the kernel that
runs it. Do not edit the rules below after a follow-up row exists. Add a dated
amendment at the end instead, and say which rows existed when it was written.

`ANALYSIS_PLAN.md` governs the 24-run sweep and is not changed. This file governs
new runs only. It answers the open items in `SWEEP_FINDINGS.md` section 6, in order:

| ID | Question | Finding it attacks | Cost |
|---|---|---|---|
| R0 | Does the harness still reproduce a logged row? | F7 | inside P1 |
| P1 | Is the sky pathway redundant with the CNN, or did fusion fail? | F2 | ~1.5 GPU-h |
| P3 | Are the stored SH coefficients l_max=10, as inferred? | F9 | inside P1 |
| P2 | Is config 7's gain caused by the Z2 constraint or by its smaller size? | F3 | ~3.3 GPU-h |

One kernel runs all of it: `MODE = "followup"` in `kaggle/train.py`.

## 0. Two facts found while designing this

These change how earlier findings read. They are recorded in `SWEEP_FINDINGS.md` as
F10 and F11. They are repeated here because the rules below depend on them.

**SWA never activated.** `fit` averages weights only from epoch index 14 (the 15th
epoch). The longest sweep run stopped after 13 epochs. So every `auc` in the sweep
is the best-val-loss checkpoint, not an SWA average.

**The channel-shuffle augmentation is on for every config.** With probability 0.5
it swaps the H1 and L1 channels of a training sample. The SH coefficients are not
recomputed. Three consequences:

1. Every concat model is trained toward H1<->L1 swap invariance. The Z2 symmetry
   is therefore already imposed softly, by data. Config 7 imposes it exactly, by
   architecture. Config 3 versus config 7 is "soft versus hard", not "absent versus
   present".
2. A swap-invariant CNN cannot represent the part of the sky geometry that changes
   sign under the swap (the sign of the H1-L1 delay, and which of H1/L1 pairs with
   which Virgo delay). This works against the redundancy interpretation of F2.
3. For an isotropic source population, 16.3% of swapped signal samples have a delay
   triplet that no sky direction produces (9.3% beyond a one-sample tolerance at
   2048 Hz). Computed from the detector positions in `sky_feasibility.py`.

## R0. Reproduction gate

The probe job retrains config 1, seed 0, under the sweep recipe. It is logged as
`probe_1_none_concat` so it never enters the sweep table.

Reference row: `kaggle/sweep_log/v65.jsonl`, git `c703ac6`.

| field | reference |
|---|---|
| `auc` | 0.863110875492951 |
| `best_auc` | 0.864665029661324 |
| `best_epoch` / `epochs_run` | 7 / 12 |
| `final_val_loss` | 0.4412289270082228 |

**Rule.** R0 passes if `auc` and `best_auc` match to 1e-6 and the epochs match.

**If R0 fails,** read the row's `env` field (new in this version). A changed torch,
CUDA or cuDNN version means the Kaggle image moved, and new rows are not comparable
to old rows at the precision P2 needs. Then P2 is not read against the old config 3
and 7 rows. Configs 3 and 7 are re-run at seeds 0-2 in the new environment first,
and P2 is read only on rows from the same environment. P1 does not depend on R0,
because it compares feature sets on one model.

## P3. Stored-feature check

`run_probe` recomputes the SH coefficients for every validation sample with the
current `SkyGeometry` and compares them to the stored ones.

**Rule.** F9 is confirmed if `sh_check.stored_shape` is `[150000, 121]` and
`max_rel_diff` < 1e-3. Otherwise stop: the sweep's sky features are not what the
code says they are, and every sky-config row needs re-examination.

## P1. Redundancy probe

### What is measured

Model: the R0 retrain of config 1. Config 1 has no sky pathway, so its pooled
features (512 values, the input to the classifier head) come from the CNN only.

Data: the 150,000 validation samples, which the CNN never trained on. A fixed
permutation (seed 0) splits them into 75,000 probe-train and 75,000 probe-test.

Label probes: one fixed MLP head (input -> 256 -> 64 -> 1, SiLU, dropout 0.2, AdamW
1e-3, early stopping on 10% of probe-train) is trained on each feature set and
scored by AUC on probe-test. The feature sets are `cnn`, `sky`, `cnn+sky`,
`sky_l0`, `sky_lge1`, `cnn+sky_lge1`, `sky_even`, `sky_odd` and `cnn_random`.

Representation probes: ridge regression, penalty chosen per target by 5-fold CV on
probe-train, held-out R^2 on probe-test. Targets: the 121 a_lm, the 11 log p_l, and
the swap-even and swap-odd parts of a_lm. Inputs: the trained features, and the
features of the same architecture at its untrained init with recalibrated BatchNorm.

### Primary statistic and rule

Delta = AUC(`cnn+sky`) - AUC(`cnn`) on probe-test, with a paired-bootstrap 95%
interval (1000 resamples). Implemented in `probe.read_outcome`, applied in the kernel.

| outcome | condition | reading |
|---|---|---|
| `sky_uninformative` | AUC(`sky`) < 0.55 | The probe cannot discriminate. Contradicts F8. |
| `complementary` | CI lower bound > 0 | The information is there and the end-to-end sky pathway did not use it. F2's redundancy interpretation is rejected. Fusion failure (F2 explanation 1) becomes the lead. |
| `redundant` | CI upper bound < +0.001 | SH carries label information alone (F8) but adds none to the CNN features. |
| `inconclusive` | otherwise | Report the interval. No interpretation. |

The 0.55 gate is the existing feasibility gate in `sky_feasibility.py`. The 0.001
margin is about twice the F1 deficit (0.00047) and above every sky-readout contrast
in F2 (largest 0.00065). A smaller margin would demand more precision than 75,000
test samples are likely to give; a larger one could hide an effect of the size the
sweep was measuring.

### Wording rule for a write-up

The sentence "the CNN already extracts the inter-detector geometry the SH features
encode" may appear only if all four hold:

1. outcome is `redundant`;
2. the secondary delta, AUC(`cnn+sky_lge1`) - AUC(`cnn`), also has CI upper bound < +0.001;
3. AUC(`sky_lge1`) >= 0.55, so there is direction-dependent information to be
   redundant with;
4. ridge `r2_lge1` for the trained features exceeds that for the random-init
   features by at least 0.05, so training put direction-dependent sky structure
   into the CNN.

If 1 holds and any of 2-4 fail, the allowed sentence is the weaker one: "the SH
features add no label information to the CNN features". It says nothing about why.

### Predictions, stated before the data

These are not decision rules. They are written down so they can be scored.

- **Q1.** `swap.z_odd_var_frac` for the trained model is below 0.05 and below
  `swap.z_rand_odd_var_frac`. Reason: the channel-shuffle augmentation trains
  approximate swap invariance; the concat architecture at init has none.
- **Q2.** If Q1 holds, ridge R^2 from trained features to `alm_odd` is below 0.05.
- **Q3.** AUC(`sky_odd`) is below the 0.55 gate. If it is above, the augmentation
  is hiding usable label information from the CNN, and the outcome should be
  `complementary`. A `redundant` outcome together with AUC(`sky_odd`) >= 0.55 is a
  contradiction to investigate before anything is written.
- **Q4.** AUC(`cnn`) on probe-test is within 0.002 of `label_auc.model_on_probe_test`.
  The probe head and the model's own head see the same features.

## P2. Capacity control for config 7

### The confound

Config 7 (power readout, symmetric merge) beat config 3 (power readout, concat
merge) by +0.00121 `best_auc`. It also has 36% fewer parameters (3,026,648 against
4,704,168), almost all of it in the fusion stack (12n against 16n channels). Two
factors changed at once.

### The control: config 9, `power_project_match`

Config 9 is config 7 with one change. At the two points where config 7 merges the
H1 and L1 tensors with the mean (backbone output into the joint branch, and branch
output into the fused stack), config 9 uses a learned 1x1 convolution on the
ordered concatenation [H1, L1]. Each projection is bias-free, 8n -> 4n.

- **Same function at init.** The projections start at [I/2, I/2], which is the
  mean. Every other weight, and the global RNG state after construction, is
  identical to config 7 at the same seed (tested:
  `test_project_merge_equals_symmetric_at_init`).
- **Same capacity.** 3,043,032 parameters: +16,384 (+0.54%) over config 7, against
  a 1,677,520 gap to config 3.
- **Constraint lifted.** Once the two halves differ, the model is not swap-invariant
  (tested). It still sees the channel-shuffle augmentation, so it is pushed toward,
  not held at, the symmetric solution.

This puts three configs at three corners of a 2x2 design:

| | small (12n fusion) | large (16n fusion) |
|---|---|---|
| hard symmetry | 7 | (not run) |
| soft symmetry (augmentation only) | 9 | 3 |

### Rule

Metric: `best_auc`, primary. `auc` reported beside it. `best_auc` is chosen over the
registered `auc` because F5 showed checkpoint-selection noise 3x to 20x larger on
`auc`; the per-epoch curves are now logged, so any other selection rule can be
applied later from the same rows. Seeds {0, 1, 2}. Pooled sd as in `ANALYSIS_PLAN.md`.

- d_constraint = mean(7) - mean(9), threshold sd_79 = pooled sd of 7 and 9.
- d_capacity = mean(9) - mean(3), threshold sd_93 = pooled sd of 9 and 3.

| reading | condition |
|---|---|
| constraint causes the gain | d_constraint > sd_79 and abs(d_capacity) <= sd_93 |
| capacity causes the gain | d_capacity > sd_93 and abs(d_constraint) <= sd_79 |
| both contribute | both exceed their thresholds, same sign as the F3 gain |
| not resolved | anything else |

A reading counts as strong only if it holds on both `best_auc` and `auc`.

"Not resolved" is a likely result: the F3 gain is 2.3 pooled sd at n=3, and this
design splits it into two parts. That outcome still settles one thing: the
symmetry narrative in F3 stays labelled speculative.

## Budget and order

One kernel, 8.5 h budget: probe job (~1.2 h training + ~0.4 h probe), then config 9
seeds 0, 1, 2 (~1.1 h each). About 5 GPU-hours. The second concurrent session stays
free for the R0-failure contingency.

After the kernel, pull `sweep_results.jsonl` and `probe_results.jsonl` into
`kaggle/sweep_log/` before pushing anything else (`SWEEP_FINDINGS` section 4).

## Not in this plan

- More seeds on configs 3 and 4 (F1), and full-data replication (F6). Both remain
  the highest-value compute after this plan.
- A no-shuffle variant of configs 1 and 2. Whether it is worth running depends on
  Q1 and Q3. If it is run, its rule is registered here as an amendment first.
