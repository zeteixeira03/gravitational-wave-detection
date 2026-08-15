# Sky-readout sweep: findings log

**Status: provisional. Nothing here is settled.**

Everything below was produced or concluded in one session (2026-08-12 to 2026-08-13).
Each finding carries a confidence score and a "how to challenge this" note. Treat the
confidence scores as the author's estimate at the time of writing, not as measured
quantities. A later session should attack the low-confidence items first, and should
feel free to overturn the high-confidence ones given evidence.

Raw data: `kaggle/sweep_log/v65-v69.jsonl`, 24 rows, one per (config, seed). Every
number quoted here is recomputable from those files. Nothing was hand-transcribed and
no row was excluded.

## 1. Provenance

| kernel | rows | git hash | configs |
|---|---|---|---|
| v65 | 3 | `c703ac6` | 1, 2, 3 |
| v66 | 6 | `7d2a1f9` | 1, 2, 3, 5 |
| v67 | 6 | `f8de55d` | 2, 3, 4, 5 |
| v68 | 6 | `384bccd` | 6, 7 |
| v69 | 3 | `d516368` | 8 |

24 runs, 22.0 GPU-hours on Kaggle P100. Seeds {0, 1, 2} for all 8 configs.

An earlier run (kernel v64, config 1 seed 0, hash `ab41478`) used a different recipe
and is **excluded from all analysis**. It is not in `sweep_log/`. See finding F4.

## 2. Result table

Recipe shared by all 24 runs: 100k train samples, 150k val, 20 epochs max, warmup 3,
early stopping patience 4, SWA from epoch 14, mixup off, n_channels 16, AMP on.

| config | auc mean | sd | best_auc mean | sd | params | cond. params |
|---|---|---|---|---|---|---|
| 7 power/symmetric | 0.86367 | 0.00148 | 0.86411 | 0.00073 | 3,026,648 | 114,670 |
| 1 none | 0.86213 | 0.00086 | 0.86360 | 0.00099 | 4,556,010 | 0 |
| 2 mlp121 | 0.86300 | 0.00013 | 0.86337 | 0.00034 | 4,703,964 | 147,954 |
| 4 mlp121/match | 0.86300 | 0.00013 | 0.86337 | 0.00034 | 4,703,964 | 147,954 |
| 8 bispectrum | 0.86145 | 0.00108 | 0.86297 | 0.00039 | 4,704,084 | 148,074 |
| 3 power/match | 0.85998 | 0.00349 | 0.86290 | 0.00007 | 4,704,168 | 148,158 |
| 6 scramble | 0.86065 | 0.00172 | 0.86272 | 0.00046 | 4,703,964 | 147,954 |
| 5 sky-head-only | 0.60166 | 0.00161 | 0.60469 | 0.00050 | 24,563 | 24,563 |

`auc` is the final SWA-averaged model, selected by val_loss. `best_auc` is the best
epoch's val AUC. See F5 for why both are reported.

## 3. Findings

### F1. The registered decision rule does not fire. Outcome B.

Config 3 (power, matched) against config 4 (mlp121, matched), paired by seed:

- `best_auc`: -0.00085, -0.00033, -0.00024. Mean -0.00047, pooled sd 0.00024.
- `auc`: -0.00007, -0.00223, -0.00677. Mean -0.00302, pooled sd 0.00247.

`ANALYSIS_PLAN.md` section 2 requires the invariant readout to *exceed* the
non-invariant one by more than the pooled standard deviation. It is below on both
metrics and in all three seeds.

**Confidence: 0.9** that the rule does not fire as written.
**Confidence: 0.6** that the underlying effect is a real (small) deficit rather than
noise. On `best_auc` the sign is consistent across all three seeds and the magnitude
is roughly twice the pooled sd, which is suggestive but rests on n=3.

**How to challenge:** run more seeds. Three is enough to see a sign, not enough to
size an effect of 0.0005. If a later session can afford 10 seeds on configs 3 and 4
alone, that is the single highest-value additional experiment.

### F2. The sky pathway contributes capacity, not information.

Three controls, all on `best_auc`, paired by seed:

| comparison | mean | pooled sd | reading |
|---|---|---|---|
| mlp121 vs none | -0.00023 | 0.00074 | sky features add nothing |
| scramble vs mlp121 | -0.00065 | 0.00040 | destroying the geometry costs nothing |
| bispectrum vs power | +0.00007 | 0.00028 | restoring phase recovers nothing |

Every sky-concat variant lies in a 0.0007 band from 0.86272 to 0.86337, and the
scramble control lies inside that band. The no-sky baseline sits at the top of it.

**Confidence: 0.8** in the empirical statement (the controls are flat).

**Confidence: 0.5** in the *interpretation* offered during the session, which was:
the CNN already extracts the inter-detector timing structure that the SH decomposition
encodes, so the sky features are redundant before invariance enters the picture.

That interpretation is the most challengeable claim in this document. It was never
tested directly. Competing explanations that were not ruled out:

1. The classifier head cannot use the SH coefficients regardless of their content
   (a fusion problem, not a redundancy problem). Prior FiLM work in `CLAUDE.md`
   argued fusion was not the bottleneck, but that was at l_max=8 with a different
   recipe.
2. The SH features are computed pre-augmentation from whitened signals; whitening may
   remove amplitude ratios that carry sky information.
3. 100k training samples may be too few for the network to exploit a weak second
   pathway, and the result may not transfer to the full 410k regime. See F6.
4. l_max=10 may be the wrong resolution, in either direction.

**How to challenge:** the direct test of redundancy is a probe. Train a linear readout
on the CNN's pooled features to predict the SH coefficients. High R^2 supports
redundancy; low R^2 kills the interpretation while leaving the empirical result intact.
This is cheap and should be done before the interpretation appears in a paper.

### F3. Config 7 (Z2-symmetric H1/L1 merge) is the one positive signal, and it is confounded.

Config 7 beats config 3 (same power readout, concat merge) by +0.00121 on `best_auc`,
positive in all three seeds (+0.00141, +0.00046, +0.00176), at roughly 2.3 pooled sd.
It is the only config whose mean exceeds the no-sky baseline on both metrics, though
against config 1 it is +0.00051 with pooled sd 0.00087 and therefore not established.

**The confound:** config 7 has 3,026,648 parameters against config 3's 4,704,168. The
symmetric merge shares weights between H1 and L1, which shrinks the backbone by 36%.
So config 7 differs from config 3 in two ways at once, symmetry and capacity, and the
gain could be a regularization effect from the smaller model rather than anything to do
with the Z2 structure. The sweep cannot separate these.

A narrative was proposed during the session: invariance pays when the symmetry is exact
and the map is lossless (Z2 detector exchange), and costs when invariance is bought by
discarding information (SO(3) via power spectrum). It is an appealing story and it is
**not supported by a controlled comparison**.

**Confidence: 0.75** that config 7 outperforms config 3.
**Confidence: 0.3** that the symmetry rather than the parameter reduction causes it.

**How to challenge:** run a capacity-matched symmetric variant, or a non-symmetric
weight-shared variant. Either isolates the factor. Until then the narrative should be
labelled speculative in any write-up.

### F4. The first kernel's config-1 row is not comparable and was discarded.

Kernel v64 ran config 1 seed 0 on 410k samples for 23 epochs and reported AUC 0.8669
(SWA) against a best of 0.8714 at epoch 9. The recipe was then cut (F6), so that row
shares no training configuration with any of the 24 and is excluded.

**Confidence: 0.95.** Mixing recipes across rows would invalidate the contrast.

### F5. The registered primary metric is fragile; conclusions are unchanged under the robust one.

Primary-metric standard deviations run 3x to 20x the `best_auc` ones. Config 3's sd is
0.00349 on `auc` against 0.00007 on `best_auc`, the tightest number in the table.

Cause: patience 4 makes the val_loss-selected checkpoint noisy. Two runs stopped very
early. Config 3 seed 2 stopped at epoch 5 (best epoch 5) and scored 0.8561 on `auc`
against 0.8630 on `best_auc`. Config 6 seed 0 stopped at epoch 9 with its best at
epoch 4.

Both rows were kept. No post-hoc exclusion, which is the point of the pre-registration.

Every conclusion in this document holds under both metrics.

**Confidence: 0.85** that the fragility is an artifact of checkpoint selection rather
than a property of any config.

**How to challenge:** re-derive the table from the saved per-epoch histories with a
different selection rule. If a rule exists under which the sign of F1 flips, that must
be reported.

### F6. The recipe was cut mid-project for cost reasons, and the results are conditional on it.

The original recipe cost 7.97 hours for one run, and 21 runs did not fit the GPU quota
before the deadline. Changes: train subsampled to 100k of 410k (2 shards of 9), epochs
50 to 20, warmup 7 to 3, patience 10 to 4, SWA start 15 to 14.

Absolute AUC sits near 0.862 against the project's historical 0.874 no-sky baseline.
Part of that gap is the subsample and the shorter schedule, and part is mixup being off
for the whole sweep by design.

**Confidence: 0.85** that the relative contrast between configs survives subsampling,
since every config pays the same cost.
**Confidence: 0.45** that the same conclusions hold at 410k samples. A weak second
pathway is exactly the kind of thing that could need more data to become useful.

**How to challenge:** re-run configs 1, 3 and 4 at full data, one seed each, for about
24 GPU-hours. If the ordering is preserved, F1 and F2 become much stronger.

Note also: warmup was kept at 3 epochs while epochs contain 4x fewer optimizer steps
than the tuning that produced the original schedule. This was not re-tuned.

### F7. The harness is deterministic.

Config 4 reproduces config 2 bit-identically at all three seeds: 0.8629758026,
0.8631510752, 0.8628862650, matching to 10 decimal places, along with `best_auc`,
`best_epoch`, `epochs_run` and both losses.

This was predicted before the runs. `match_params` targets mlp121 at hidden=128, so
matching mlp121 to itself is a no-op, and the parameter counts were measured equal
(147,954) in a smoke test before any AUC existed.

**Confidence: 0.95.** Seed spread in the table is training variance, not run-to-run
drift.

### F8. Sky features carry weak but real signal on their own.

Config 5 (sky-head-only, no CNN, 24,563 parameters) scores 0.60166 across three seeds,
matching the ~0.60 predicted by the earlier l=0 monopole feasibility gate.

**Confidence: 0.9.** This matters because it rules out the trivial explanation for F2,
that the SH features are empty. They are informative and still not additive.

### F9. The tensor dataset carries l_max=10, inferred rather than observed.

Derived from Kaggle file sizes: differencing shard_00 (50,000 samples, 2,482,202,154
bytes) against shard_11 (10,000 samples, 496,442,154 bytes) gives 49,644 bytes per
sample. Subtracting 49,152 for signals (3 x 4096 float32) and 8 for an int64 label
leaves 484 bytes, which is 121 float32 coefficients. Container overhead is 2,154 bytes
in both shards, and 11 x 50,000 + 10,000 matches the G2Net train set exactly.

**Confidence: 0.85.** The arithmetic is self-consistent but assumes int64 labels and
that no other field is stored. It was never verified by loading a shard and printing
`sh_coeffs.shape`.

**How to challenge:** one line in a scratch kernel. Worth doing before the paper.

## 4. Process notes worth keeping

- **Each kernel writes its own `sweep_results.jsonl` to `/kaggle/working`, and
  `kaggle kernels output` only returns the latest version.** Rows must be pulled and
  archived before the next kernel is pushed or they become unreachable. This is why
  `kaggle/sweep_log/` exists.
- **A run must not start unless its full time budget fits the kernel.** The first
  sweep kernel shrank `max_train_hours` to the remaining time, which would have
  produced rows whose AUC reflected the queue rather than the config. Fixed in
  `c703ac6`.
- **`max_train_hours` 8.0 inside an 8.5h kernel budget** meant one run per kernel and
  four silent skips. The guard behaved correctly; the two constants were incompatible.

## 5. A correction made during the session

Config 4 was dropped from the tier lists on the argument that its parameter count is
identical to config 2, making it duplicate compute. That was wrong. `ANALYSIS_PLAN.md`
section 2 pins the decision rule to config 3 against config 4 and states that config 3
against config 2 is only a supporting comparison. Dropping config 4 would have left the
registered primary comparison with no data under its own label. It was restored in
`f8de55d` and run at all three seeds, which also produced F7.

Generalization worth carrying forward: an argument that two configs are numerically
identical is not an argument for deleting one of them when a pre-registration names
both.

## 6. What a later session should do first

1. The redundancy probe in F2. It is cheap and it decides whether the paper's central
   interpretation survives.
2. The capacity-matched control in F3. Without it, config 7's story cannot be told.
3. The l_max verification in F9. One line.
4. More seeds on configs 3 and 4 (F1), if compute allows.
5. Full-data replication of configs 1, 3, 4 (F6), if compute allows.

## 7. Overall

**Answer:** Outcome B. The invariant readout does not beat the non-invariant one at
matched parameters, and the controls indicate the sky pathway is not contributing
information in any readout form.

**Confidence: 0.75** in the headline. The empirical numbers are solid; the diagnosis
that explains them is not yet tested.

**Critical uncertainties:** whether the redundancy interpretation is correct (F2),
whether config 7's gain is symmetry or capacity (F3), whether any of this survives at
full data (F6), and whether effects of 0.0005 are resolvable at n=3 seeds (F1).
