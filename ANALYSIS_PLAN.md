# Paper Skeleton
Target: NeurReps 2026 (Symmetry and Geometry in Neural Representations), deadline 23 August 2026, 11:59 UTC. Fallback: RPS, 30 August.

---

## 0. Before you run anything: pre-register

Write this into the repo as `ANALYSIS_PLAN.md` before the first run, and commit it. It takes ten minutes and it does three things: it stops you selecting a result post hoc, it makes the paper defensible when a reviewer asks, and you can state in the paper that the analysis plan was fixed in advance, which almost no workshop submission does.

Fix these before running:

- **Primary metric:** AUC on the held-out validation split. One metric. Accuracy, F1, precision, and recall are reported but are not the thing being tested.
- **Seeds:** 3 minimum, 5 if runtime allows. Same seed set across every configuration.
- **Decision rule:** the invariant readout is considered to help if its mean AUC exceeds the non-invariant readout's mean AUC by more than the pooled standard deviation across seeds. State this number before you see it.
- **What counts as the answer to the diagnostic question:** if the scramble control (see Config 7) performs the same as the real sky features, the information is not being used, and that is the finding.

### Locked values (fixed before the first run)

- **Seed set:** {0, 1, 2}. Identical across every configuration. Each seed controls model init, data-loader shuffling, augmentation RNG, mixup, and stochastic depth.
- **Primary decision comparator:** the invariant readout `power` at matched conditioning-path parameter count (Config 3, `match_params=true`) versus `mlp121` at matched conditioning-path parameter count (Config 4, `match_params=true`). The rule binds on this pair only. `power` is judged to help iff its mean validation AUC exceeds `mlp121`-matched's mean validation AUC by more than the pooled standard deviation, pooled_sd = sqrt((s_power^2 + s_mlp121^2) / 2) over the 3 seeds each.
- **Rationale:** matched parameter count is the only comparison that isolates SO(3)-invariance from capacity. `power` vs unmatched `mlp121` (Config 3 vs 2) is reported as a supporting comparison but does not trigger the decision rule.

---

## 1. The claim

One sentence, falsifiable, written now:

> Physically motivated geometric features that encode cross-detector consistency on S² are rotation-covariant, but the standard MLP readout destroys that structure; we test whether restoring SO(3)-invariance in the readout recovers the performance the features fail to deliver.

Note that **the title and abstract cannot be finalised until the experiment runs**, because two outcomes give two different papers. Write the skeleton to accommodate both.

**Outcome A (invariance helps):** the paper is "correct invariance is what makes physically motivated features usable." Title along the lines of *Invariant readouts recover geometric sky features in gravitational-wave detection*.

**Outcome B (invariance does not help):** the paper is a negative result plus a diagnosis, and the diagnosis is the contribution. Title along the lines of *Do invariant readouts rescue geometric features? A negative result on three-detector gravitational-wave detection*.

Outcome B is more likely given what you have already seen, and it is the more interesting paper. Do not treat it as the failure case.

---

## 2. Configurations to run

Two axes. Sky readout, and the H1↔L1 discrete symmetry.

**Sky readout variants:**

| ID | Readout | Dim into FiLM MLP | Invariance |
|---|---|---|---|
| S0 | none (sky pathway disabled) | 0 | n/a |
| S1 | flattened aℓm, BatchNorm, 2-layer MLP (current) | 121 | none (mixes ℓ-blocks) |
| S2 | angular power spectrum pℓ = Σm abs(aℓm)² | 11 | SO(3)-invariant |
| S3 | power spectrum plus a bispectrum subset | 11 + k | SO(3)-invariant, retains relative phase |

**Z2 variants:**

| ID | H1/L1 handling before fusion |
|---|---|
| Zc | ordered concatenation (current, breaks the swap symmetry) |
| Zs | symmetric pooling (mean or elementwise max, exactly invariant under the swap) |

**Priority tiers.** Do not attempt all eight combinations. Measure the runtime of one training run on day one, then decide how far down this list you get.

**Tier 1, required for any paper (5 configs):**

1. **S0 / Zc** baseline, no sky. You have this at AUC 0.874, but rerun it under the same seed set so the comparison is clean.
2. **S1 / Zc** current model. Reported at 0.865 to 0.871.
3. **S2 / Zc** the main test. Invariant readout, everything else identical.
4. **S1-matched / Zc** parameter-matched control (see below). This is the config reviewers will ask for.
5. **Sky head alone.** SH coefficients into a small classifier, no CNN at all. Cheap to train, and it tells you how much signal is in the geometric features independently. You already know the ℓ=0 monopole alone gives AUC around 0.60, so this extends a data point you have.

**Tier 2, strengthens the paper (2 configs):**

6. **S2 / Zs** the fully symmetry-respecting model. Both invariances imposed.
7. **Scramble control.** Feed S1 randomly permuted aℓm coefficients, preserving the marginal statistics and destroying the physical structure. If performance is unchanged, the network was never using the geometry, and that is the cleanest possible diagnosis. This is cheap and it is the single most convincing control in the whole set.

**Tier 3, only if everything else is done (1 config):**

8. **S3 / Zc** bispectrum.

### The parameter-matching problem

S2 feeds 11 numbers into the FiLM MLP where S1 feeds 121, so S2 has substantially fewer parameters. If S2 underperforms, a reviewer will immediately ask whether that is invariance or capacity, and if you cannot answer, the paper dies there.

Fix it by matching parameter count in the FiLM conditioning path across S1 and S2, either by widening S2's hidden layer or by shrinking S1's. State the parameter count for every configuration in the results table. This is Tier 1, not optional.

---

## 3. Section-by-section

Target 4 pages excluding references. Roughly 2,200 words. Check the NeurReps call for the exact page limit of the track you choose, since the extended-abstract track is shorter.

**Title.** Decide after the experiment. See Section 1.

**Abstract, 150 words.** Problem, construction, what was tested, the number, what it means. One quantitative result in the abstract, non-negotiable. Write this last.

**1. Introduction, ~350 words.** Three paragraphs plus a contribution list.

- Para 1: three-detector networks carry a geometric constraint (a real source has a single sky position producing consistent pairwise delays) that learned detectors do not encode explicitly.
- Para 2: encoding it on S² gives coefficients that transform covariantly under SO(3), but standard readouts (flatten, MLP) mix the ℓ-blocks and discard that structure. Whether restoring invariance matters is untested.
- Para 3: what we do, in two sentences.
- Contributions, two or three bullets. Not five. Something like: (i) a physically derived cross-detector consistency map on S² with a spherical-harmonic readout; (ii) a controlled comparison of covariant-but-not-invariant against invariant readouts at matched parameter count; (iii) the diagnostic result.

**2. Related work, ~200 words.** Compressed to one paragraph or two. Minimum citations: deep learning for GW detection (George and Huerta; Gabbard et al.), matched filtering as the classical baseline, spherical CNNs (Cohen et al.), Clebsch-Gordan and e3nn-style equivariant architectures (Kondor et al.; Thomas et al.), FiLM (Perez et al.), and the G2Net dataset. Read them properly. Getting an attribution wrong here loses the reviewer permanently.

**3. The sky consistency map, ~400 words.** This is the part of your existing document that transfers almost directly. Keep: the delay formula τij, the construction of C(θ,φ) by evaluating pairwise cross-correlations at geometrically predicted delays per sky pixel, the SH expansion, ℓmax = 10 giving 121 coefficients. Cut: everything about whitening, PSD estimation, noise taxonomy, and what a CNN is. One sentence saying the strain is bandpassed to 20 to 500 Hz and whitened against a noise-only PSD estimate, with a citation, is enough.

**4. Readouts and invariance, ~350 words.** The technical heart, and the reason NeurReps is the right venue. State precisely: under an SO(3) rotation each multipole transforms within its own (2ℓ+1)-dimensional block by a Wigner D-matrix; the flattened MLP mixes blocks and is therefore not equivariant; pℓ = Σm abs(aℓm)² is invariant because rotation acts by a unitary matrix within each block and preserves the norm; the bispectrum retains relative phase through Clebsch-Gordan contractions. Then the Z2 paragraph: shared extractor weights make feature extraction equivariant under the H1↔L1 swap, ordered concatenation breaks it, symmetric pooling restores it exactly with fewer parameters.

This section is mostly already written inside your portfolio document. It needs tightening, not drafting.

**5. Experiments, ~600 words.** The bulk.

- Dataset paragraph: G2Net, sample counts for train and validation, sample rate, window length, class balance. Currently under-specified in your document; state it precisely.
- Setup paragraph: seeds, optimiser, schedule, hardware, runtime. Reference the pre-registered analysis plan.
- Table 1 (configurations and parameter counts) and Table 2 (results).
- Two or three paragraphs of reading the results, in this order: does invariance change anything at matched capacity; what does the sky head alone achieve; what does the scramble control do.

**6. Limitations, ~120 words.** Specific and quantitative. Candidates, all true: one dataset; three ground-based detectors with short baselines, where timing coincidence is already strong and geometry is plausibly redundant; ℓmax fixed at 10 without a sweep; the bispectrum tested only as a subset; no comparison against a full spherical CNN on the map itself; results are AUC on a fixed validation split with no test-set holdout.

**7. Discussion, ~150 words.** One paragraph, and this is where the LISA argument lives. Ground-based networks are the regime where geometric conditioning has the least to offer, because three detectors with short baselines already provide strong timing coincidence. Long-baseline space configurations with sources in band for months are where the prior could pay. State it as a hypothesis, not a promise, and do not spend more than three sentences on it. Overreaching in the discussion is the fastest way to undo a careful experiments section.

---

## 4. Tables

**Table 1: configurations.** Columns: ID, sky readout, readout dimension, invariance property, Z2 handling, parameters in the conditioning path, total parameters. Every row of the sweep appears here.

**Table 2: results.** Columns: ID, AUC (mean ± std over seeds), accuracy, F1. Bold nothing until the numbers exist. Report every configuration you ran, including the ones that went nowhere. Selective reporting is the single most damaging thing you could do to this paper's credibility, and it is also the thing you will be tempted to do at 2am on 22 August.

---

## 5. Figures

Four pages supports two or three figures. Prioritise in this order.

**Figure 1 (required): the construction.** A Mollweide projection of C(θ,φ) for one signal sample beside one noise sample, with the SH coefficient vector below. Your existing Figure 4 (the SH bar chart) is informative but abstract; the sphere makes the idea legible in two seconds, which is what a poster needs. You should have everything required to generate this already.

**Figure 2 (required): the main result.** AUC by configuration, points with error bars over seeds, baseline drawn as a horizontal reference line. Not a bar chart. The reader must be able to see instantly whether the error bars overlap, because the effect you are measuring is roughly 0.008 in AUC and the honest answer may be that it sits inside the noise.

**Figure 3 (if space): the diagnostic.** Either ROC curves for the sky-head-alone model against the full model, or the scramble control beside the real features. Whichever more directly supports the diagnosis you end up making.

**Cut:** the architecture diagram (Figure 3 in your document). It is a beautiful figure and it belongs in the blog post, but in a 4-page paper it consumes a third of a page to convey information a paragraph handles. If the extended-abstract track has room, reconsider.

**Cut:** the PSD plot, the spectrograms, the learning curves, the confusion matrix. All blog material.

Axis labels with units, readable at print size, colourblind-safe palette, no default matplotlib styling.

---

## 6. Go / no-go on 13 days

Measure this on day one, before anything else: **how long does one training run take on your hardware?**

- Under 90 minutes: Tier 1 plus Tier 2 is comfortable. Run overnight across the week.
- 90 minutes to 4 hours: Tier 1 only, 3 seeds. Still a paper.
- Over 4 hours: 13 days is not realistic. Move to RPS on 30 August, or reduce to 3 seeds on Tier 1 configs 1, 2, 3, 4 only and accept a thinner paper.

The sky-head-alone and scramble configurations are cheap and should be run regardless, because they carry most of the diagnostic weight per GPU-hour.

---

## 7. Calendar

- **10 to 11 Aug:** OpenReview account. Read the NeurReps call and pick the track. Write `ANALYSIS_PLAN.md`. Time one training run.
- **11 to 17 Aug:** implement the readout variants and run Tier 1. Runs go overnight; your evening block is for implementation and analysis, not for watching training.
- **17 to 19 Aug:** Tier 2 if the timing allows. Generate the figures and tables from logged results, not by transcription.
- **19 to 21 Aug:** write. Sections 3 and 4 first, since they largely exist. Introduction last.
- **21 to 22 Aug:** external reader. Anonymise. Check the page limit against the correct definition.
- **22 Aug:** submit. A full day early, because submission systems fall over on deadline day.
