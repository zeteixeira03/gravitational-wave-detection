# Developer Notes

This document records the reasoning behind architectural decisions in this project. Each section describes what was tried, what was learned, and how it informed the next step. The goal is to make the chain of reasoning legible.

---

## Phase 1: Optimization (concluded)

Phase 1 focused on training dynamics and regularization. The question was: is the model's performance limited by how well we train it, or by what it can represent?

### Results

Each change was tested on the full 560k-sample dataset. Changes were cumulative.

| Run | Change introduced | AUC | Accuracy | Val loss |
|-----|-------------------|-----|----------|----------|
| 1 | Baseline (3 epochs, LR 1e-3) | 0.768 | 0.549 | 3.538 |
| 2 | LR 1e-4, dropout 0.5, 50 epochs | 0.854 | 0.784 | 0.564 |
| 3 | Re-run | 0.854 | 0.785 | 0.593 |
| 4 | + weight decay 1e-4 | 0.852 | 0.781 | 0.714 |
| 5 | Re-run | 0.854 | 0.784 | 0.684 |
| 6 | + mixup augmentation (alpha=0.2) | 0.859 | 0.791 | 0.447 |
| 7 | Re-run | 0.856 | 0.786 | 0.472 |
| 8 | + LR warmup (5 epochs) | 0.858 | 0.788 | 0.468 |
| 9 | + LR 5e-5, weight decay 5e-4 | 0.858 | 0.788 | 0.459 |
| 10 | + cosine annealing (T_0=10) | 0.858 | 0.787 | 0.461 |

Key observations:

- **Runs 2-5** (LR/dropout/weight decay tuning): AUC stuck at 0.852-0.854. The model was overfitting (train loss ~0.23, val loss ~0.69).
- **Run 6** (mixup): Best single improvement. AUC jumped to 0.859, and the train-val gap collapsed (0.40 vs 0.45), confirming overfitting was addressed. But val performance barely moved.
- **Runs 8-10** (warmup, LR reduction, cosine annealing): No measurable improvement. AUC flat at 0.858.

### Conclusion

The model fits the training data cleanly but cannot extract more signal from it. The remaining error is not from poor optimization or overfitting. It is architectural: the model cannot represent the features it would need to improve further. This motivated Phase 2.

### Phase 1 Implementation Summary

- Time shift augmentation (0-20 samples per detector)
- Gaussian noise injection (1-10% of signal std)
- Mixup ($\alpha=0.2$)
- LR warmup (5 epochs, linear from 1e-6 to target LR)
- Cosine annealing ($\eta_{min}$=1e-6)
- AdamW with weight decay 5e-4
- Early stopping (patience=10)

### Put on Hold:

These remaining ideas were deprioritized because the plateau is architectural:

| Technique | Reason skipped |
|---|---|
| Label smoothing | Mixup already provides soft labels |
| Conv layer dropout | Overfitting is already solved |
| Reduced model capacity | Model is not overfitting; fewer params would hurt |
| Amplitude scaling augmentation | Unlikely to help given the architectural ceiling |
| Channel dropout augmentation | Conceptually better addressed by per-branch losses |
| Skip connections | Architectural change, moved to Phase 2 scope |

---

## Phase 2: Architecture experiments (concluded)

With training dynamics confirmed healthy, Phase 2 tested two hypotheses about *where* the architectural bottleneck lies: in the per-detector feature quality, or in how detector features are combined.

### Phase 2a: Auxiliary per-branch losses

**Hypothesis:** Individual detector branches produce weak features because the main loss only supervises the concatenated output. Adding per-branch classification heads would force each branch to learn independently useful representations.

**Result ($w_{\text{aux}}$=0.2):** AUC 0.857, accuracy 0.780 -> no improvement. The per-branch supervision did not unlock new capacity.

Thus the bottleneck is not weak per-detector features. Each branch already extracts what it can from its input. The problem lies elsewhere.

### Phase 2b: GNN aggregation head

**Hypothesis:** The concatenation-based classifier treats the three detectors as an ordered tuple, which is permutation-sensitive. A graph neural network over the 3-detector graph would enforce permutation invariance ($S_3$ symmetry) and incorporate cross-correlation edge features encoding the physical time delays between detectors.

**Result:** AUC 0.858 -- no measurable change. The GNN has since been removed from the codebase.

**This tells us two things:**

First, the $S_3$ symmetry argument was wrong. The three detectors are *not* exchangeable: LIGO Hanford and Livingston are the same instrument design (4 km arms, similar noise profiles), but Virgo is fundamentally different (3 km arms, different seismic environment, different sensitivity curve). The correct symmetry group is $Z_2$ (swap the two LIGO detectors), not $S_3$. Enforcing the wrong symmetry constrains the model without benefit.

Second, a complete graph on 3 nodes is topologically trivial. One round of message passing on $K_3$ is equivalent to a fully connected layer that happens to share weights across node positions (essentially an MLP with extra steps). There is no graph structure for the GNN to exploit. The cross-correlation edge features (113-dimensional vectors per pair) added compute cost but no information the network could not already access through the concatenated features.

### Phase 2 Summary:

The failure of both per-branch losses (2a) and GNN aggregation (2b) narrowed the diagnosis. The bottleneck is not in per-detector feature quality (2a showed features are already as good as the architecture allows) and not in the fusion method (2b showed that more sophisticated aggregation adds nothing). The remaining explanation is the **feature extractor itself**: four plain convolutional blocks without residual connections cannot represent the hierarchical structure of gravitational wave signals.

---

## Phase 3: Deep residual backbone + geometric cross-correlation (concluded)

### Why depth matters: a first-principles argument

A gravitational wave from a compact binary merger is a *chirp*: the frequency increases monotonically as the binary spirals inward, sweeping from ~30 Hz to several hundred Hz over the final seconds before coalescence. The waveform has structure at multiple timescales simultaneously:

1. **Individual cycles** (~2-30 ms): the oscillation period at a given moment, determined by the instantaneous orbital frequency.
2. **Frequency evolution** (~100-500 ms): the rate at which the cycle period shortens, governed by the masses and spins of the binary.
3. **Amplitude envelope** (~1-2 s): the overall growth in signal strength as the binary radiates more energy at higher frequencies, culminating in the merger.

A single convolutional layer with a 64-sample kernel (~31 ms) can capture individual cycles. But it cannot capture the fact that successive cycles are shorter and louder in a specific pattern governed by GR. That requires composing features across layers: layer 1 detects oscillations, layer 2 detects that the oscillation frequency is changing, layer 3 detects that the rate of change follows a power law, and so on.

The Phase 1-2 architecture had four convolutional blocks. After four rounds of convolution and 4x pooling, the temporal resolution was reduced by a factor of 256 (4096 -> 16 time steps). The network had exactly four opportunities to compose features before the signal was compressed into 16 temporal positions. This was not enough to build the hierarchical representation described above.

The solution is therefore adding depth, but with a caveat. Adding more convolutional layers to a plain network causes vanishing gradients: the loss signal attenuates exponentially as it propagates backward through many layers, and the early layers (which see the raw waveform and are arguably the most important) receive almost no useful gradient. Residual connections solve this by providing a direct path for gradients to flow backward:

$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

The gradient of the loss with respect to any early layer always includes a term that bypasses all intermediate layers. This makes the optimization landscape smoother and allows training networks with tens or hundreds of layers.

The second design choice is width. The Phase 1-2 model was wide (32 -> 64 -> 128 -> 256 filters) on the assumption that more filters capture more diverse features. But gravitational wave strain is a one-dimensional scalar quantity: there is no spatial structure or color channels as in images. The diversity of features at any given scale is limited by the physics -- the signal is a chirp with a few parameters (masses, spins, distance, sky position). A narrower network with residual connections and many more layers can represent the same feature space more efficiently, because depth allows it to compose simple features into complex ones rather than needing many parallel filters to capture complex patterns in a single layer. Current runs use base width n=16 (progressing 16 -> 32 -> 64 through the backbone, then 256 channels in the fusion stage), which fits the Kaggle GPU budget comfortably and leaves room for the depth increase.

The implemented architecture: 10 residual blocks (20 convolutional layers) with base width n=16, decreasing kernel sizes (64 -> 31 -> 15 -> 7), GeM pooling for learned downsampling, and stochastic depth. Drop probability increases linearly from 0 at the first block to `drop_path_rate` at the last block of the 16-level schedule (10 backbone + 2 parallel branch + 4 fusion). All 10 backbone blocks are subject to stochastic depth, including the 3 that downsample. When a downsampling block is dropped, the shortcut path still handles projection and GeM, so the sample gets downsampled without learned features rather than skipped entirely.

### Weight sharing

The Phase 1-2 model shared weights across all three detectors. Phase 2b's failure highlighted that $S_3$ is the wrong symmetry group. The correct structure is:

- **LIGO H1 and L1**: same instrument design, same arm length, similar noise characteristics. The same learned filters should apply to both.
- **Virgo**: different arm length (3 km vs 4 km), different seismic environment, different sensitivity curve. It should have its own feature extractor.

The residual backbone uses a shared extractor for the LIGO pair and a separate extractor for Virgo, enforcing the $Z_2$ symmetry by construction.

### $S^2$ geometric cross-correlation (implemented)

The second architectural addition exploits the geometric structure of the multi-detector network. This is described in detail in [THE_SCIENCE.md](THE_SCIENCE.md) (section: Geometric Structure of the Detector Network). In brief: the detector network defines a natural geometric domain (the sky sphere $S^2$), and cross-detector consistency can be represented as a scalar field on that sphere. Decomposing this field into spherical harmonics produces a compact, rotation-equivariant feature vector that encodes detector agreement in a physically-informed way.

The implementation uses a `SkyGeometry` class that precomputes detector positions, time delay tables for 192 HEALPix sky pixels, and the SH basis matrix once. SH coefficients are precomputed at shard creation time via `create_tensors.py` (or added to existing shards with `--add-sky --l-max <n>`); the DataLoader falls back to on-the-fly computation from a clean (pre-augmentation) copy of the signal when shards lack coefficients. The coefficients condition the pooled CNN features through a FiLM layer (see below), not through concatenation.

### Implementation sequence

Each step was gated on the previous one showing improvement.

**Step 1: Deep residual backbone.** Replaced the 4 plain conv blocks with ~10 residual blocks. Added separate Virgo extractor, GeM pooling. AUC: 0.866 (up from the 0.858 plateau). Recall improved substantially (0.64 -> 0.75). Stochastic depth (drop_path_rate=0.2) resolved overfitting (val loss < train loss) without further AUC gain. Spectral dropout and channel shuffle augmentations added alongside.

**Step 1b: Two-stage branch fusion.** V2-style fusion with n=16 channels: after the shared backbone, 4 parallel paths (H1, L1, V1 individual branches + joint branch with 1x1 projection) each with 2 ResBlocks, then all 4 concatenated (256 channels) through 4 fusion ResBlocks. LIGO H1/L1 share branch weights. ConcatPool produces 512 features for a 3-layer classifier head. Stochastic depth extended across all 16 depth levels. LR schedule switched from cosine warm restarts to plain cosine annealing. Wall-clock time budget (8h) added to prevent Kaggle timeout. AUC: 0.874 (no-sky baseline, up from 0.866 backbone-only).

**Step 2: $S^2$ cross-correlation features.** The offline feasibility gate passed cleanly: the $\ell = 0$ monopole of the sky consistency map reaches AUC $\sim 0.60$ as a univariate classifier, and roughly half of the remaining SH coefficients exceed AUC 0.56 individually. The features carry discriminative signal. In spite of that, every run with sky features turned on landed in the 0.865-0.871 band, just under the 0.874 no-sky baseline.

The first sky run (concatenation, all augmentations on) reached 0.873, within the no-sky error bar. The working hypothesis was that several of the Phase 1 augmentations were actively hostile to the sky map: input-space mixup blends signals from different sky positions (the interpolation corresponds to no physical source), spectral dropout randomly zeroes FFT bins and destroys inter-detector phase coherence, and time shift per detector breaks the precomputed delay geometry the sky map depends on. Stripping those three augmentations should have let the sky features express themselves. The rest of Phase 3 was a regularization ablation to find out whether a sky-compatible replacement stack could push past the baseline.

| Run | Config changes vs. no-sky baseline | Train loss | Val loss | AUC |
|-----|------------------------------------|------------|----------|-----|
| Baseline (no sky) | all augs on | 0.59 | 0.42 | 0.874 |
| First sky run (concat) | + sky concat, all augs on | -- | -- | 0.873 |
| Exp E | strip mixup / spectral / time-shift / noise, dropout 0.3, no aux | 0.34 | 0.46 | 0.870 |
| Exp F | restore dropout 0.5 + Gaussian noise | 0.33 | 0.43 | 0.872 |
| Exp G | + label smoothing 0.1 + amplitude scaling | 0.45 | 0.43 | 0.868 |
| FiLM pivot | concat -> FiLM fusion, $\ell_{\max}=8$, no-decay param group | 0.34 | 0.46 | 0.871 |
| Final | manifold mixup + SWA + drop_path 0.3 + $\ell_{\max}=10$ | 0.43 | 0.47 | 0.865 |

Each row taught something specific:

- *Stripping augmentations hurts generalization.* Exp E removed mixup, spectral dropout, time shift, and Gaussian noise together. Train loss collapsed to 0.34 (pure memorization) while val loss climbed to 0.46. The regularization deficit swallowed whatever contribution the sky features could have made; the run did not isolate the sky effect.
- *Dropout and Gaussian noise alone do not replace the stripped stack.* Exp F restored dropout 0.5 and input-space Gaussian noise (both sky-compatible). The train-val gap improved, but AUC only recovered to 0.872, still below the no-sky 0.874.
- *Label smoothing fixed overfitting but hurt AUC.* Exp G added label smoothing 0.1 and amplitude scaling. Train loss rose to 0.45, val loss held at 0.43, but AUC dropped to 0.868. Smoothing prevents the model from committing to confident correct predictions on low-SNR positives, which is exactly the regime where the sky map is supposed to help most.
- *Concatenation was not the bottleneck, but FiLM is still the right call.* Swapping concat for FiLM at $\ell_{\max}=8$ reached 0.871, within the same band. The fusion mechanism was not what kept sky features from helping. FiLM stays in the final model because it is the structurally cleaner way to combine a narrow physics-derived vector with a much wider learned representation: the 121-dim SH vector gets direct multiplicative control over every CNN channel instead of being swallowed by the 512-dim CNN path at the input of the classifier.

The final run stacked four sky-compatible changes, each targeting a different failure mode from the ablations above: *manifold mixup* (alpha=0.4) on the pooled feature vectors, which restores mixup-strength regularization without touching the phase-sensitive time-domain representation; *stochastic weight averaging* over the last $\sim 15$ epochs with an SWALR constant LR schedule, chosen to smooth out the noise in the final solution; *drop_path_rate* 0.2 $\to$ 0.3, a cheap bump to the already-present stochastic depth; and *$\ell_{\max}$* 8 $\to$ 10, which grows the SH basis from 81 to 121 coefficients so the sky side matches the order of magnitude of the 512-dim CNN features the FiLM MLP conditions. AUC landed at 0.865 on the SWA-averaged model (peak single-epoch AUC was 0.871 at epoch 11), worse than the no-sky 0.874 baseline and within the same 0.865-0.871 band every sky-on run has occupied.

### What the sky features cost, and why they stay

The empirical story is simple: on this dataset, the sky features do not help. Three detectors, an already-saturated cross-detector fusion block, and a training set where the augmentations most effective at regularizing 1D CNNs happen to be precisely the ones that destroy cross-detector phase coherence. None of the sky-compatible replacements tried in the Phase 3 ablation recovered the generalization that the stripped augmentations provided. Every sky-on configuration paid a regularization tax that the geometric features could not repay.

The construction itself is still worth keeping on the record. The sky consistency map is a physical object: for each direction on the sphere, it asks how well the three detectors agree that a signal came from there, using only cross-correlations at the geometrically predicted time delays. The spherical harmonic decomposition turns that function into a compact, rotation-structured feature vector. The FiLM layer lets those coefficients modulate a learned CNN representation channel-by-channel rather than compete with it at a concatenation point. The whole pipeline is bounded ($\gamma \in (-1, 1)$), starts at identity (zero-init output projection), and is trivially toggled on and off. It is also general: any detector network that can be mapped to sky-direction-dependent time delays admits the same construction.

LISA is the obvious place where this kind of geometric conditioning could matter. A ground-based network of three detectors with short baselines already extracts nearly all the cross-detector coincidence information through simple time-domain fusion. LISA is a very different regime: three spacecraft in a 2.5 million km triangle, sources that dwell in band for weeks to months, a strong source-direction dependence in the time-delay interferometry combinations, and a detection problem where geometric consistency across the network is central rather than incidental. The sky-map + SH + FiLM construction is left in the code as a small contribution in that direction -- a testable, physics-grounded way to inject geometric evidence into a learned detection pipeline, even if the G2Net dataset was not the right testbed for it.

### Final model

The saved model for this project is the final run: V2 residual backbone + V2 two-stage fusion + SkyFiLM driven by 121 SH coefficients, trained with manifold mixup, SWA, and the sky-compatible augmentation stack. Validation metrics (full 112k-sample split):

| Accuracy | AUC | Precision | Recall | Specificity |
|----------|-----|-----------|--------|-------------|
| 0.798 | 0.865 | 0.933 | 0.640 | 0.954 |

The high precision / low recall profile is a direct consequence of the training objective: BCE on a mildly imbalanced dataset with no calibration step, so the threshold-0.5 operating point sits deep in the high-precision corner of the ROC curve. The full curves and the dashboard are in [assets/dashboard.png](assets/dashboard.png).

---

### Sources

- Farhadi et al. (2023) -- Mixup for acoustic signal detection
- Sun et al. (2024) -- Data imbalance and training strategies for wave detection
- Owusu et al. (2025) -- SpecAugment-style masking for signal classification
- Ta et al. (2023) -- Dropout and Gaussian noise for 1D CNNs
- Nair et al. (2023) -- Skip connections in GW detection architectures
- Sacco et al. (2022) -- Cosine annealing with warm restarts
- ZiyueWang25 / Kaggle_G2Net (GitHub) -- V2 two-stage fusion reference implementation
