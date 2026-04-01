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

### Phase 2c: Loss constraints (delayed)

Three auxiliary loss terms were planned (cross-detector consistency, time-delay prior, SNR-aware weighting). These were deprioritized after Phase 2b showed that the bottleneck is not in how features are combined but in the features themselves. Loss-level constraints cannot help if the feature extractor lacks the capacity to produce the features they would constrain. SNR-aware weighting could still become useful later.

### Phase 2 Summary:

The failure of both per-branch losses (2a) and GNN aggregation (2b) narrowed the diagnosis. The bottleneck is not in per-detector feature quality (2a showed features are already as good as the architecture allows) and not in the fusion method (2b showed that more sophisticated aggregation adds nothing). The remaining explanation is the **feature extractor itself**: four plain convolutional blocks without residual connections cannot represent the hierarchical structure of gravitational wave signals.

---

## Phase 3: Deep residual backbone + geometric cross-correlation (current)

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

The second design choice is width. The Phase 1-2 model was wide (32 -> 64 -> 128 -> 256 filters) on the assumption that more filters capture more diverse features. But gravitational wave strain is a one-dimensional scalar quantity, i.e. there is no spatial structure or color channels as in images. The diversity of features at any given scale is limited by the physics: the signal is a chirp with a few parameters (masses, spins, distance, sky position). A narrower network (e.g., 32 filters throughout) with residual connections and many more layers can represent the same feature space more efficiently, because depth allows it to compose simple features into complex ones rather than needing many parallel filters to capture complex patterns in a single layer.

The implemented architecture: ~10 residual blocks (20 convolutional layers) with width 32, decreasing kernel sizes (64 -> 31 -> 15 -> 7), GeM pooling for learned downsampling, and stochastic depth. Drop probability increases linearly from 0 at block 0 to `drop_path_rate` at block 9. All 10 blocks are subject to stochastic depth, including the 3 that downsample. When a downsampling block is dropped, the shortcut path still handles projection and GeM, so the sample gets downsampled without learned features rather than skipped entirely. At `drop_path_rate=0.1` the chance of multiple downsampling blocks dropping simultaneously for one sample is negligible.

### Weight sharing

The Phase 1-2 model shared weights across all three detectors. Phase 2b's failure highlighted that $S_3$ is the wrong symmetry group. The correct structure is:

- **LIGO H1 and L1**: same instrument design, same arm length, similar noise characteristics. The same learned filters should apply to both.
- **Virgo**: different arm length (3 km vs 4 km), different seismic environment, different sensitivity curve. It should have its own feature extractor.

The residual backbone uses a shared extractor for the LIGO pair and a separate extractor for Virgo, enforcing the $Z_2$ symmetry by construction.

### $S^2$ geometric cross-correlation (implemented)

The second architectural addition exploits the geometric structure of the multi-detector network. This is described in detail in [THE_SCIENCE.md](THE_SCIENCE.md) (section: Geometric Structure of the Detector Network). In brief: the detector network defines a natural geometric domain (the sky sphere $S^2$), and cross-detector consistency can be represented as a scalar field on that sphere. Decomposing this field into spherical harmonics produces a compact, rotation-equivariant feature vector that encodes detector agreement in a physically-informed way.

The implementation uses a `SkyGeometry` class that precomputes detector positions, time delay tables for 192 HEALPix sky pixels, and the SH basis matrix once. The DataLoader computes sky features on-the-fly: for each sample, it computes normalized cross-correlations via FFT for all three detector pairs, evaluates them at the predicted delays per sky pixel, squares and sums them into a consistency score, and projects the resulting sky map onto the SH basis (l_max=8, 81 coefficients). These coefficients pass through a BatchNorm layer and concatenate with the ConcatPool output (512 + 81 = 593 features) before the classifier head.

The sky features are SO(3)-equivariant intermediate representations; the final detection output is invariant (whether a signal is present does not depend on sky position). The $Z_2$ symmetry (H1-L1 swap) is inherited automatically since cross-correlation is symmetric under pair reordering.

### Implementation sequence

Each step is gated on the previous one showing improvement.

**Step 1: Deep residual backbone (implemented).** Replaced the 4 plain conv blocks with ~10 residual blocks. Added separate Virgo extractor, GeM pooling. AUC: 0.866 (up from 0.858 plateau). Recall improved substantially (0.64 -> 0.75). Stochastic depth (drop_path_rate=0.2) resolved overfitting (val loss < train loss) without further AUC gain. Spectral dropout and channel shuffle augmentations added.

**Step 1b: Two-stage branch fusion (implemented).** V2-style fusion with n=16 channels: after the shared backbone, 4 parallel paths (H1, L1, V1 individual branches + joint branch with 1x1 projection) each with 2 ResBlocks, then all 4 concatenated (256 channels) through 4 fusion ResBlocks. LIGO H1/L1 share branch weights. ConcatPool produces 512 features for a 3-layer classifier head (512 -> 256 -> 64 -> 1). Stochastic depth extended across all 16 depth levels. LR schedule switched from cosine warm restarts to plain cosine annealing. Wall-clock time budget (8h) added to prevent Kaggle timeout. AUC: 0.874 (up from 0.866 backbone-only).

**Step 2: $S^2$ cross-correlation features (implemented, under investigation).** The model requires 81 spherical harmonic coefficients as a second input alongside the signal tensor. A `SkyGeometry` class precomputes 192 HEALPix sky pixels, the time delay model for all detector pairs, and the SH basis matrix (l_max=8). For each sample, normalized cross-correlations between all detector pairs are computed via FFT, evaluated at the predicted time delays per sky pixel, and combined into a consistency score (sum of squared correlations). The resulting sky map is projected onto the SH basis, batch-normalized, and concatenated with the CNN's 512-dimensional ConcatPool output, giving 593 classifier input features. SH coefficients are precomputed and stored in the tensor shards via `create_tensors.py` (or added to existing shards with `--add-sky`). The DataLoader falls back to on-the-fly computation if shards lack precomputed coefficients. Hyperparameters: `sky_n_pix` (default 192), `sky_l_max` (default 8).

First run with SH features: AUC 0.873, no measurable improvement over the Step 1b baseline (0.873-0.874). The offline feasibility gate passed (l=0 monopole AUC ~0.60, roughly half of all coefficients above 0.56), so the SH coefficients carry some discriminative signal. Two suspected attenuators:

1. **Mixup corrupts SH features.** Mixup linearly blends SH coefficient vectors from two unrelated samples (`lam * sky + (1-lam) * sky[perm]`). For CNN inputs this is standard, but SH coefficients encode the sky map of a specific source: mixing two sky maps from different sky positions produces a vector that corresponds to no physical configuration. This teaches the classifier to partially ignore the SH input.

2. **Feature dimension imbalance.** With n=16, ConcatPool produces 512 CNN features vs 81 SH features (13.7% of the 593-dim classifier input). Combined with dropout 0.5 in the classifier head, the SH contribution is structurally disadvantaged.

**Step 3: Training refinements.** MC dropout at inference, pseudo-labeling, rank loss fine-tuning.

---

### Please Help Out!

If you read my code and found a bug, or a possible improvement that I didn't think of, please contact me so we can discuss this further. I'd love to hear about it!

---

### Sources

- Farhadi et al. (2023) -- Mixup for acoustic signal detection
- Sun et al. (2024) -- Data imbalance and training strategies for wave detection
- Owusu et al. (2025) -- SpecAugment-style masking for signal classification
- Ta et al. (2023) -- Dropout and Gaussian noise for 1D CNNs
- Nair et al. (2023) -- Skip connections in GW detection architectures
- Sacco et al. (2022) -- Cosine annealing with warm restarts
