# Next Steps

This document tracks planned improvements across three phases: fixing training dynamics, embedding physical knowledge into the architecture and loss function, and (if warranted) adding waveform structure constraints.

---

## Phase 1: Optimization

The model currently plateaus at 78.5% accuracy with signs of early overfitting. The priority is to fix training dynamics before adding complexity.

### The Problem

1. Validation loss bottoms at epoch 3: abnormally fast convergence, suggesting the model finds local patterns but fails to generalize
2. Training loss continues dropping after validation diverges, meaning the model has capacity to learn more, but memorizes instead of generalizing
3. No data augmentation: the model sees identical samples each epoch (of course it will memorize)

The core issue is that the model learns fast, not well.

### Data Augmentation

The pipeline has zero data augmentation. This is the single biggest lever for improvement.

| Technique | Rationale | Implementation |
|---|---|---|
| Time shift | GW signals arrive at different times at each detector (up to ~10ms offset); shifting is label-preserving | Random offset of 0-20 samples per detector |
| Gaussian noise | Detector noise varies between observations; small noise injection builds robustness | Add noise with std = 0.01-0.1 of signal std |
| Amplitude scaling | GW amplitude varies with source distance; scaling preserves signal structure | Scale by factor 0.8-1.2 |
| Mixup | Interpolates training pairs to create virtual samples; proven for time-series (Farhadi et al., 2023) | $$x_{new} = \lambda \cdot x_1 + \frac{1 - \lambda}{x_2}, \lambda \sim Beta(0.2, 0.2)$$ |
| Channel dropout | Zeroing one detector forces the model to not over-rely on any single channel | Set one of 3 channels to zero with p=0.1 |
| SpecAugment masking | Masks time/frequency regions; forces distributed feature learning (Owusu et al., 2025) | Random rectangular masks in time domain |

Priority: time shift + Gaussian noise first, then Mixup.

### Training Dynamics

| Change | Rationale | Current -> Planned |
|---|---|---|
| LR warmup | Prevents early aggressive updates pushing into poor local minima | None -> 3-5 epoch linear warmup (1e-6 to 1e-4) |
| Cosine annealing | Smoother decay than step reduction | ReduceLROnPlateau -> CosineAnnealing |
| Lower initial LR | Current 1e-4 may be too high (model converges in 3 epochs) | 1e-4 -> 5e-5 or 3e-5 |
| Early stopping patience | With better regularization, model needs more epochs | 5 -> 10-15 |

### Regularization

| Parameter | Current | Planned |
|---|---|---|
| Dropout | 0.5 in FC layers only | Add 0.1-0.2 to conv layers (Spatial Dropout1D) |
| Weight decay | 1e-4 | Increase to 5e-4 or 1e-3 |
| BN momentum | 0.99 | 0.9 for faster adaptation |

### Architecture (if all else is not enough)

| Change | Rationale |
|---|---|
| Reduce model capacity | 2.5M params may be excessive; try halving filter counts |
| Add skip connections | To preserve input information (Nair et al., 2023) |
| Replace GeM with global average pooling | Simpler pooling may reduce overfitting |

### Implementation Order

**Step 1** (highest impact, lowest effort):
- Time shift + Gaussian noise augmentation
- LR warmup (5 epochs)
- Early stopping patience -> 10

**Step 2** (if overfitting persists):
- Mixup augmentation
- Label smoothing (0.1)
- Reduce LR to 5e-5, increase weight decay to 5e-4

**Step 3** (if still overfitting):
- Conv layer dropout
- Reduce model capacity
- Cosine annealing LR

Expected outcome from Step 1: training extends to 15-25 epochs, accuracy improves to 82-84%, train/val loss gap shrinks from 0.28 to <0.1.

---

## Phase 2: Architecture and Physics-Informed Training

Once training dynamics are healthy, the next step is embedding physical knowledge into both the architecture and the loss function. The current model learns entirely from data. The only physics it has is what's baked into the architecture (shared detector weights) and preprocessing. Changes are introduced one at a time, with ablation studies to measure individual impact. If a change degrades performance, it gets removed.

### Phase 2a: Auxiliary Per-Branch Losses

The current model has no supervision of individual detector representations, i.e. the classifier head only sees the concatenated features. Adding lightweight classification heads on each detector's 256-d output before aggregation gives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\text{combined}) + \lambda \sum_{i} \mathcal{L}_{\text{BCE}}(\text{detector}_i)$$

This forces each branch to maintain individually discriminative features, preventing any single detector from carrying the entire signal. Use a small λ (~0.1--0.3) since individual branches cannot reliably classify at low SNR (soft constraint).

### Phase 2b: GNN Aggregation Head

The current classifier concatenates the three detector feature vectors into a flat 768-d vector and passes it through FC layers. This is permutation-sensitive and treats cross-detector relationships as something the FC layers must discover implicitly.

The replacement: a small graph where each detector is a node (256-d features) and each edge encodes the physical relationship between that detector pair. Edge features are computed from the whitened signals before pooling:

- Cross-correlation peak value between the pair
- Peak lag (in samples)
- A windowed slice of the cross-correlation centered on the physically-allowed lag range

The allowed lag ranges are constants from known separations: LIGO-H/L ~10ms (~20 samples at 2048 Hz), LIGO-Virgo and Livingston-Virgo ~27ms (~55 samples). These bounds determine which lags are physically meaningful and define the window around which the cross-correlation slice is extracted.

With 3 nodes and 3 edges, we only need a hand-rolled message passing step (should be ~50 lines). This replaces the FC1 layer (768→256) in the classifier head.

### Phase 2c: Loss Constraints

Each constraint is added as an auxiliary loss term and evaluated individually:

**Cross-detector consistency**

Penalize predictions where confidence is driven disproportionately by a single detector's features, encouraging reliance on the cross-detector correlation that defines a real signal.

**Time-delay prior**

Penalize learned feature alignments that imply physically impossible arrival times, using the same lag bounds as Phase 2b.

**SNR-aware loss weighting**

Not all samples are equally informative. The lowest-SNR signals are genuinely undetectable. This is a physics limit, not a model failure (see [THE_SCIENCE.md](THE_SCIENCE.md), Detection Limits). Weight the loss by estimated SNR to focus training on learnable examples first, gradually including harder cases as training progresses.

---

## Phase 3: Waveform Structure Constraint

Compact binary mergers produce a chirp, defined by the frequency increasing over time as the objects spiral inward. A soft constraint could reward internal representations that correlate with chirp-like frequency evolution, without requiring exact waveform templates. This sits between pure data-driven learning and full matched filtering, and is the highest-complexity addition in the roadmap.

Only pursue Phase 3 if Phase 2 shows the model is still missing frequency structure that the current architecture cannot capture.

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