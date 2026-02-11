# Next Steps

This document tracks planned improvements in two phases: fixing the model's training dynamics, then embedding physical knowledge into the training process.

---

## Phase 1: Optimization

The model currently plateaus at 78.5% accuracy with signs of early overfitting. The priority is to fix training dynamics before adding complexity.

### The Problem

1. Validation loss bottoms at epoch 3: abnormally fast convergence, suggesting the model finds local patterns but fails to generalize
2. Training loss continues dropping (0.44 -> 0.31) after validation diverges: the model has capacity to learn more, but memorizes instead of generalizing
3. No data augmentation: the model sees identical samples each epoch, making memorization trivial

The core issue: the model learns fast (high learning rate + no augmentation = quick memorization), not well.

### Data Augmentation

The pipeline has zero data augmentation. This is the single biggest lever for improvement.

| Technique | Rationale | Implementation |
|---|---|---|
| Time shift | GW signals arrive at different times at each detector (up to ~10ms offset); shifting is label-preserving | Random offset of 0-20 samples per detector |
| Gaussian noise | Detector noise varies between observations; small noise injection builds robustness | Add noise with std = 0.01-0.1 of signal std |
| Amplitude scaling | GW amplitude varies with source distance; scaling preserves signal structure | Scale by factor 0.8-1.2 |
| Mixup | Interpolates training pairs to create virtual samples; proven for time-series (Farhadi et al., 2023) | x\_new = lambda\*x1 + (1-lambda)\*x2, lambda ~ Beta(0.2, 0.2) |
| Channel dropout | Zeroing one detector forces the model to not over-rely on any single channel | Set one of 3 channels to zero with p=0.1 |
| SpecAugment masking | Masks time/frequency regions; forces distributed feature learning (Owusu et al., 2025) | Random rectangular masks in time domain |

Priority: time shift + Gaussian noise first, then Mixup.

### Training Dynamics

| Change | Rationale | Current -> Suggested |
|---|---|---|
| LR warmup | Prevents early aggressive updates pushing into poor local minima | None -> 3-5 epoch linear warmup (1e-6 to 1e-4) |
| Cosine annealing | Smoother decay than step reduction | ReduceLROnPlateau -> CosineAnnealing |
| Lower initial LR | Current 1e-4 may be too high (model converges in 3 epochs) | 1e-4 -> 5e-5 or 3e-5 |
| Early stopping patience | With better regularization, model needs more epochs | 5 -> 10-15 |

### Regularization

| Parameter | Current | Suggested |
|---|---|---|
| Dropout | 0.5 in FC layers only | Add 0.1-0.2 to conv layers (Spatial Dropout1D) |
| Weight decay | 1e-4 | Increase to 5e-4 or 1e-3 |
| Label smoothing | None | 0.1 -- prevents overconfident predictions, improves calibration (Yao et al., 2022) |
| BN momentum | 0.99 | 0.9 for faster adaptation |

### Architecture (if above insufficient)

| Change | Rationale |
|---|---|
| Reduce model capacity | 2.5M params may be excessive; try halving filter counts |
| Add skip connections | Competition winners used these to preserve input information (Nair et al., 2023) |
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

## Phase 2: Physics-Informed Training

Once training dynamics are healthy, the next step is embedding physical knowledge directly into the loss function. The current model learns entirely from data -- the only physics it has is what's baked into the architecture (shared detector weights) and preprocessing (whitening, bandpass filtering). Adding constraints could help the model learn more efficiently, or it could conflict with patterns the data-driven model has already found. The point is to find out.

### Potential Constraints

**Cross-detector consistency**

A real gravitational wave must appear in multiple detectors. A regularization term could penalize predictions where the model's confidence is driven disproportionately by features from a single detector, encouraging it to rely on the cross-detector correlation that defines a real signal.

**Time-delay prior**

Gravitational waves travel at the speed of light. The maximum delay between any detector pair is bounded by their physical separation (~10ms for Hanford-Livingston, ~27ms for LIGO-Virgo). A constraint could penalize learned feature alignments that imply physically impossible arrival times.

**Waveform structure**

Compact binary mergers produce a chirp -- frequency increasing over time as the objects spiral inward. A soft constraint could reward internal representations that correlate with chirp-like frequency evolution, without requiring exact waveform templates. This sits between pure data-driven learning and full matched filtering.

**SNR-aware loss weighting**

Not all samples are equally informative. The lowest-SNR signals are genuinely undetectable -- a physics limit, not a model failure (see [THE_SCIENCE.md](THE_SCIENCE.md), Detection Limits). Weighting the loss by estimated SNR could help the model focus on learnable examples first, gradually including harder cases as training progresses.

### Approach

Each constraint will be added as an auxiliary loss term:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{consistency}} + \lambda_2 \mathcal{L}_{\text{delay}} + \ldots$$

The lambda weights control how strongly each constraint influences training. They will be introduced one at a time, with ablation studies to measure individual impact. If a constraint degrades performance, it gets removed.



### Please Help Out!

If you read my code and found a bug, or a possible improvement that I didn't think of, please contact me so we can discuss this further. I'd love to hear about it!

---

### Sources

- Farhadi et al. (2023) -- Mixup for acoustic signal detection
- Sun et al. (2024) -- Data imbalance and training strategies for wave detection
- Yao et al. (2022) -- Label smoothing for reducing overfitting
- Owusu et al. (2025) -- SpecAugment-style masking for signal classification
- Ta et al. (2023) -- Dropout and Gaussian noise for 1D CNNs
- Nair et al. (2023) -- Skip connections in GW detection architectures
- Sacco et al. (2022) -- Cosine annealing with warm restarts
 