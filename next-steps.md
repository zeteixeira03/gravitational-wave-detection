# Next Steps

This document tracks planned improvements across three phases: fixing training dynamics, embedding physical knowledge into the architecture and loss function, and (if warranted) adding waveform structure constraints.

---

## Phase 1: Optimization

The model currently plateaus at ~79% accuracy. It's quite conservative (symmetric BCE + mixup soft labels discourage confident positive
predictions), resulting in high false-negative rates. Training dynamics have been fixed. Current priority is improving recall before adding complexity.

### Data Augmentation

The pipeline has some data augmentation, and this has produced good results in fixing early overfitting. As of now, time shift, Gaussian noise, and mixup have been implemented. The rest are saved in case they're needed further down the line.

| Technique | Rationale | Implementation |
|---|---|---|
| Amplitude scaling | GW amplitude varies with source distance; scaling preserves signal structure | Scale by factor 0.8-1.2 |
| Channel dropout | Zeroing one detector forces the model to not over-rely on any single channel | Set one of 3 channels to zero with p=0.1 |

### Training Dynamics

Early overfitting has been mostly solved by adding time shifts, Gaussian noise injection, and mixup augmentation. The current issue is that the model has been stuck at ~79%/AUC 0.858 despite the improvements. This suggests it's picking up a local minimum and struggling to escape it.

The current learning rate scheduling works in only one direction: once loss plateaus, it reduces and stays reduced. Cosine annealing is a technique which can be used to combat this problem. The learning rate will periodically increase to allow the model to escape the local minimum. The cosine cycle should start after warmup completes to avoid the two schedulers conflicting. If cosine annealing doesn't move the needle, the plateau is likely an architecture ceiling rather than an optimization issue, and Phase 2 is the answer.

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
| Replace GeM with global average pooling | Simpler pooling reduces model complexity |

### Implementation Order

**Step 1** (highest impact, lowest effort):
- Time shift + Gaussian noise augmentation (done)
- LR warmup (5 epochs) (done)
- Early stopping patience -> 10 (done)

**Step 2** (if plateau persists after Step 1):
- Mixup augmentation (done)
- Label smoothing (0.1)
- Reduce LR to 5e-5, increase weight decay to 5e-4 (done)

**Step 3** (if plateau persists after Step 2):
- Conv layer dropout
- Reduce model capacity

Expected outcome from Step 1: training extends to 15-25 epochs (confirmed), accuracy improves to ~80%, train/val loss gap shrinks from 0.28 to <0.1 (confirmed).

---

## Phase 2: Architecture and Physics-Informed Training

Once training dynamics are healthy, the next step is embedding physical knowledge into both the architecture and the loss function. The current model learns entirely from data. The only physics it has is what's baked into the architecture (shared detector weights) and preprocessing. Changes are introduced one at a time, with ablation studies to measure individual impact. If a change degrades performance, it gets removed.

### Phase 2a: Auxiliary Per-Branch Losses

The current model has no supervision of individual detector representations, i.e. the classifier head only sees the concatenated features. Adding lightweight classification heads on each detector's 256-d output before aggregation gives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\text{combined}) + \lambda \sum_{i} \mathcal{L}_{\text{BCE}}(\text{detector}_i)$$

This forces each branch to maintain individually discriminative features, preventing any single detector from carrying the entire signal. Use a small λ (~0.1--0.3) since individual branches cannot reliably classify at low SNR (soft constraint).

### Phase 2b: GNN Aggregation Head

The current classifier concatenates the three detector feature vectors into a flat 768-d vector and passes it through FC layers. This is permutation-sensitive: the model can in principle learn to treat detector 0 differently from detector 1 for no physical reason, since detector labelling is arbitrary. The physics is invariant to which detector is called H1, L1, or V1.

The replacement is a permutation-invariant aggregation via a small graph, where each detector is a node (256-d features) and messages are passed symmetrically between all pairs. Because message passing treats all nodes equivalently by construction, the aggregated representation is invariant to detector relabelling -- the symmetry group S₃ acting on the detector set is respected by design rather than left for the FC layers to discover from data.

Edge features encode the physical relationship between each detector pair, computed from the whitened signals before pooling:

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