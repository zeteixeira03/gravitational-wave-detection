# Next Steps

This document tracks planned improvements across three phases: fixing training dynamics, embedding physical knowledge into the architecture and loss function, and (if warranted) adding waveform structure constraints.

---

## Phase 1: Optimization (concluded)

Phase 1 focused on fixing training dynamics and regularization. After exhausting all optimization-level improvements, the model plateaus at ~79% accuracy / AUC 0.858. The plateau is architectural, not an optimization issue. Phase 2 is the path forward.

### Results

Each change was tested on the full 560k-sample dataset (80/20 train/val split, Kaggle P100 GPU). Changes were cumulative, i.e.each run includes all previous improvements.

| Run | Date | Change introduced | AUC | Accuracy | Val loss |
|-----|------|-------------------|-----|----------|----------|
| 1 | Jan 17 | Baseline (3 epochs, LR 1e-3) | 0.768 | 0.549 | 3.538 |
| 2 | Jan 26 | LR 1e-4, dropout 0.5, 50 epochs | 0.854 | 0.784 | 0.564 |
| 3 | Feb 04 | Re-run | 0.854 | 0.785 | 0.593 |
| 4 | Feb 16 | + weight decay 1e-4 | 0.852 | 0.781 | 0.714 |
| 5 | Feb 16 | Re-run | 0.854 | 0.784 | 0.684 |
| 6 | Mar 02 | + mixup augmentation (alpha=0.2) | 0.859 | 0.791 | 0.447 |
| 7 | Mar 02 | Re-run | 0.856 | 0.786 | 0.472 |
| 8 | Mar 03 | + LR warmup (5 epochs) | 0.858 | 0.788 | 0.468 |
| 9 | Mar 03 | + LR 5e-5, weight decay 5e-4 | 0.858 | 0.788 | 0.459 |
| 10 | Mar 07 | + cosine annealing (T_0=10) | 0.858 | 0.787 | 0.461 |

Key observations:

- **Runs 2-5** (LR/dropout/weight decay tuning): AUC stuck at 0.852-0.854. The model was overfitting (train loss ~0.23, val loss ~0.69).
- **Run 6** (mixup): Best single improvement. AUC jumped to 0.859, and the train-val gap collapsed (0.40 vs 0.45), confirming overfitting was addressed. But val performance barely moved.
- **Runs 8-10** (warmup, LR reduction, cosine annealing): No measurable improvement. AUC flat at 0.858.

Conclusion: the model fits the training data cleanly but cannot extract more signal. We can now conclude that the remaining error is not from poor optimization or overfitting. Rather, it is (primarily) from the architecture's inability to model cross-detector correlations, which are the primary physical signature of a real gravitational wave.

### What was implemented

- Time shift augmentation (0-20 samples per detector)
- Gaussian noise injection (1-10% of signal std)
- Mixup ($\alpha=0.2$)
- LR warmup (5 epochs, linear from 1e-6 to target LR)
- Cosine annealing with warm restarts ($T_0$=10, $\eta_{min}$=1e-6)
- AdamW with weight decay 5e-4
- Early stopping (patience=10)

### What was not pursued

These remaining Phase 1 ideas were deprioritized because the plateau is architectural:

| Technique | Reason skipped |
|---|---|
| Label smoothing | Mixup already provides soft labels |
| Conv layer dropout | Overfitting is already solved |
| Reduced model capacity | Model is not overfitting; fewer params would hurt |
| Amplitude scaling augmentation | Unlikely to help given the architectural ceiling |
| Channel dropout augmentation | Conceptually better addressed by Phase 2a (per-branch losses) |
| Skip connections | Architectural change better suited to Phase 2 scope |

---

## Phase 2: Architecture and Physics-Informed Training (current)

Training dynamics are healthy (Phase 1 confirmed). The next step is embedding physical knowledge into both the architecture and the loss function. The current model learns entirely from data. The only physics it has is what's baked into the architecture (shared detector weights) and preprocessing. Changes are introduced one at a time, with ablation studies to measure individual impact. If a change degrades performance, it gets removed.

### Phase 2a: Auxiliary Per-Branch Losses

The current model has no supervision of individual detector representations, i.e. the classifier head only sees the concatenated features. Adding lightweight classification heads on each detector's 256-d output before aggregation gives:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{BCE}}(\text{combined}) + \lambda \sum_{i} \mathcal{L}_{\text{BCE}}(\text{detector}_i)$$

This forces each branch to maintain individually discriminative features, preventing any single detector from carrying the entire signal. $\lambda$ should be small (~0.1--0.3) since individual branches cannot reliably classify at low SNR (soft constraint).

**Result ($\lambda$=0.2):** AUC 0.857, accuracy 0.780 -- no improvement over baseline. The per-branch supervision did not unlock new capacity, confirming the bottleneck is not weak per-detector features but the fusion step that combines them. The auxiliary heads remain in the model as a diagnostic tool (individual branch AUCs can be checked after training) but do not contribute to performance.

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