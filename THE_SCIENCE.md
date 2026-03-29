# (Some of) The Science Behind Gravitational Wave Detection

## Gravitational Wave Detectors

Gravitational waves are ripples in the geometry of spacetime, produced when massive objects accelerate violently. The strongest sources we can detect (merging black holes and neutron stars) radiate enormous energy as gravitational radiation during their final inspiral and coalescence. By the time these waves reach Earth, the spacetime distortion they produce is almost inconceivably small: a fractional change in length of roughly $10^{-21}$. For a 4 km baseline, that corresponds to a displacement of about $10^{-18}$ meters. This corresponds to a thousandth of the diameter of a proton.

Measuring this requires a Michelson interferometer. The concept is simple: split a laser beam into two perpendicular paths, send them down long arms, bounce them off mirrors at the far ends, and recombine them at a photodetector. When the arm lengths are precisely equal, the two returning beams interfere destructively, cancelling each other out and making it so no light reaches the detector. When a gravitational wave passes through, it stretches space along one arm while compressing it along the other. The path lengths change, the destructive interference becomes imperfect, and light appears at the photodetector.

That's the principle. Making it work at the $10^{-21}$ level is, to put it mildly, an engineering challenge. LIGO uses 4 km arms (Virgo uses 3 km), Fabry-Perot cavities that bounce the laser back and forth hundreds of times to effectively multiply the arm length, seismic isolation to decouple the mirrors from ground vibrations, and one of the most stable laser systems ever developed. The entire optical path sits inside one of the most perfect vacuum chambers ever built (by humans; interplanetary and intergalactic space are far more empty) to eliminate scattering from air molecules.

Three detectors operate simultaneously, spread across two continents:

- **LIGO Hanford (H1)** — Washington State, USA
- **LIGO Livingston (L1)** — Louisiana, USA
- **Virgo (V1)** — Cascina, Italy

The geographic separation is essential. A real gravitational wave, traveling at the speed of light, must appear in all three detectors with specific time delays determined by the wave's sky position. A local disturbance (could literally be a truck driving by a few miles away, or even a logger felling a tree) affects only one site. Requiring coincident signals across the detector network is one of the most powerful tools we have for rejecting false alarms. If you're interested, I suggest reading on the future LISA mission to build one of these in space rather than on the ground.

Each detector samples its output at 2048 Hz, producing 4096 data points over each 2-second observation window. It records a dimensionless quantity known as its *strain*: the fractional change in arm length, $h(t) = \Delta L / L$. 

## The Signal

A gravitational wave is described by two polarization "strain" fields, $h_+$ and $h_\times$, obeying:

$$\frac{\partial^2 h_+}{\partial t^2} = c^2 \nabla^2 h_+ \qquad \frac{\partial^2 h_\times}{\partial t^2} = c^2 \nabla^2 h_\times ,$$

where $c$ is the speed of light. This is just a wave equation representing the stretching and squeezing of spacetime along orthogonal axes. Each detector, depending on its location and orientation, measures its strain as a linear combination of the two:

$$h(t) := \frac{\Delta L}{L} = F_+(t)\,h_+(t) + F_\times(t)\,h_\times(t),$$

where $F_+$ and $F_\times$ are the beam pattern functions, determined by the detector's location, the source's sky position, and the polarization angle of the wave.

An asymmetry worth noting: gravitational wave signals are correlated across detectors (a wave arriving at Hanford at time $t$ reaches Livingston at $t + \Delta t$), but noise is not, depending rather on each detector's local apparatus and environment. Because the noise is uncorrelated, the joint probability of the data across detectors factorizes, and the log-likelihood ratio (signal present vs. noise only) reduces to a sum over individual detectors:

$$\log \Lambda[\mathbf{x}] = \sum_{i} \log \Lambda_i[x_i]$$

Each detector contributes independently to the total evidence for a signal. This is the fundamental distinction that the network must learn to exploit.

### On Noise

Each detector's output is the sum of signal and noise:

$$x(t) = h(t) + n(t),$$

where $n(t)$ has amplitude comparable to the signal — roughly $\sim10^{-21}$ — giving a signal-to-noise ratio close to 1. Feeding raw data directly into a neural network would be futile: we must *understand* the noise to properly mitigate its effects. Three dominant noise sources affect laser interferometers:

1. **Seismic noise** (low frequency) — ground vibrations from natural and human activity;
2. **Thermal noise** (mid frequency) — random molecular motion in mirror coatings and suspension fibres;
3. **Shot noise** (high frequency) — quantum fluctuations in photon arrival times at the photodetector.

Seismic and shot noise are handled by frequency selection. Gravitational wave signals from compact binary mergers lie within a narrow band: below 20 Hz, seismic noise dominates completely; above 500 Hz, shot noise takes over. A bandpass filter keeping only 20–500 Hz removes the frequencies where no signal is recoverable.

This still leaves the signal with the thermal noise, as well as the residual tails of seismic and shot noise. Two properties make this tractable. First, the noise is approximately **stationary** over short periods (the 2-second observation windows), meaning its statistical properties do not change between samples. Second, it is approximately **Gaussian** (a consequence of the Central Limit Theorem applied to many independent noise sources), which means it is fully characterized by its second-order statistics: the power at each frequency. The resulting noise is *colored*: its power varies dramatically across the spectrum.

These properties motivate computing the average **Power Spectral Density** (PSD):

$$S_n(f) = 2\,\mathbb{E}\left[|n(f)|^2\right]$$

estimated by averaging $|n(f)|^2$ over many noise-only samples (those labelled target=0). Using noise-only samples avoids biasing the estimate with signal power, which would inflate the apparent noise floor and reduce the effective SNR after whitening.

With the PSD in hand, whitening is straightforward:

1. Compute the Fourier Transform of the signal;
2. Divide by $\sqrt{S_n(f)}$;
3. Transform back to the time domain.

This flattens the noise spectrum: colored noise becomes approximately white, with equal power at every frequency. The noise is still present, but it no longer carries structure that a neural network could mistake for signal features. After whitening, SNR contributions are equalized across frequencies, and the network can focus on the actual waveform.

The cross-detector correlation then becomes the decisive feature. After whitening, the remaining noise in each detector is independent, but a real gravitational wave produces a correlated pattern across all three. The model architecture exploits this at multiple stages: the shared residual backbone extracts per-detector features, then a two-stage fusion topology (individual branch paths plus a joint branch that sees all three detectors concatenated) learns cross-detector interactions through dedicated residual blocks before the final classifier.

## How the Neural Network Detects Signals

After the preprocessing described above, the model receives three whitened, filtered, normalized time series (one per detector) each 4096 samples long. The task is binary classification: is there a gravitational wave in this data, or not?

The architecture is a 1D Convolutional Neural Network (CNN). If you've encountered CNNs in the context of image recognition, the idea here is the same, just in one dimension instead of two. A convolutional filter (a small kernel of learned weights) slides along the time axis, computing a weighted sum at each position. The result is a new sequence that responds strongly wherever the input matches the pattern the filter has learned. Stack several of these layers, and the network builds a hierarchy: early layers pick up simple oscillation patterns, while deeper layers combine those into more complex waveform structures.

```
Input (3 detectors x 4096 samples)
    |
    |---> Detector H1 -----|
    |                      +---> LIGO Extractor ----------|
    |---> Detector L1 -----|                              |
    |                                                     +---> Shared Residual Backbone (10 blocks)
    |---> Detector V1 ---------> Virgo Extractor ---------|           |
                                                                      |
                               +--------------------------------------+--------------------------------------+---------------------------+
                               |                                      |                                      |                           |
                          H1 features                            L1 features                           V1 features                       |
                               |                                      |                                      |                           |
                    LIGO branch (2 blocks) *                LIGO branch (2 blocks) *               Virgo branch (2 blocks)          Joint branch:
                               |                                      |                                      |                  concat all 3 -> 1x1 proj
                               |                                      |                                      |                     -> 2 ResBlocks
                               |                                      |                                      |                           |
                               |                                      |                                      |                           |
                               |                                      |                                      |                           |
                               +--------------------------------------+--------------------------------------+---------------------------+
                                                                      |
                                                     Concatenate 4 paths (256 channels)
                                                                      |
                                                          4 Fusion ResBlocks (256 ch)
                                                                      |
                                                              ConcatPool (512)
                                                                      |
                                                         3-layer classifier -> logits

                                                                                                                   * shared weights (same instrument)
```

A critical design choice is weight sharing. LIGO Hanford and Livingston are the same instrument design (4 km arms, similar noise characteristics), so their signals pass through the same extractor and branch blocks with shared parameters. Virgo is a different instrument (3 km arms, different seismic environment, different sensitivity curve) and gets its own extractor and branch blocks with independent weights. This enforces the correct $Z_2$ symmetry by construction: a gravitational wave, once whitened, produces a similar waveform shape in both LIGO detectors (up to arrival time offsets and amplitude differences from beam pattern functions), but Virgo's different characteristics warrant separate learned filters. After extraction, all three branches share the same residual backbone.

The architecture has two stages. First, a deep residual backbone: an extractor followed by 10 residual blocks (20 convolutional layers) with progressive downsampling and channel widening:

| Stage | Channels | Kernel Size | Downsample | Temporal dim |
|-------|----------|-------------|------------|-------------|
| Extractor | 1 -> 16 | 64 | GeM(2) | 4096 -> 2048 |
| Group 1 (2 blocks) | 16 | 31 | GeM(4) + identity | 2048 -> 512 |
| Group 2 (2 blocks) | 16 | 31 | identity | 512 |
| Group 3 (2 blocks) | 32 | 15 | GeM(4) + identity | 512 -> 128 |
| Group 4 (2 blocks) | 64 | 7 | GeM(4) + identity | 128 -> 32 |
| Group 5 (2 blocks) | 64 | 7 | identity | 32 |

The extractor uses a kernel of 64 samples, corresponding to $\sim31$ ms of data at 2048 Hz. This is deliberate. Gravitational wave chirps from binary mergers have structure on timescales of tens of milliseconds, and a large initial kernel lets the network capture these broad oscillation patterns directly. Subsequent groups use progressively smaller kernels (31, 15, 7) to refine the features, picking up finer temporal details from the patterns already extracted by earlier layers. Residual connections in every block provide direct gradient paths, allowing the network to be this deep without vanishing gradients. Stochastic depth randomly skips residual branches during training, regularizing the network by implicitly training an ensemble of shallower sub-networks and reducing co-adaptation between blocks.

Second, a two-stage fusion step. The backbone outputs 64-channel features at 32 time steps per detector. These feed into four parallel paths:

| Path | Input | Processing | Output |
|------|-------|-----------|--------|
| H1 individual | H1 backbone features (64 ch) | 2 ResBlocks(64, k=7) | 64 ch x 32 |
| L1 individual | L1 backbone features (64 ch) | 2 ResBlocks(64, k=7), shared with H1 | 64 ch x 32 |
| V1 individual | V1 backbone features (64 ch) | 2 ResBlocks(64, k=7), separate weights | 64 ch x 32 |
| Joint | concat(H1, L1, V1) = 192 ch | 1x1 Conv projection to 64 ch + 2 ResBlocks(64, k=7) | 64 ch x 32 |

The four path outputs are concatenated (4 x 64 = 256 channels) and processed through 4 fusion ResBlocks(256, k=7). This is the stage where cross-detector correlation becomes relevant. The individual branch paths refine per-detector features while the joint branch captures cross-detector interactions. The fusion blocks then learn from all paths simultaneously: a real gravitational wave produces correlated features across detectors, while noise does not.

After the fusion blocks, AdaptiveConcatPool1d concatenates adaptive average and max pooling, producing a 512-dimensional feature vector (256 from each pooling mode). The classifier head (a 3-layer MLP: 512 -> 256 -> 64 -> 1) maps this to a raw logit, which is passed through a sigmoid function at inference time to produce a probability between 0 and 1.

### Training

The loss function is binary cross-entropy (BCE):

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

where $y_i \in \{0,1\}$ is the true label and $\hat{y}_i$ is the model's predicted probability. This is the natural choice for binary classification: it is the negative log-likelihood of a Bernoulli distribution, so minimizing it is equivalent to maximum likelihood estimation. The logarithm means confident wrong answers are penalized far more heavily than uncertain ones. Predicting 0.99 when the true label is 0 incurs a much larger loss than predicting 0.6.

In addition to the main classifier loss, each detector branch produces its own auxiliary prediction through a small per-branch head. These auxiliary losses encourage each detector's convolutional features to be independently useful for classification, rather than relying on the other two branches to compensate. The total loss is a weighted sum:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + w_{\text{aux}} \cdot \frac{1}{3}\sum_{d=1}^{3} \mathcal{L}_{\text{branch},d}$$

where $w_{\text{aux}}$ controls how much the auxiliary task influences training. The intent is to improve gradient flow into the early convolutional layers via deep supervision. In practice, this did not measurably improve performance (see [developer-notes.md](developer-notes.md), Phase 2a), suggesting the bottleneck is in the feature extractor's representational capacity rather than in individual branch quality. The auxiliary heads remain useful as a diagnostic tool.

The optimizer is AdamW, which adds weight decay to the Adam framework. Weight decay adds a penalty proportional to the magnitude of the weights, gently pulling them toward zero each step. This discourages the network from fitting too closely to the training data by keeping the learned parameters small. Unlike classical L2 regularization, AdamW applies the decay directly to the weights rather than through the gradient, which interacts more cleanly with adaptive learning rate methods.

The learning rate follows a schedule with two phases. First, a linear warmup ramps the learning rate from near-zero up to the target value over the first few epochs, preventing the large random gradients of an untrained network from causing destructive early updates. After warmup, cosine annealing smoothly reduces the learning rate following a cosine curve from the target value down to a minimum, giving the optimizer progressively finer control over parameter updates as training proceeds.

Two forms of regularization combat overfitting. Dropout randomly zeroes a fraction of neuron activations during training, forcing the network to learn redundant representations rather than relying on any single feature pathway. Mixup takes pairs of training examples and creates synthetic samples by linearly interpolating both the inputs and their labels: $\tilde{x} = \lambda x_i + (1-\lambda) x_j$, $\tilde{y} = \lambda y_i + (1-\lambda) y_j$, where $\lambda$ is drawn from a Beta distribution. This smooths the decision boundary and reduces the model's tendency to memorize individual training examples.

Finally, early stopping monitors validation loss and halts training when it stops improving, restoring the model weights from the best-performing epoch. This prevents the network from training past the point of diminishing returns into pure overfitting.

### Geometric Structure of the Detector Network

The CNN architecture described above processes each detector independently and combines them by concatenation. This captures what each detector sees, but it does not explicitly encode a fundamental physical constraint: a real gravitational wave must arrive at the three detectors with specific, geometrically determined time delays. A gravitational wave source at sky position $\hat{n}(\theta, \varphi)$ reaches detectors $i$ and $j$ with a time delay

$$\tau_{ij} = \frac{(\mathbf{r}_i - \mathbf{r}_j) \cdot \hat{n}(\theta, \varphi)}{c}$$

where $\mathbf{r}_i$ and $\mathbf{r}_j$ are the detector positions on Earth's surface, $\hat{n}$ is the normal vector pointing towards the source's sky location (assumed to be the same for all 3 detectors), and $c$ is the speed of light. The crucial constraint is consistency: for a real signal, all three pairwise delays $\tau_{12}$, $\tau_{13}$, $\tau_{23}$ must correspond to a single point on the sky sphere $S^2$. For noise, cross-correlations at any combination of delays are uncorrelated. This geometric consistency is one of the strongest discriminators between signal and noise, and neither the CNN backbone nor the classifier head encodes it explicitly.

To exploit this structure, we construct a scalar field on $S^2$ that we call the **sky consistency map**. The idea is to ask, for every direction on the sky, how well the three detectors agree that a signal arrived from that direction. Given $N$ points on the sky, we compute the predicted time delays $\tau_{12}(k)$, $\tau_{13}(k)$, $\tau_{23}(k)$ for each sky pixel $k$ from the known detector coordinates. These are constants that depend only on the detector network geometry and are precomputed once. For each sample, we compute the full cross-correlation between each detector pair via FFT, evaluate it at the predicted delay for each sky pixel, and combine the three values into a single consistency score $C(k)$. The result is a function $C(\theta, \varphi)$ defined on the sphere: for noise, it is a flat random field; for a real signal, it (ideally) peaks at the true source position, where all three detector pairs show correlated signal at the geometrically predicted delays.

This sky consistency map is a function on $S^2$, and therefore admits a natural expansion in spherical harmonics:

$$C(\theta, \varphi) = \sum_{\ell=0}^{\ell_{\max}} \sum_{m=-\ell}^{\ell} a_{\ell m} \, Y_{\ell m}(\theta, \varphi)$$

Truncating at $\ell_{\max} \sim 12$ gives $(\ell_{\max}+1)^2 = 169$ real coefficients, giving a compact representation of the sky map's angular structure. The $\ell = 0$ coefficient is the average consistency across the entire sky. Higher multipoles encode progressively finer angular structure, with a strong signal producing power at multipoles corresponding to the angular resolution set by the detector baselines.

The detection problem has a natural symmetry: whether a gravitational wave is present does not depend on where on the sky it came from. But the *evidence* for a signal does depend on direction: it lives on $S^2$ and transforms under rotations. The spherical harmonic coefficients form what is called an equivariant representation: under an SO(3) rotation of the sky, each multipole $\ell$ transforms independently (via the Wigner $D$-matrices), and the full set of coefficients transforms in a known, structured way rather than being scrambled arbitrarily. The classifier head then collapses this structured representation into a single invariant output: the probability that a signal is present regardless of its origin. The aim is to process the data through intermediate representations that respect the symmetries of the domain, rather than hoping the network discovers those symmetries from the data alone.

The detector network also carries a discrete symmetry. LIGO Hanford and Livingston are the same instrument design, so swapping them should not change the detection outcome (the individual signals will differ in amplitude, but not in morphology). This $Z_2$ symmetry is enforced in the CNN backbone by weight sharing between the two LIGO branches (while Virgo, a different instrument, gets its own extractor). The sky consistency map inherits this symmetry automatically, since cross-correlation is symmetric under pair reordering.

In the full architecture (to be developed), the spherical harmonic coefficients are concatenated with the CNN backbone features before the classifier head. The two paths are complementary: the CNN captures signal morphology (what the waveform looks like in each detector), while the sky map captures geometric consistency (whether the detectors agree at delays that correspond to a single astrophysical source). A single detector might exhibit a chirp-like feature due to a noise artifact, but only the sky map can confirm that all three detectors see correlated signal with the correct relative timing for a real source at a specific sky position.

## But why a Neural Network?

The classical approach to gravitational wave detection is *matched filtering*: correlate the data with a template waveform and compare to a threshold. For Gaussian noise and a known waveform shape, matched filtering is optimal. It maximizes signal-to-noise ratio over all linear filters. So why replace it with a CNN?

1. **No templates needed.** Matched filtering requires knowing the waveform family. For binary classification (Signal/No Signal)) a CNN can learn to detect the presence of any correlated signal across detectors without exact waveform templates.

2. **Robust to non-Gaussian noise.** Real detector data has non-Gaussian tails: instrumental glitches, scattered light, environmental disturbances. The matched filter's optimality proof assumes Gaussianity. A CNN can learn to be robust to these artifacts.

3. **Nonlinear features.** Matched filtering is linear (it computes an inner product). A CNN with nonlinear activations can learn higher-order statistics that help distinguish signals from noise.

4. **Joint multi-detector processing.** While classical methods combine detectors through carefully derived likelihood functions, the CNN processes all three detector channels jointly and learns the optimal combination automatically, including any inter-detector correlations that might help.

## Detection Limits

Not all gravitational wave signals are detectable. The probability of detection depends on the signal-to-noise ratio. This is a fundamental physics constraint, not a model limitation. The optimal SNR is:

$$\rho^2 = 4 \int_0^\infty \frac{|\tilde{h}(f)|^2}{S_n(f)} \, df$$

This tells you that detectability depends on signal power *relative to noise power* at each frequency. A strong signal in a frequency band with even stronger noise is invisible; a weak signal in a quiet band may be detectable.

For our binary classifier:
- **False alarm** — predicting a signal when none exists (noise mimicking a signal pattern)
- **Missed detection** — predicting no signal when one exists (signal too quiet to distinguish from noise)

The ROC curve captures the tradeoff between these errors across all classification thresholds. The model will inevitably miss the lowest-SNR signals in the dataset. This is expected: even a theoretically optimal detector would miss them.

---

For visualizations of the data and preprocessing pipeline, see the [data exploration notebook](notebooks/01_data_exploration.ipynb). For model analysis and interpretability, see the [model explorer notebook](notebooks/02_model_explorer.ipynb). For the reasoning behind architectural decisions, see [developer-notes.md](developer-notes.md).
