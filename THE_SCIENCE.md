# (Some of) The Science Behind Gravitational Wave Detection

## Gravitational Wave Detectors

Gravitational waves are ripples in the geometry of spacetime, produced when massive objects accelerate violently. The strongest sources we can detect (merging black holes and neutron stars) radiate enormous energy as gravitational radiation during their final inspiral and coalescence. By the time these waves reach Earth, the spacetime distortion they produce is almost inconceivably small: a fractional change in length of roughly $10^{-21}$. For a 4 km baseline, that corresponds to a displacement of about $10^{-18}$ meters — a thousandth of the diameter of a proton.

Measuring this requires a Michelson interferometer. The concept is elegant: split a laser beam into two perpendicular paths, send them down long arms, bounce them off mirrors at the far ends, and recombine them at a photodetector. When the arm lengths are precisely equal, the two returning beams interfere destructively — they cancel each other out, and no light reaches the detector. When a gravitational wave passes through, it stretches space along one arm while compressing it along the other. The path lengths change, the destructive interference becomes imperfect, and light appears at the photodetector.

That's the principle. Making it work at the $10^{-21}$ level is, to put it mildly, an engineering challenge. LIGO uses 4 km arms (Virgo uses 3 km), Fabry-Perot cavities that bounce the laser back and forth hundreds of times to effectively multiply the arm length, seismic isolation to decouple the mirrors from ground vibrations, and one of the most stable laser systems ever built. The entire optical path sits inside one of the most perfect vacuum chambers ever built by humans to eliminate scattering from air molecules.

Three detectors operate simultaneously, spread across two continents:

- **LIGO Hanford (H1)** — Washington State, USA
- **LIGO Livingston (L1)** — Louisiana, USA
- **Virgo (V1)** — Cascina, Italy

The geographic separation is essential. A real gravitational wave, traveling at the speed of light, must appear in all three detectors with specific time delays determined by the wave's sky position. A local disturbance (could literally be a truck driving by a few miles away, or even a logger felling a tree) affects only one site. Requiring coincident signals across the detector network is one of the most powerful tools we have for rejecting false alarms. If you're interested, I suggest reading on the future LISA mission to build one of these in space rather than on the ground.

Each detector samples its output at 2048 Hz, producing 4096 data points over each 2-second observation window. What it records is the *strain*: the fractional change in arm length, $h(t) = \Delta L / L$. 

## The Signal

A gravitational wave is described by two polarization "strain" fields, $h_+$ and $h_\times$, obeying:

$$\frac{\partial^2 h_+}{\partial t^2} = c^2 \nabla^2 h_+ \qquad \frac{\partial^2 h_\times}{\partial t^2} = c^2 \nabla^2 h_\times ,$$

where $c$ is the speed of light. This is just a wave equation representing the stretching and squeezing of spacetime along orthogonal axes. Each detector, depending on its location and orientation, measures its strain as a linear combination of the two:

$$h(t) := \frac{\Delta L}{L} = F_+(t)\,h_+(t) + F_\times(t)\,h_\times(t),$$

where $F_+$ and $F_\times$ are the beam pattern functions, determined by the detector's location, the source's sky position, and the polarization angle of the wave.

A crucial asymmetry: gravitational wave signals are correlated across detectors (a wave arriving at Hanford at time $t$ reaches Livingston at $t + \Delta t$), but noise is not — it depends on each detector's local apparatus and environment. Because the noise is uncorrelated, the joint probability of the data across detectors factorizes, and the log-likelihood ratio (signal present vs. noise only) reduces to a sum over individual detectors:

$$\log \Lambda[\mathbf{x}] = \sum_{I} \log \Lambda_I[x_I]$$

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

$$S_n(f) = 2\,\mathbb{E}\!\left[|n(f)|^2\right]$$

estimated by averaging $|n(f)|^2$ over many noise-only samples (those labelled target=0). Using noise-only samples avoids biasing the estimate with signal power, which would inflate the apparent noise floor and reduce the effective SNR after whitening.

With the PSD in hand, whitening is straightforward:

1. Compute the Fourier Transform of the signal;
2. Divide by $\sqrt{S_n(f)}$;
3. Transform back to the time domain.

This flattens the noise spectrum: colored noise becomes approximately white, with equal power at every frequency. The noise is still present, but it no longer carries structure that a neural network could mistake for signal features. After whitening, SNR contributions are equalized across frequencies, and the network can focus on the actual waveform.

The cross-detector correlation then becomes the decisive feature. After whitening, the remaining noise in each detector is independent, but a real gravitational wave produces a correlated pattern across all three. The model architecture exploits this directly: shared convolutional layers process each detector identically, and the classifier head combines their features to detect precisely this correlation.

## How the Neural Network Detects Signals

After the preprocessing described above, the model receives three whitened, filtered, normalized time series (one per detector) each 4096 samples long. The task is binary classification: is there a gravitational wave in this data, or not?

The architecture is a 1D Convolutional Neural Network (CNN). If you've encountered CNNs in the context of image recognition, the idea here is the same, just in one dimension instead of two. A convolutional filter (a small kernel of learned weights) slides along the time axis, computing a weighted sum at each position. The result is a new sequence that responds strongly wherever the input matches the pattern the filter has learned. Stack several of these layers, and the network builds a hierarchy: early layers pick up simple oscillation patterns, while deeper layers combine those into more complex waveform structures.

```
Input (3 detectors x 4096 samples)
    |
    |---> Detector H1 ---\
    |---> Detector L1 ----+---> Shared Conv Layers ---> GeM Pool ---> 256 features each
    |---> Detector V1 ---/
                                                            |
                                                            v
                                                  Concatenate (768 features)
                                                            |
                                                            v
                                                  Dense (256) -> Dense (64) -> Dense (1) -> Sigmoid
```

A critical design choice is  sharing weights across detectors. All three signals pass through the exact same convolutional layers, with the exact same learned parameters. This is an implicit physical constraint in the architecture: a gravitational wave, once whitened, should produce a similar waveform shape in every detector (up to arrival time offsets and amplitude differences from detector orientation). The network learns what a gravitational wave looks like once, and applies that knowledge three times.

Each convolutional block follows the sequence: convolution, batch normalization, SiLU activation, max pooling. Four such blocks progressively reduce the 4096-sample input into a compact 256-dimensional feature vector per detector:

| Layer | Filters | Kernel Size | Pool Size |
|-------|---------|-------------|-----------|
| Conv1 | 32      | 64          | 4         |
| Conv2 | 64      | 32          | 4         |
| Conv3 | 128     | 16          | 4         |
| Conv4 | 256     | 8           | 4         |

The first layer uses a kernel of 64 samples , corresponding to $\sim31$ ms of data at 2048 Hz. This is deliberate: gravitational wave chirps from binary mergers have structure on timescales of tens of milliseconds, and a large initial kernel lets the network capture these broad oscillation patterns directly. Subsequent layers use progressively smaller kernels (32, 16, 8) to refine the features, picking up finer temporal details from the patterns already extracted by earlier layers.

After the convolutional stack, Generalized Mean (GeM) pooling compresses whatever temporal dimension remains into a single feature vector. GeM computes:

$$\mathbf{f} = \left(\frac{1}{T}\sum_{t=1}^{T} x_t^{\,p}\right)^{1/p}$$

where $p$ is a learnable parameter. When $p = 1$, this is average pooling; as $p \to \infty$, it approaches max pooling. The network learns whether to focus on the strongest activations (max-like) or the overall signal level (average-like).

With 256 features extracted from each of the three detectors, the vectors are concatenated into a single 768-dimensional representation. This is where the cross-detector correlation from the previous section becomes relevant. Up to this point, each detector was processed in isolation. Now, the classifier head learns to find patterns that span all three feature sets simultaneously. A real gravitational wave produces correlated features across detectors. Noise, being independent, does not. The final sigmoid activation maps the output to a probability between 0 and 1: the model's confidence that a gravitational wave is present.

Training uses binary cross-entropy loss, which is the natural choice for binary classification. It penalizes confident wrong answers more heavily than uncertain ones. AdamW optimization adds weight decay to the parameters, discouraging the network from fitting too closely to the training data. Randomly zeroing a share of neuron activations during training (Dropout) provides additional regularization by forcing the network to learn redundant representations rather than relying on any single feature pathway.

## Why a Neural Network?

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

For visualizations of the data and preprocessing pipeline, see the [data exploration notebook](notebooks/01_data_exploration.ipynb). For model analysis and interpretability, see the [model explorer notebook](notebooks/02_model_explorer.ipynb). For planned improvements, see [next-steps.md](next-steps.md).
