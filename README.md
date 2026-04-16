# Gravitational Wave Detection with Neural Networks

A 1D Convolutional Neural Network for binary classification of gravitational wave signals in LIGO/Virgo detector noise, built from scratch.

This branch uses PyTorch. It builds on an earlier TensorFlow implementation preserved on the `tensorflow` branch.

## About This Project

I've taken some classes on Machine Learning, and done some academic projects, but I didn't *learn*. That was my goal here: I wanted to build something real, from scratch.

But why gravitational waves? I have a Physics degree, and did my thesis on Numerical Relativity. It was a modest, though complex project: modelling the Einstein Field Equations in a spherically symmetric system. I learned quite a lot about Relativity doing it, and became somewhat obsessed with the idea of teaching a machine to understand the code that I'd built. The initial goal was a Physics-Informed Neural Network to detect gravitational waves, but that turned out to be too big a step without prior experience building neural networks. So I started with a binary classifier, and introduce physics into the architecture as I go.

The current iteration is built around two ideas. First, the CNN backbone has to be deep: a chirp has hierarchical temporal structure (individual cycles, frequency evolution, amplitude envelope) and a shallow network cannot compose features across those timescales. Second, the meaningful geometric structure in this problem lives on the sky sphere $S^2$: a gravitational wave from a given direction arrives at each detector with specific time delays, and for a real signal all three pairwise delays must be consistent with a single point on the sky. Noise doesn't have this property. The model packages that consistency constraint as a scalar field on $S^2$, decomposes it into spherical harmonics, and uses the coefficients to modulate the CNN features through a FiLM layer before the classifier. The CNN captures what each detector sees; the sky map captures whether they agree in a way that is geometrically consistent with an astrophysical source.

If you're interested, [THE_SCIENCE.md](THE_SCIENCE.md) explains the detector physics, the preprocessing pipeline, and the full description of the network. Two Jupyter notebooks cover the dataset ([01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)) and let you poke at the model ([02_model_explorer.ipynb](notebooks/02_model_explorer.ipynb)). The record of experiments, what worked, and what did not is in [developer-notes.md](developer-notes.md). For a visual overview of the preprocessing and training pipeline, check out [PIPELINE.md](PIPELINE.md).

## Architecture

```
Input (3 detectors x 4096 samples)
    |
    |---> Detector H1 -----|
    |                      +---> LIGO Extractor (shared) ---|
    |---> Detector L1 -----|                                |
    |                                                       +---> Shared Residual Backbone (10 blocks)
    |---> Detector V1 ---------> Virgo Extractor -----------|           |
                                                                        |
                              +-----------------------------------------+-----------------------------------+-----------------------+
                              |                                         |                                   |                       |
                         H1 features                               L1 features                        V1 features                   |
                              |                                         |                                   |                       |
                    LIGO branch (2 blocks) *               LIGO branch (2 blocks) *            Virgo branch (2 blocks)         Joint branch:
                              |                                         |                                   |              concat all 3 -> 1x1 proj
                              |                                         |                                   |                 -> 2 ResBlocks
                              +-----------------------------------------+-----------------------------------+-----------------------+
                                                                        |
                                                    Concatenate 4 paths (16n = 256 ch)
                                                                        |
                                                         4 Fusion ResBlocks (256 ch)
                                                                        |
                                                        AdaptiveConcatPool -> 512 features
                                                                        |
                                                              +--------------------+       +--------- 121 SH coefs (l_max=10)
                                                              |     SkyFiLM        | <-----+       (S2 consistency map)
                                                              |  (1+gamma)*f+beta  |
                                                              +--------------------+
                                                                        |
                                                           3-layer classifier -> logits

                                                                                                                    * shared weights (same instrument)
```

V2 two-stage fusion: shared LIGO extractor + separate Virgo extractor feed a 10-block residual backbone. After the backbone, 4 parallel paths (H1, L1, V1 individual branches + a joint branch with a 1x1 projection) each run 2 residual blocks, are concatenated, and processed through 4 fusion blocks. AdaptiveConcatPool produces a 512-dim feature vector. S2 spherical harmonic coefficients (121 dimensions for l_max=10) are passed through a BatchNorm + 2-layer MLP that produces per-channel (gamma, beta); the pooled features become `(1 + gamma) * features + beta`. Both gamma and beta are tanh-bounded so (1+gamma) stays in (0, 2); the output projection is zero-initialized so the module starts as identity. Base channel width is n=16.

## Current Performance

| Accuracy | AUC | Precision | Recall | F1 |
|----------|-----|-----------|--------|----|
| 0.798 | 0.865 | 0.933 | 0.640 | 0.759 |

<p align="center"><img src="assets/dashboard.png" width="700"></p>

These numbers are the final run of the project: V2 fusion with the sky-feature FiLM layer on, 121 spherical-harmonic coefficients ($\ell_{\max} = 10$), manifold mixup, and stochastic weight averaging over the last stretch of training. The deep residual backbone lifted AUC from the 0.858 plateau to 0.866, V2 two-stage fusion reached 0.874 without sky features, and every run with sky features turned on has landed in the 0.865-0.871 band. The sky-consistency features never pushed past the no-sky baseline on this dataset: stripping sky-incompatible augmentations (input-space mixup, spectral dropout, time shift) cost more generalization than the physics-grounded features could recover, and every sky-compatible replacement stack traded one failure mode for another. The features stay in the code regardless, because the construction is a small contribution worth keeping on the record: a physical cross-detector consistency map projected onto the sky sphere $S^2$, decomposed in spherical harmonics, and used to modulate a learned 1D CNN through FiLM. Ground-based networks like G2Net are not where it pays off (three detectors, short baselines, already strong timing coincidence), but LISA is exactly where this kind of geometric conditioning could matter: three spacecraft arms, millions of kilometres apart, sources that dwell in band for months. The full development story is in [developer-notes.md](developer-notes.md).

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. 

**Dependencies**: PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm.

**Dataset**: [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection) on Kaggle. For local exploration, set `G2NET_DATASET_PATH` or place the dataset in `data/g2net-gravitational-wave-detection/`.

## Usage

### 1. Compute Average PSD (one-time)

```bash
python src/data/compute_psd.py
```

Computes the average noise Power Spectral Density from noise-only samples and saves it to `avg_psd.npz`. Required before creating tensor shards or training locally.

### 2. Train on Kaggle

Training runs on Kaggle's GPU. Preprocess the dataset into tensor shards locally, upload them alongside the source code as Kaggle datasets, and push the training kernel.

```bash
# one-time: preprocess into tensor shards
python src/data/create_tensors.py --input <path-to-dataset> --output <path-to-output>

# upload source code and preprocessed data
kaggle datasets version -p src -m "update" --dir-mode zip
kaggle datasets version -p <tensors-path> -m "update"

# run training and pull results
kaggle kernels push -p kaggle
kaggle kernels output zeteixeira/gw-training -p kaggle/output
```

For local training: `python src/model_runs.py`. You need the full dataset on disk and access to a GPU (or enough patience for a CPU run that will take days).

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── g2net.py              # Dataset loading
│   │   ├── preprocessing.py      # Signal preprocessing (whitening, filtering)
│   │   ├── compute_psd.py        # PSD computation (run once before training)
│   │   ├── create_tensors.py     # Tensor shard generation for Kaggle (includes SH coefficients)
│   │   └── download_data.py      # Dataset download helper
│   ├── models/
│   │   └── diy_model.py          # 1D CNN implementation (+ S2 sky features)
│   ├── sky_feasibility.py        # Sky map feasibility analysis (offline diagnostic)
│   ├── model_runs.py             # Training pipeline
│   └── visualization.py          # Plotting utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_explorer.ipynb
├── kaggle/
│   ├── train.py                  # Kaggle kernel entry point
│   └── kernel-metadata.json      # Kernel configuration
├── models/saved/                  # Trained model weights
└── requirements.txt
```

## Acknowledgments

This project uses data from the [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection) Kaggle competition, which provides simulated gravitational wave signals injected into real LIGO/Virgo detector noise.


