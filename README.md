# Gravitational Wave Detection with Neural Networks

Binary classification model to detect gravitational wave signals buried in detector noise. The challenge: real signals have extremely low signal-to-noise ratios, making them nearly indistinguishable from background noise to the human eye.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

**Dataset setup**: Set the environment variable `G2NET_DATASET_PATH` to point to your local copy of the dataset, or place it in `data/g2net-gravitational-wave-detection/`.

**Dependencies**: TensorFlow, NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm.

## The Problem

### The Detectors

Gravitational waves are detected using laser interferometers - L-shaped instruments that measure tiny changes in the distance between mirrors. This project uses data from three detectors:

- **LIGO Hanford (H1)** - Washington, USA
- **LIGO Livingston (L1)** - Louisiana, USA
- **Virgo (V1)** - Cascina, Italy

When a gravitational wave passes through Earth, it stretches space in one direction and compresses it in the perpendicular direction, causing measurable (but tiny) changes in the interferometer arm lengths.

### The Data

Each sample consists of:
- **3 time series** (one per detector)
- **2 seconds** of data at **2048 Hz** sampling rate = **4096 data points** per detector
- Values represent **strain**: h(t) = ΔL/L (fractional change in arm length)
- Typical values are on the order of **10⁻²¹** - extraordinarily small

### The Challenge

The dataset is perfectly balanced (50% signal, 50% noise-only), but the difficulty lies in the signal-to-noise ratio. Gravitational wave signals from binary black hole mergers are buried deep within detector noise. Visual inspection of the raw waveforms rarely reveals the presence of a signal.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── g2net.py          # Dataset loading utilities
│   │   ├── preprocessing.py  # Signal preprocessing (whitening, filtering)
│   │   ├── compute_psd.py    # PSD computation script (run once before training)
│   │   ├── features.py       # Feature engineering (39 features, for experimentation)
│   │   └── download_data.py  # Dataset download helper
│   ├── models/
│   │   ├── diy_model.py      # 1D CNN implementation
│   │   └── base_model.py     # Base model utilities
│   ├── model_runs.py         # Main training pipeline
│   └── visualization.py      # Plotting utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Data visualization and analysis
│   └── 02_baseline_model.ipynb    # Baseline model experiments
├── models/saved/             # Trained model weights and metrics
└── requirements.txt
```

## Usage

### Step 1: Compute Average PSD (one-time setup)

Before training, compute the average noise Power Spectral Density from noise-only samples:

```bash
python src/data/compute_psd.py
```

This computes the average noise spectrum used for whitening and saves it to `avg_psd.npz` in the dataset directory. Only needs to be run once.

### Step 2: Train the Model

```bash
python src/model_runs.py
```

This will:
1. Load the precomputed PSD
2. Preprocess all signals (bandpass filter + whitening + normalization)
3. Cache preprocessed signals to `signals_whitened.npz` (first run only)
4. Train the 1D CNN with an 80/20 train/validation split
5. Save weights and metrics to `models/saved/`
6. Generate performance plots

For data exploration, see the notebooks in `notebooks/`.

## Model Architecture

### Signal Preprocessing

The preprocessing pipeline transforms raw detector signals into a form where GW signals become detectable:

1. **Bandpass Filter (20-500 Hz)**: Removes frequencies outside the detector's sensitive range
   - Below 20 Hz: seismic noise dominates
   - Above 500 Hz: shot noise dominates, fewer GW signals

2. **Spectral Whitening**: The key technique from successful Kaggle solutions
   - Computes average noise PSD from noise-only samples
   - Divides signal spectrum by noise spectrum, flattening the noise to be uniform
   - Makes the GW signal's characteristic chirp pattern visible

3. **Tukey Window**: Reduces edge artifacts from filtering

4. **Normalization**: Zero mean, unit variance per detector

### 1D Convolutional Neural Network

A custom TensorFlow implementation that processes whitened signals directly:

```
Input (3 detectors x 4096 samples)
    │
    ├──► Detector H1 ──┐
    ├──► Detector L1 ──┼──► Shared Conv Layers ──► GeM Pool ──► 256 features each
    └──► Detector V1 ──┘
                                                        │
                                                        ▼
                                              Concatenate (768 features)
                                                        │
                                                        ▼
                                              Dense (256) + BatchNorm + SiLU + Dropout
                                                        │
                                                        ▼
                                              Dense (64) + BatchNorm + SiLU + Dropout
                                                        │
                                                        ▼
                                              Dense (1) + Sigmoid → Binary prediction
```

**Convolutional Layers** (shared across detectors):
| Layer | Filters | Kernel Size | Pool Size |
|-------|---------|-------------|-----------|
| Conv1 | 32      | 64          | 4         |
| Conv2 | 64      | 32          | 4         |
| Conv3 | 128     | 16          | 4         |
| Conv4 | 256     | 8           | 4         |

Each conv block: Conv1D → BatchNorm → SiLU activation → MaxPool

**Key Components**:
- **Shared weights**: All three detectors use the same conv layers, learning detector-agnostic features
- **GeM Pooling**: Generalized Mean Pooling (learnable parameter) for global feature aggregation
- **SiLU activation**: Smooth approximation of ReLU (x * sigmoid(x))
- **BatchNorm**: Applied after each conv and dense layer for training stability

**Training**:
- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (default lr=0.001)
- **Regularization**: Dropout (default 0.3)
- **Batch size**: 32 (smaller due to CNN memory requirements)

## Acknowledgments

This project uses data from the [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection) Kaggle competition, which provides simulated gravitational wave signals injected into real LIGO/Virgo detector noise.
