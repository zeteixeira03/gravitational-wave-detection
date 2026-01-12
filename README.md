# Gravitational Wave Detection with Neural Networks

Binary classification model to detect gravitational wave signals buried in detector noise. The challenge: real signals have extremely low signal-to-noise ratios, making them nearly indistinguishable from background noise to the human eye.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+.

**Dataset setup**: Set the environment variable `G2NET_DATASET_PATH` to point to your local copy of the dataset, or place it in `data/g2net-gravitational-wave-detection/`.

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
│   │   ├── features.py       # Feature engineering (39 features)
│   │   └── download_data.py  # Dataset download helper
│   ├── models/
│   │   ├── diy_model.py      # Custom neural network implementation
│   │   ├── cnn_model.py      # CNN-based approach
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

Run the training pipeline:

```bash
python src/model_runs.py
```

This will:
1. Load and precompute features (cached after first run)
2. Train the model with an 80/20 train/validation split
3. Save weights, scaler, and metrics to `models/saved/`
4. Generate performance plots

For data exploration, see the notebooks in `notebooks/`.

## Model Architecture

### Feature Engineering

The model uses 39 hand-crafted features extracted from each sample, designed to capture both signal strength and signal shape:

**Raw Amplitude Features (18 features)** - Computed before normalization to preserve signal strength information:
- Per-detector statistics: standard deviation, max absolute value, RMS (log-scaled)
- Cross-detector amplitude ratios (H1/L1, H1/V1, L1/V1)
- Total spectral power per detector
- Low-frequency vs high-frequency band power ratios

**Normalized Shape Features (21 features)** - Computed after normalization to capture signal morphology:
- Cross-detector correlations (gravitational waves should produce correlated patterns)
- Fractional power distribution across 5 frequency bands (20-64, 64-128, 128-256, 256-512, 512-1024 Hz)
- Peak frequency location per detector

### Neural Network

A simple feedforward network implemented in TensorFlow:

```
Input (39 features)
    │
    ▼
Dense (64 units) + ReLU + Dropout
    │
    ▼
Dense (32 units) + ReLU + Dropout
    │
    ▼
Dense (1 unit) + Sigmoid → Binary prediction
```

- **Initialization**: He initialization for ReLU layers
- **Regularization**: Dropout (default 0.3)
- **Loss**: Binary cross-entropy
- **Optimization**: Gradient descent with configurable learning rate

## Acknowledgments

This project uses data from the [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection) Kaggle competition, which provides simulated gravitational wave signals injected into real LIGO/Virgo detector noise.
