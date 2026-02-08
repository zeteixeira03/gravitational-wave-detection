# Gravitational Wave Detection with Neural Networks

A 1D Convolutional Neural Network for binary classification of gravitational wave signals in LIGO/Virgo detector noise, built from scratch in TensorFlow.

For the physics of gravitational wave detection and how the neural network works, see [THE_SCIENCE.md](THE_SCIENCE.md).

## Installation

```bash
pip install -r requirements.txt 
```

Requires Python 3.8+.

**Dependencies**: TensorFlow, NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm.

**Dataset**: Set the environment variable `G2NET_DATASET_PATH` to point to your local copy of the [G2Net dataset](https://www.kaggle.com/c/g2net-gravitational-wave-detection), or place it in `data/g2net-gravitational-wave-detection/`.

## Usage

### 1. Compute Average PSD (one-time)

```bash
python src/data/compute_psd.py
```

Computes the average noise Power Spectral Density from noise-only samples and saves it to `avg_psd.npz` in the dataset directory.

### 2. Train the Model

```bash
python src/model_runs.py
```

This will:
1. Load the precomputed PSD
2. Preprocess all signals (bandpass filter + whitening + normalization)
3. Cache preprocessed signals to `signals_whitened.npz` (first run only)
4. Train the 1D CNN with an 80/20 train/validation split
5. Save weights and metrics to `models/saved/`

For data exploration, see the notebooks in `notebooks/`.

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── g2net.py              # Dataset loading
│   │   ├── preprocessing.py      # Signal preprocessing (whitening, filtering)
│   │   ├── compute_psd.py        # PSD computation (run once before training)
│   │   ├── create_tfrecords.py   # TFRecord generation for Kaggle
│   │   └── download_data.py      # Dataset download helper
│   ├── models/
│   │   └── diy_model.py          # 1D CNN implementation
│   ├── model_runs.py             # Training pipeline
│   └── visualization.py          # Plotting utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_explorer.ipynb
├── models/saved/                  # Trained model weights
└── requirements.txt
```

## Acknowledgments

This project uses data from the [G2Net Gravitational Wave Detection](https://www.kaggle.com/c/g2net-gravitational-wave-detection) Kaggle competition, which provides simulated gravitational wave signals injected into real LIGO/Virgo detector noise.
