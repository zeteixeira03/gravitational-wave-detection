import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------
# Project / import setup
# ---------------------------------------------------------------------

# Resolve project root as the directory that contains src/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import find_dataset_dir, load_labels, load_sample
from data.features import compute_features


# ---------------------------------------------------------------------
# Feature-building function
# ---------------------------------------------------------------------

def build_feature_matrix(df): 
    """
    Given a DataFrame with columns ['id', 'target'], build:
      - X: feature matrix, shape (n_samples, n_features)
      - y: target vector,  shape (n_samples,)

    Features are computed with data.features.compute_features
    from the waveform loaded by data.g2net.load_sample.
    """
    # ids must be strings
    ids = df["id"].astype(str).values
    targets = df["target"].values

    X_list = []
    y_list = []

    n_samples = len(df)
    print(f"About to build feature matrix for {n_samples} samples...")

    for sample_id, target in tqdm(zip(ids, targets), total=n_samples):
        sample_id = str(sample_id).strip()

        # load sample, compute its features, and store it in feats
        sample = load_sample(sample_id)
        feats = compute_features(sample)

        X_list.append(feats)
        y_list.append(target)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    print(f"Finished building feature matrix X (shape = {X.shape}), and target vector y (shape = {y.shape})")
    return X, y


# ---------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------

def main():
    DATASET_DIR = find_dataset_dir()
    print("Using dataset from:", DATASET_DIR)

    df = load_labels(DATASET_DIR)
    print("Total labeled samples:", len(df))

    X_all, y_all = build_feature_matrix(df)

    out_path = DATASET_DIR / "features_logreg_full.npz"
    print("Saving features to:", out_path)
    np.savez(out_path, X=X_all, y=y_all)
    print("Done.")


if __name__ == "__main__":
    main()