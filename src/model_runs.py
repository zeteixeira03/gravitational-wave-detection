"""
DIY Model Training Pipeline

Complete training cycle for the DIY neural network model:
1. Load and precompute features
2. Initialize model with hyperparameters
3. Train with validation split
4. Evaluate performance
5. Save weights and metrics
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import find_dataset_dir, load_labels, load_sample
from data.features import compute_features
from models.diy_model import DIYModel
from visualization import (
    plot_learning_curves,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_prediction_distribution,
    plot_all_metrics
)

tf.random.set_seed(426425)

# =====================================================================
# SECTION 1: FEATURE COMPUTATION
# =====================================================================

def precompute_features(df, batch_size=1000, save_path=None):
    """
    Precompute features for all samples in batches.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'id' and 'target' columns
    batch_size : int
        Number of samples to process per batch
    save_path : Path, optional
        Path to save precomputed features

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Labels of shape (n_samples,)
    """
    ids = df["id"].astype(str).values
    targets = df["target"].values
    n_samples = len(df)

    print(f"Precomputing features for {n_samples} samples...")
    print(f"Processing in batches of {batch_size}")

    X_batches = []
    y_batches = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Batches"):
        batch_ids = ids[i:i+batch_size]
        batch_targets = targets[i:i+batch_size]

        X_batch = []
        y_batch = []

        for sample_id, target in zip(batch_ids, batch_targets):
            try:
                sample = load_sample(sample_id)
                feats = compute_features(sample)
                X_batch.append(feats)
                y_batch.append(target)
            except Exception as e:
                print(f"\nError processing {sample_id}: {e}")
                continue

        if X_batch:
            X_batches.append(np.stack(X_batch, axis=0))
            y_batches.append(np.array(y_batch))

    # Concatenate all batches
    X = np.vstack(X_batches).astype(np.float32)
    y = np.concatenate(y_batches).astype(np.int64)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Class balance: {y.sum()} positive / {len(y)} total ({100*y.mean():.2f}%)")

    if save_path:
        print(f"Saving features to: {save_path}")
        np.savez_compressed(save_path, X=X, y=y, ids=ids[:len(y)])
        print("Features saved.")

    return X, y


def load_precomputed_features(path):
    """Load precomputed features from disk."""
    print(f"Loading precomputed features from: {path}")
    data = np.load(path)
    X = data['X']
    y = data['y']
    print(f"Loaded X: {X.shape}, y: {y.shape}")
    return X, y


# =====================================================================
# SECTION 2: MODEL TRAINING
# =====================================================================

def train_diy_model(X, y, hyperparameters, test_size=0.2, random_state=42):
    """
    Train DIY model with specified hyperparameters.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    hyperparameters : dict
        Model hyperparameters (input_dim, hidden_dim, learning_rate, epochs)
    test_size : float
        Validation split ratio
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Training results including model, metrics, and split data
    """
    print("\n" + "="*60)
    print("INITIALIZING MODEL")
    print("="*60)

    # Extract hyperparameters
    input_dim = hyperparameters.get('input_dim', 30)
    hidden_dim = hyperparameters.get('hidden_dim', 64)
    hidden_dim2 = hyperparameters.get('hidden_dim2', 32)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    dropout_rate = hyperparameters.get('dropout_rate', 0.3)
    epochs = hyperparameters.get('epochs', 100)
    batch_size = hyperparameters.get('batch_size', 256)

    print(f"Input dimension: {input_dim}")
    print(f"Hidden layer 1: {hidden_dim}")
    print(f"Hidden layer 2: {hidden_dim2}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")

    # Split data
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} samples ({y_train.sum()} positive, {100*y_train.mean():.2f}%)")
    print(f"Val set: {X_val.shape[0]} samples ({y_val.sum()} positive, {100*y_val.mean():.2f}%)")

    # Standardize features (fit on train, transform both)
    print("\n" + "="*60)
    print("STANDARDIZING FEATURES")
    print("="*60)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print(f"Scaler fitted on training data")
    print(f"Train features: mean={X_train.mean():.4f}, std={X_train.std():.4f}")
    print(f"Val features:   mean={X_val.mean():.4f}, std={X_val.std():.4f}")

    # Initialize model
    model = DIYModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        hidden_dim2=hidden_dim2,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )

    # Train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    history = model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    print("\nTraining complete.")

    # Evaluate on both sets
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    # Training set metrics
    y_train_proba = model.predict_proba(X_train)
    y_train_pred = model.predict(X_train, threshold=0.5)
    train_metrics = model.evaluate(X_train, y_train)
    train_auc = roc_auc_score(y_train, y_train_proba)

    print(f"\nTraining Set:")
    print(f"  Accuracy:    {train_metrics['accuracy']:.4f}")
    print(f"  AUC:         {train_auc:.4f}")
    print(f"  Precision:   {train_metrics['precision']:.4f}")
    print(f"  Recall:      {train_metrics['recall']:.4f}")
    print(f"  Specificity: {train_metrics['specificity']:.4f}")

    # Validation set metrics
    y_val_proba = model.predict_proba(X_val)
    y_val_pred = model.predict(X_val, threshold=0.5)
    val_metrics = model.evaluate(X_val, y_val)
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"\nValidation Set:")
    print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"  AUC:         {val_auc:.4f}")
    print(f"  Precision:   {val_metrics['precision']:.4f}")
    print(f"  Recall:      {val_metrics['recall']:.4f}")
    print(f"  Specificity: {val_metrics['specificity']:.4f}")

    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'train_metrics': {
            'accuracy': float(train_metrics['accuracy']),
            'auc': float(train_auc),
            'precision': float(train_metrics['precision']),
            'recall': float(train_metrics['recall']),
            'specificity': float(train_metrics['specificity'])
        },
        'val_metrics': {
            'accuracy': float(val_metrics['accuracy']),
            'auc': float(val_auc),
            'precision': float(val_metrics['precision']),
            'recall': float(val_metrics['recall']),
            'specificity': float(val_metrics['specificity'])
        }
    }


# =====================================================================
# SECTION 3: SAVE RESULTS
# =====================================================================

def save_model_and_metrics(results, hyperparameters, save_dir):
    """
    Save model weights, hyperparameters, and metrics.

    Parameters
    ----------
    results : dict
        Training results from train_diy_model()
    hyperparameters : dict
        Model hyperparameters used
    save_dir : Path
        Directory to save model files

    Returns
    -------
    dict
        Paths to saved files
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"diy_{timestamp}"

    # Save weights
    weights_path = save_dir / f"{base_name}_weights.npz"
    results['model'].save_weights(str(weights_path))
    print(f"Weights saved to: {weights_path}")

    # Save scaler parameters
    scaler_path = save_dir / f"{base_name}_scaler.npz"
    np.savez(
        scaler_path,
        mean=results['scaler'].mean_,
        scale=results['scaler'].scale_
    )
    print(f"Scaler saved to: {scaler_path}")

    # Save hyperparameters
    config_path = save_dir / f"{base_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Save metrics
    metrics_data = {
        'timestamp': timestamp,
        'hyperparameters': hyperparameters,
        'train_metrics': results['train_metrics'],
        'val_metrics': results['val_metrics'],
        'final_train_loss': float(results['history']['train_loss'][-1]),
        'final_val_loss': float(results['history']['val_loss'][-1]),
    }

    metrics_path = save_dir / f"{base_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return {
        'weights': weights_path,
        'scaler': scaler_path,
        'config': config_path,
        'metrics': metrics_path,
        'base_name': base_name
    }


# =====================================================================
# SECTION 4: VISUALIZATION
# =====================================================================

def generate_plots(results, save_dir, base_name):
    """
    Generate and save all performance visualization plots.

    Parameters
    ----------
    results : dict
        Training results from train_diy_model()
    save_dir : Path
        Directory to save plot files
    base_name : str
        Base filename for saved plots

    Returns
    -------
    dict
        Paths to saved plot files
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    model = results['model']
    history = results['history']
    X_val = results['X_val']
    y_val = results['y_val']

    saved_plots = {}

    print("Generating plots...")

    # 1. Learning curves
    print("  - Learning curves")
    learning_path = plots_dir / f"{base_name}_learning_curves.png"
    plot_learning_curves(history, metrics=['loss', 'acc'], save_path=str(learning_path))
    saved_plots['learning_curves'] = learning_path

    # 2. ROC curve
    print("  - ROC curve")
    roc_data = model.roc_curve(X_val, y_val)
    roc_path = plots_dir / f"{base_name}_roc_curve.png"
    plot_roc_curve(roc_data, save_path=str(roc_path))
    saved_plots['roc_curve'] = roc_path

    # 3. Precision-Recall curve
    print("  - Precision-Recall curve")
    pr_data = model.precision_recall_curve(X_val, y_val)
    pr_path = plots_dir / f"{base_name}_pr_curve.png"
    plot_precision_recall_curve(pr_data, save_path=str(pr_path))
    saved_plots['pr_curve'] = pr_path

    # 4. Confusion matrix
    print("  - Confusion matrix")
    cm_data = model.confusion_matrix(X_val, y_val)
    cm_path = plots_dir / f"{base_name}_confusion_matrix.png"
    plot_confusion_matrix(cm_data, normalize=True, save_path=str(cm_path))
    saved_plots['confusion_matrix'] = cm_path

    # 5. Prediction distribution
    print("  - Prediction distribution")
    y_proba = model.predict_proba(X_val)
    dist_path = plots_dir / f"{base_name}_prediction_dist.png"
    plot_prediction_distribution(y_proba, y_val, save_path=str(dist_path))
    saved_plots['prediction_dist'] = dist_path

    # 6. Combined dashboard
    print("  - Combined dashboard")
    dashboard_path = plots_dir / f"{base_name}_dashboard.png"
    plot_all_metrics(model, X_val, y_val, history=history, save_path=str(dashboard_path))
    saved_plots['dashboard'] = dashboard_path

    print(f"Plots saved to: {plots_dir}")

    return saved_plots


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    """
    Main execution flow:
    1. Load/precompute features
    2. Initialize and train DIY model
    3. Evaluate performance and save weights/metrics
    4. Generate and save performance plots
    """

    # ========== HYPERPARAMETERS ==========
    HYPERPARAMETERS = {
        'input_dim': 39,         
        'hidden_dim': 64,        
        'hidden_dim2': 32,       
        'learning_rate': 0.001,  
        'dropout_rate': 0.3,     
        'epochs': 100,           
        'batch_size': 256,       
    }

    # ========== SETUP PATHS ==========
    print("="*60)
    print("SETUP")
    print("="*60)

    dataset_dir = find_dataset_dir()
    print(f"Dataset directory: {dataset_dir}")

    features_path = dataset_dir / "features_full.npz"
    models_dir = PROJECT_ROOT / "models" / "saved"

    print(f"Models directory: {models_dir}")

    # ========== STAGE 1: PRECOMPUTE FEATURES ==========
    if not features_path.exists():
        print("\n" + "="*60)
        print("STAGE 1: PRECOMPUTING FEATURES")
        print("="*60 + "\n")

        df = load_labels(dataset_dir)
        print(f"Total samples: {len(df)}")

        X, y = precompute_features(
            df,
            batch_size=1000,
            save_path=features_path
        )
    else:
        print("\n" + "="*60)
        print("STAGE 1: LOADING PRECOMPUTED FEATURES")
        print("="*60 + "\n")
        X, y = load_precomputed_features(features_path)

    # ========== STAGE 2: TRAIN MODEL ==========
    print("\n" + "="*60)
    print("STAGE 2: MODEL TRAINING")
    print("="*60)

    results = train_diy_model(
        X, y,
        hyperparameters=HYPERPARAMETERS,
        test_size=0.2,
        random_state=42
    )

    # ========== STAGE 3: SAVE RESULTS ==========
    print("\n" + "="*60)
    print("STAGE 3: SAVING RESULTS")
    print("="*60 + "\n")

    saved_paths = save_model_and_metrics(
        results,
        HYPERPARAMETERS,
        models_dir
    )

    # ========== STAGE 4: GENERATE PLOTS ==========
    print("\n" + "="*60)
    print("STAGE 4: GENERATING PLOTS")
    print("="*60 + "\n")

    saved_plots = generate_plots(
        results,
        models_dir,
        saved_paths['base_name']
    )

    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\nModel: {saved_paths['base_name']}")
    print(f"\nFinal Validation Metrics:")
    print(f"  Accuracy:    {results['val_metrics']['accuracy']:.4f}")
    print(f"  AUC:         {results['val_metrics']['auc']:.4f}")
    print(f"  Precision:   {results['val_metrics']['precision']:.4f}")
    print(f"  Recall:      {results['val_metrics']['recall']:.4f}")
    print(f"  Specificity: {results['val_metrics']['specificity']:.4f}")
    print(f"\nFiles saved:")
    print(f"  Weights: {saved_paths['weights'].name}")
    print(f"  Scaler:  {saved_paths['scaler'].name}")
    print(f"  Config:  {saved_paths['config'].name}")
    print(f"  Metrics: {saved_paths['metrics'].name}")
    print(f"\nPlots saved:")
    for name, path in saved_plots.items():
        print(f"  {name}: {path.name}")
    print(f"\nFeatures cached at: {features_path}")


if __name__ == "__main__":
    main()
