"""
1D CNN Model Training Pipeline

Training cycle using preprocessed TFRecords:
1. Load TFRecords (preprocessed signals, must run compute_psd.py and create_tfrecords before - in this order)
2. Initialize 1D CNN model
3. Train with early stopping and LR scheduling
4. Evaluate and save metrics
5. Generate evaluation plots
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import tensorflow as tf

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import is_kaggle, get_output_dir
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
# SECTION 1: TRAINING LOOP
# =====================================================================

def fit(
    model,
    train_dataset,
    val_dataset,
    n_train,
    n_val,
    epochs,
    batch_size,
    verbose=True,
    early_stopping_patience=5,
    lr_reduce_patience=3,
    lr_reduce_factor=0.5,
    min_lr=1e-6
):
    """
    Train model using tf.data.Dataset generators.

    Parameters
    ----------
    model : DIYModel
        Model to train
    train_dataset : tf.data.Dataset
        Training dataset (streaming)
    val_dataset : tf.data.Dataset
        Validation dataset (streaming)
    n_train : int
        Number of training samples
    n_val : int
        Number of validation samples
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size (for progress calculation)
    verbose : bool
        Whether to print progress
    early_stopping_patience : int
        Stop training if val_loss doesn't improve for this many epochs
    lr_reduce_patience : int
        Reduce LR if val_loss doesn't improve for this many epochs
    lr_reduce_factor : float
        Factor to multiply LR by when reducing
    min_lr : float
        Minimum learning rate (won't reduce below this)

    Returns
    -------
    dict
        Training history
    """
    n_batches = (n_train + batch_size - 1) // batch_size
    n_val_batches = (n_val + batch_size - 1) // batch_size

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    # early stopping and LR scheduling
    best_val_loss = float('inf')
    best_weights = None
    best_bn_state = None
    epochs_without_improvement = 0
    epochs_since_lr_reduce = 0

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(enumerate(train_dataset), total=n_batches, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)

        for batch_num, (X_batch, y_batch) in pbar:
            y_batch = tf.cast(tf.reshape(y_batch, (-1, 1)), tf.float32)
            loss = model.train_step(X_batch, y_batch)
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")

        train_loss = np.mean(epoch_losses)
        history['train_loss'].append(train_loss)

        # validation loss
        val_losses = []
        val_correct = 0
        val_total = 0
        for X_batch, y_batch in val_dataset:
            y_batch_float = tf.cast(tf.reshape(y_batch, (-1, 1)), tf.float32)
            pred = model.forward(X_batch, training=False)
            loss = model.compute_loss(y_batch_float, pred).numpy()
            val_losses.append(loss)

            # accuracy on this batch
            pred_labels = (pred.numpy() >= 0.5).astype(int).flatten()
            val_correct += (pred_labels == y_batch.numpy()).sum()
            val_total += len(y_batch)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(0.0)  # skip train acc for speed - train loss is a good enough metric without doing a full forward pass

        # check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save trainable weights
            best_weights = {name: var.numpy().copy() for name, var in zip(
                [v.name for v in model.variables], model.variables
            )}
            # save batch norm moving averages (critical for inference)
            best_bn_state = {
                'conv_bn': [(m.numpy().copy(), v.numpy().copy())
                            for _, _, m, v in model.bn_layers],
                'fc1_bn': (model.fc1_bn[2].numpy().copy(), model.fc1_bn[3].numpy().copy()),
                'fc2_bn': (model.fc2_bn[2].numpy().copy(), model.fc2_bn[3].numpy().copy()),
            }
            epochs_without_improvement = 0
            epochs_since_lr_reduce = 0
            improvement_marker = " *"
        else:
            epochs_without_improvement += 1
            epochs_since_lr_reduce += 1
            improvement_marker = ""

        # learning rate reduction
        current_lr = float(model.optimizer.learning_rate.numpy())
        if epochs_since_lr_reduce >= lr_reduce_patience and current_lr > min_lr:
            new_lr = max(current_lr * lr_reduce_factor, min_lr)
            model.optimizer.learning_rate.assign(new_lr)
            epochs_since_lr_reduce = 0
            if verbose:
                print(f"  Reducing learning rate: {current_lr:.2e} -> {new_lr:.2e}")

        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}{improvement_marker}")

        # early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping: val_loss hasn't improved for {early_stopping_patience} epochs")
            break

    # restore best weights and batch norm state
    if best_weights is not None:
        if verbose:
            print(f"Restoring best weights (val_loss: {best_val_loss:.4f})")
        for var, name in zip(model.variables, best_weights.keys()):
            var.assign(best_weights[name])

        # restore batch norm moving averages
        for i, (mean, var) in enumerate(best_bn_state['conv_bn']):
            model.bn_layers[i][2].assign(mean)
            model.bn_layers[i][3].assign(var)
        model.fc1_bn[2].assign(best_bn_state['fc1_bn'][0])
        model.fc1_bn[3].assign(best_bn_state['fc1_bn'][1])
        model.fc2_bn[2].assign(best_bn_state['fc2_bn'][0])
        model.fc2_bn[3].assign(best_bn_state['fc2_bn'][1])

    return history


def evaluate(model, dataset):
    """Evaluate model on dataset, return predictions and labels."""
    y_true_all = []
    y_proba_all = []

    for X_batch, y_batch in dataset:
        pred = model.predict_proba(X_batch.numpy())
        y_proba_all.append(pred)
        y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all)
    y_proba = np.concatenate(y_proba_all)

    return y_true, y_proba


# =====================================================================
# SECTION 2: SAVE RESULTS
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

    # Save hyperparameters
    config_path = save_dir / f"{base_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Save metrics
    metrics_data = {
        'timestamp': timestamp,
        'hyperparameters': hyperparameters,
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
        'config': config_path,
        'metrics': metrics_path,
        'base_name': base_name
    }


# =====================================================================
# SECTION 3: VISUALIZATION
# =====================================================================

def generate_plots(results, save_dir, base_name, max_plot_samples=10000):
    """
    Generate evaluation plots after training.

    Loads a subset of validation data from TFRecords for plotting.

    Parameters
    ----------
    results : dict
        Training results from train_from_tfrecords()
    save_dir : Path
        Directory to save plot files
    base_name : str
        Base filename for saved plots
    max_plot_samples : int
        Maximum samples to load for plotting

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
    tfrecord_path = results['tfrecord_path']
    n_val = results['n_val']

    # load validation subset for plotting
    n_plot = min(max_plot_samples, n_val)
    print(f"Loading {n_plot} validation samples for plotting...")

    full_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    full_dataset = full_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    plot_dataset = full_dataset.take(n_plot).batch(256).prefetch(tf.data.AUTOTUNE)

    # collect data
    X_list = []
    y_list = []
    for X_batch, y_batch in plot_dataset:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())

    X_val = np.concatenate(X_list, axis=0)
    plot_y = np.concatenate(y_list, axis=0)

    saved_plots = {}

    print("Generating plots...")

    # 1. Learning curves
    print("  - Learning curves")
    learning_path = plots_dir / f"{base_name}_learning_curves.png"
    plot_learning_curves(history, metrics=['loss', 'acc'], save_path=str(learning_path))
    saved_plots['learning_curves'] = learning_path

    # 2. ROC curve
    print("  - ROC curve")
    roc_data = model.roc_curve(X_val, plot_y)
    roc_path = plots_dir / f"{base_name}_roc_curve.png"
    plot_roc_curve(roc_data, save_path=str(roc_path))
    saved_plots['roc_curve'] = roc_path

    # 3. Precision-Recall curve
    print("  - Precision-Recall curve")
    pr_data = model.precision_recall_curve(X_val, plot_y)
    pr_path = plots_dir / f"{base_name}_pr_curve.png"
    plot_precision_recall_curve(pr_data, save_path=str(pr_path))
    saved_plots['pr_curve'] = pr_path

    # 4. Confusion matrix
    print("  - Confusion matrix")
    cm_data = model.confusion_matrix(X_val, plot_y)
    cm_path = plots_dir / f"{base_name}_confusion_matrix.png"
    plot_confusion_matrix(cm_data, normalize=True, save_path=str(cm_path))
    saved_plots['confusion_matrix'] = cm_path

    # 5. Prediction distribution
    print("  - Prediction distribution")
    y_proba = model.predict_proba(X_val)
    dist_path = plots_dir / f"{base_name}_prediction_dist.png"
    plot_prediction_distribution(y_proba, plot_y, save_path=str(dist_path))
    saved_plots['prediction_dist'] = dist_path

    # 6. Combined dashboard
    print("  - Combined dashboard")
    dashboard_path = plots_dir / f"{base_name}_dashboard.png"
    plot_all_metrics(model, X_val, plot_y, history=history, save_path=str(dashboard_path))
    saved_plots['dashboard'] = dashboard_path

    print(f"Plots saved to: {plots_dir}")

    return saved_plots


# =====================================================================
# SECTION 4: TFRECORD LOADING
# =====================================================================

def parse_tfrecord(example_proto):
    """Parse a single TFRecord example."""
    feature_spec = {
        'signal': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_spec)

    signal = tf.io.decode_raw(parsed['signal'], tf.float32)
    signal = tf.reshape(signal, [3, 4096])
    label = parsed['label']

    return signal, label


def create_tfrecord_dataset(tfrecord_path, batch_size=128, shuffle=True, buffer_size=50000):
    """
    Create a tf.data.Dataset from a TFRecord file.

    Parameters
    ----------
    tfrecord_path : str or Path
        Path to the TFRecord file
    batch_size : int
        Batch size
    shuffle : bool
        Whether to shuffle the data
    buffer_size : int
        Shuffle buffer size

    Returns
    -------
    tf.data.Dataset
        Dataset yielding (signal, label) batches
    """
    dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def train_from_tfrecords(
    tfrecord_path,
    n_samples,
    hyperparameters,
    val_split=0.2
):
    """
    Train DIY 1D CNN model from preprocessed TFRecords.

    Parameters
    ----------
    tfrecord_path : Path
        Path to TFRecord file
    n_samples : int
        Total number of samples in the TFRecord
    hyperparameters : dict
        Model hyperparameters
    val_split : float
        Fraction of data to use for validation

    Returns
    -------
    dict
        Training results
    """
    print("\n" + "="*60)
    print("INITIALIZING 1D CNN MODEL (TFRECORD MODE)")
    print("="*60)

    n_samples_config = hyperparameters.get('n_samples', 4096)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    dropout_rate = hyperparameters.get('dropout_rate', 0.3)
    epochs = hyperparameters.get('epochs', 15)
    batch_size = hyperparameters.get('batch_size', 128)

    print(f"Signal length: {n_samples_config}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total samples: {n_samples}")
    print("Mode: TFRECORD (preprocessed data, fast loading)")

    # split into train/val
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    print(f"\nTrain samples: {n_train}")
    print(f"Val samples: {n_val}")

    # create datasets
    # for validation, we take the first n_val samples (no shuffle on first pass)
    full_dataset = tf.data.TFRecordDataset(str(tfrecord_path))
    full_dataset = full_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = full_dataset.take(n_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_dataset = full_dataset.skip(n_val).shuffle(50000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # initialize model
    model = DIYModel(
        n_samples=n_samples_config,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate
    )

    # train
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    history = fit(
        model, train_dataset, val_dataset,
        n_train, n_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    print("\nTraining complete.")

    # evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    # recreate val dataset for evaluation
    val_dataset_eval = full_dataset.take(n_val).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    y_val, y_val_proba = evaluate(model, val_dataset_eval)
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, y_val_proba)
    cm = model._compute_confusion_values(y_val_pred, y_val)
    val_metrics = model._metrics_from_confusion(cm, len(y_val))

    print(f"\nValidation Set:")
    print(f"  Accuracy:    {val_metrics['accuracy']:.4f}")
    print(f"  AUC:         {val_auc:.4f}")
    print(f"  Precision:   {val_metrics['precision']:.4f}")
    print(f"  Recall:      {val_metrics['recall']:.4f}")
    print(f"  Specificity: {val_metrics['specificity']:.4f}")

    return {
        'model': model,
        'history': history,
        'val_metrics': {
            'accuracy': float(val_metrics['accuracy']),
            'auc': float(val_auc),
            'precision': float(val_metrics['precision']),
            'recall': float(val_metrics['recall']),
            'specificity': float(val_metrics['specificity'])
        },
        'n_train': n_train,
        'n_val': n_val,
        'tfrecord_path': tfrecord_path
    }


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def check_gpu():
    """Check and report GPU availability."""
    gpus = tf.config.list_physical_devices('GPU')
    print("="*60)
    print("GPU STATUS")
    print("="*60)
    if gpus:
        print(f"GPU(s) detected: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu.name}")
        print("Training will use GPU acceleration.")
    else:
        print("WARNING: No GPU detected!")
        print("Training will run on CPU (much, MUCH slower).")
        if is_kaggle():
            print("Make sure GPU is enabled in Kaggle notebook settings.")
    print()
    return len(gpus) > 0


def main():
    """
    Main execution flow for training.

    Pipeline:
    1. Load TFRecords from Kaggle dataset
    2. Train model with early stopping and LR scheduling
    3. Save weights and metrics
    4. Generate evaluation plots
    """

    # ========== GPU CHECK ==========
    has_gpu = check_gpu()

    # ========== HYPERPARAMETERS ==========
    HYPERPARAMETERS = {
        'n_samples': 4096,
        'learning_rate': 0.0001,
        'dropout_rate': 0.5,
        'epochs': 50,
        'batch_size': 64 if has_gpu else 32,
    }

    # ========== SETUP PATHS ==========
    print("="*60)
    print("SETUP (TFRECORD MODE)")
    print("="*60)

    output_dir = get_output_dir()
    print(f"Output directory: {output_dir}")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'}")
    print("Mode: TFRECORD (preprocessed data)")

    # find TFRecord file (external drive first, then project, then Kaggle)
    tfrecord_candidates = [
        Path("D:/Programming/g2net-preprocessed/train.tfrecord"),
        output_dir / "tfrecords" / "train.tfrecord",
        Path("/kaggle/input/g2net-preprocessed-tfrecords/train.tfrecord"),
    ]

    tfrecord_path = None
    for candidate in tfrecord_candidates:
        if candidate.exists():
            tfrecord_path = candidate
            break

    if tfrecord_path is None:
        raise FileNotFoundError(
            "TFRecord file not found. Expected at one of:\n" +
            "\n".join(f"  - {p}" for p in tfrecord_candidates)
        )

    print(f"TFRecord file: {tfrecord_path}")

    # load metadata to get sample count
    metadata_path = tfrecord_path.parent / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']
    else:
        n_samples = 560000  # approximate if metadata missing

    print(f"Total samples: {n_samples}")

    models_dir = output_dir / "models" / "saved"
    print(f"Models directory: {models_dir}")

    # ========== TRAIN MODEL ==========
    results = train_from_tfrecords(
        tfrecord_path,
        n_samples,
        hyperparameters=HYPERPARAMETERS,
        val_split=0.2
    )

    # ========== SAVE RESULTS ==========
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")

    saved_paths = save_model_and_metrics(
        results,
        HYPERPARAMETERS,
        models_dir
    )

    # ========== GENERATE PLOTS ==========
    print("\n" + "="*60)
    print("GENERATING PLOTS")
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
    print(f"  Config:  {saved_paths['config'].name}")
    print(f"  Metrics: {saved_paths['metrics'].name}")
    print(f"\nPlots saved:")
    for name, path in saved_plots.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
