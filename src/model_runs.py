"""
1D CNN Model Training Pipeline

Training cycle using preprocessed PyTorch tensor shards:
1. Load .pt shards (preprocessed signals, must run compute_psd.py and create_tensors.py before - in this order)
2. Initialize 1D CNN model
3. Train with early stopping and LR scheduling
4. Evaluate and save metrics
5. Generate evaluation plots
"""
import copy
import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

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

SEED = 426425
torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================================
#                           DATASET
# =====================================================================

class GWTensorDataset(Dataset):
    """Wraps signal and label tensors for DataLoader."""

    def __init__(self, signals: torch.Tensor, labels: torch.Tensor):
        self.signals = signals
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.signals[idx], self.labels[idx]


# =====================================================================
#                         TRAINING LOOP
# =====================================================================

def fit(
    model,
    train_shard_paths,
    val_loader,
    optimizer,
    device,
    epochs,
    batch_size,
    verbose=True,
    early_stopping_patience=5,
    lr_reduce_patience=3,
    lr_reduce_factor=0.5,
    min_lr=1e-6,
):
    """
    Train model by streaming shards from disk.

    Each epoch iterates over all training shards, loading one at a time to keep
    memory usage constant (~2.4 GB per shard). Works with any number of shards,
    including a single file.

    Parameters
    ----------
    model : DIYModel
        Model to train
    train_shard_paths : list[Path]
        Paths to training shard .pt files
    val_loader : DataLoader
        Validation data loader (kept in memory)
    optimizer : torch.optim.Optimizer
        Optimizer instance
    device : torch.device
        Device to train on
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    verbose : bool
        Whether to print progress
    early_stopping_patience : int
        Stop training if val_loss doesn't improve for this many epochs
    lr_reduce_patience : int
        Reduce LR if val_loss doesn't improve for this many epochs
    lr_reduce_factor : float
        Factor to multiply LR by when reducing
    min_lr : float
        Minimum learning rate

    Returns
    -------
    dict
        Training history
    """
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_reduce_patience, factor=lr_reduce_factor,
        min_lr=min_lr
    )

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    n_shards = len(train_shard_paths)

    for epoch in range(epochs):
        # ---- training ----
        model.train()
        epoch_losses = []
        train_correct = 0
        train_total = 0

        shard_order = list(range(n_shards))
        np.random.shuffle(shard_order)

        for shard_num, shard_idx in enumerate(shard_order):
            shard_path = train_shard_paths[shard_idx]
            data = torch.load(str(shard_path), weights_only=True)
            shard_dataset = GWTensorDataset(data['signals'], data['labels'])
            shard_loader = DataLoader(
                shard_dataset, batch_size=batch_size, shuffle=True,
                num_workers=0, pin_memory=(device.type == 'cuda')
            )

            desc = f"Epoch {epoch+1}/{epochs}"
            if n_shards > 1:
                desc += f" [shard {shard_num+1}/{n_shards}]"
            pbar = tqdm(shard_loader, desc=desc, disable=not verbose)

            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)

                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = model.compute_loss(y_batch, predictions)
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                with torch.no_grad():
                    pred_labels = (predictions >= 0.5).int().flatten()
                    train_correct += (pred_labels == y_batch.flatten().int()).sum().item()
                    train_total += len(y_batch)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            del data, shard_dataset, shard_loader

        train_loss = np.mean(epoch_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        history['train_loss'].append(train_loss)

        # ---- validation ----
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch_float = y_batch.float().unsqueeze(1).to(device)

                pred = model(X_batch)
                loss = model.compute_loss(y_batch_float, pred)
                val_losses.append(loss.item())

                pred_labels = (pred.cpu().numpy() >= 0.5).astype(int).flatten()
                val_correct += (pred_labels == y_batch.numpy()).sum()
                val_total += len(y_batch)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)

        # LR scheduling
        scheduler.step(val_loss)

        # check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            improvement_marker = " *"
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                  f"LR: {current_lr:.2e}{improvement_marker}")

        # early stopping
        if epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping: val_loss hasn't improved for {early_stopping_patience} epochs")
            break

    # restore best weights
    if best_state is not None:
        if verbose:
            print(f"Restoring best weights (val_loss: {best_val_loss:.4f})")
        model.load_state_dict(best_state)

    return history


def evaluate(model, data_loader, device):
    """Evaluate model on dataset, return predictions and labels."""
    model.eval()
    y_true_all = []
    y_proba_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy().flatten()
            y_proba_all.append(pred)
            y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all)
    y_proba = np.concatenate(y_proba_all)

    return y_true, y_proba


# =====================================================================
#                          SAVE RESULTS
# =====================================================================

def save_model_and_metrics(results, hyperparameters, save_dir):
    """
    Save model weights, hyperparameters, and metrics.

    Parameters
    ----------
    results : dict
        Training results from train_from_tensors()
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base_name = f"diy_{timestamp}"

    # save weights
    weights_path = save_dir / f"{base_name}_weights.pt"
    results['model'].save_weights(str(weights_path))
    print(f"Weights saved to: {weights_path}")

    # save hyperparameters
    config_path = save_dir / f"{base_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2)
    print(f"Config saved to: {config_path}")

    # save metrics
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
#                          VISUALIZATION
# =====================================================================

def generate_plots(results, save_dir, base_name, device, max_plot_samples=10000):
    """
    Generate evaluation plots after training.

    Parameters
    ----------
    results : dict
        Training results from train_from_tensors()
    save_dir : Path
        Directory to save plot files
    base_name : str
        Base filename for saved plots
    device : torch.device
        Device for inference
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

    # use validation data for plotting
    val_loader = results['val_loader']
    n_plot = min(max_plot_samples, results['n_val'])
    print(f"Loading {n_plot} validation samples for plotting...")

    X_list = []
    y_list = []
    collected = 0
    for X_batch, y_batch in val_loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())
        collected += len(y_batch)
        if collected >= n_plot:
            break

    X_val = np.concatenate(X_list, axis=0)[:n_plot]
    plot_y = np.concatenate(y_list, axis=0)[:n_plot]

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
#                       TENSOR LOADING
# =====================================================================

def train_from_tensors(data_dir, n_samples, hyperparameters, val_split=0.2):
    """
    Train DIY 1D CNN model from preprocessed .pt tensor shards.

    Streams training shards from disk one at a time (~2.4 GB each) to avoid
    loading the full dataset into memory. Validation shards are kept in memory.

    Parameters
    ----------
    data_dir : Path
        Directory containing shard_*.pt files (or a single train.pt for small datasets)
    n_samples : int
        Total number of samples
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
    print("INITIALIZING 1D CNN MODEL (TENSOR MODE)")
    print("="*60)

    n_samples_config = hyperparameters.get('n_samples', 4096)
    learning_rate = hyperparameters.get('learning_rate', 0.001)
    dropout_rate = hyperparameters.get('dropout_rate', 0.3)
    weight_decay = hyperparameters.get('weight_decay', 1e-4)
    epochs = hyperparameters.get('epochs', 15)
    batch_size = hyperparameters.get('batch_size', 128)

    print(f"Signal length: {n_samples_config}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Total samples: {n_samples}")
    print("Mode: TENSOR (preprocessed data, shard streaming)")

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # split shards into val / train
    single_file = data_dir / "train.pt"
    if single_file.exists():
        # small dataset mode: load entirely into memory
        print(f"\nLoading single file: {single_file}")
        data = torch.load(str(single_file), weights_only=True)
        all_signals = data['signals']
        all_labels = data['labels']

        n_val = int(len(all_labels) * val_split)
        n_train = len(all_labels) - n_val

        val_signals = all_signals[:n_val]
        val_labels = all_labels[:n_val]

        # save train split as a temporary shard so fit() streams it from disk
        train_shard = data_dir / "_train_split.pt"
        torch.save({'signals': all_signals[n_val:], 'labels': all_labels[n_val:]}, str(train_shard))
        train_shard_paths = [train_shard]
        del all_signals, all_labels
    else:
        # shard mode: split by shard files
        shard_files = sorted(data_dir.glob('shard_*.pt'))
        if not shard_files:
            raise FileNotFoundError(f"No shard or train.pt files found in {data_dir}")

        n_val_target = int(n_samples * val_split)

        # assign first N shards to validation until we have enough samples
        val_shard_paths = []
        train_shard_paths = []
        val_count = 0

        for f in shard_files:
            if val_count < n_val_target:
                val_shard_paths.append(f)
                data = torch.load(str(f), weights_only=True)
                val_count += len(data['labels'])
                del data
            else:
                train_shard_paths.append(f)

        # load val shards into memory (small enough: ~2-3 shards = 5-7 GB)
        print(f"\nLoading {len(val_shard_paths)} validation shards into memory...")
        val_signals_list = []
        val_labels_list = []
        for f in val_shard_paths:
            print(f"  {f.name}")
            data = torch.load(str(f), weights_only=True)
            val_signals_list.append(data['signals'])
            val_labels_list.append(data['labels'])
            del data

        val_signals = torch.cat(val_signals_list)
        val_labels = torch.cat(val_labels_list)
        del val_signals_list, val_labels_list

        n_val = len(val_labels)
        n_train = n_samples - n_val

    print(f"Training shards: {len(train_shard_paths)} (streamed from disk)")
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")

    # validation DataLoader (always in memory)
    val_dataset = GWTensorDataset(val_signals, val_labels)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # initialize model
    model = DIYModel(
        n_samples=n_samples_config,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # train
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    history = fit(
        model, train_shard_paths, val_loader, optimizer, device,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    print("\nTraining complete.")

    # evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    y_val, y_val_proba = evaluate(model, val_loader, device)
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
        'val_loader': val_loader,
        'device': device,
    }


# =====================================================================
#                        MAIN ENTRY POINT
# =====================================================================

def check_gpu():
    """Check and report GPU availability."""
    print("="*60)
    print("GPU STATUS")
    print("="*60)
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPU(s) detected: {n_gpus}")
        for i in range(n_gpus):
            print(f"  - {torch.cuda.get_device_name(i)}")
        print("Training will use GPU acceleration.")
    else:
        print("WARNING: No GPU detected!")
        print("Training will run on CPU (much, MUCH slower).")
        if is_kaggle():
            print("Make sure GPU is enabled in Kaggle notebook settings.")
    print()
    return torch.cuda.is_available()


def main():
    """
    Main execution flow for training.

    Pipeline:
    1. Load .pt tensor shards from Kaggle dataset
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
        'weight_decay': 1e-4,
        'epochs': 50,
        'batch_size': 64 if has_gpu else 32,
    }

    # ========== SETUP PATHS ==========
    print("="*60)
    print("SETUP (TENSOR MODE)")
    print("="*60)

    output_dir = get_output_dir()
    print(f"Output directory: {output_dir}")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'}")
    print("Mode: TENSOR (preprocessed data)")

    # find data directory containing shards or train.pt
    data_dir_candidates = [
        Path("D:/Programming/g2net-preprocessed"),
        output_dir / "tensors",
        Path("/kaggle/input/g2net-preprocessed-tfrecords"),
    ]

    data_dir = None
    for candidate in data_dir_candidates:
        if candidate.exists() and (
            list(candidate.glob("shard_*.pt")) or (candidate / "train.pt").exists()
        ):
            data_dir = candidate
            break

    if data_dir is None:
        raise FileNotFoundError(
            "Tensor data not found. Expected shard_*.pt or train.pt in one of:\n" +
            "\n".join(f"  - {p}" for p in data_dir_candidates)
        )

    print(f"Data directory: {data_dir}")

    # load metadata to get sample count
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        n_samples = metadata['n_samples']
    else:
        n_samples = 560000  # approximate if metadata missing

    print(f"Total samples: {n_samples}")

    models_dir = output_dir / "models" / "saved"
    print(f"Models directory: {models_dir}")

    # ========== TRAIN MODEL ==========
    results = train_from_tensors(
        data_dir,
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
        saved_paths['base_name'],
        results['device']
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
