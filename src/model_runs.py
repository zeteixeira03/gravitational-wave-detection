"""
1D CNN Model Training Pipeline

Training cycle using preprocessed PyTorch tensor shards:
1. Load .pt shards (preprocessed signals, must run compute_psd.py and create_tensors.py before - in this order)
2. Initialize 1D CNN model
3. Train
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
import torch.nn.functional as F
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
    plot_all_metrics,
    plot_lr_range_test,
)

SEED = 426425
torch.manual_seed(SEED)
np.random.seed(SEED)


# =====================================================================
#                           DATASET
# =====================================================================

class GWTensorDataset(Dataset):
    """Wraps signal and label tensors for DataLoader"""

    def __init__(self, signals: torch.Tensor, labels: torch.Tensor, augment: bool = False):
        self.signals = signals
        self.labels = labels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.signals[idx].clone()
        if self.augment:
            # time shift: roll each detector independently by 0-20 samples
            for ch in range(x.shape[0]):
                shift = int(torch.randint(0, 21, (1,)).item())
                x[ch] = torch.roll(x[ch], shift)
            # gaussian noise: scale relative to signal amplitude
            noise_scale = (0.01 + 0.09 * torch.rand(1).item()) * x.std().item()
            x = x + torch.randn_like(x) * noise_scale
        return x, self.labels[idx]


# =====================================================================
#                         TRAINING LOOP
# =====================================================================

def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup to a batch.

    Parameters
    ----------
    x : torch.Tensor
        Input signals of shape (batch, channels, time).
    y : torch.Tensor
        Labels of shape (batch, 1), may be soft.
    alpha : float
        Beta distribution concentration parameter.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Mixed signals and soft labels.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)

    return (lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm])


def fit(
    model,
    train_shard_paths,
    val_loader,
    optimizer,
    device,
    epochs,
    batch_size,
    verbose=True,
    early_stopping_patience=10,
    min_lr=1e-6,
    warmup_epochs=0,
    warmup_start_lr=1e-6,
    aux_loss_weight=0.0,
    use_amp=False,
    clip_grad_norm=None,
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
    min_lr : float
        Minimum learning rate
    warmup_epochs : int
        Number of epochs for linear LR warmup; 0 disables warmup
    warmup_start_lr : float
        Starting LR for warmup (linearly increases to the optimizer's initial LR)
    aux_loss_weight : float
        Weight for auxiliary per-branch losses (Phase 2a). 0 disables.
    use_amp : bool
        Enable mixed precision training (CUDA only).
    clip_grad_norm : float | None
        Max gradient norm for clipping. None disables.

    Returns
    -------
    dict
        Training history
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=min_lr)

    # amp setup (CUDA only)
    use_amp = use_amp and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0
    n_shards = len(train_shard_paths)
    target_lr = optimizer.param_groups[0]['lr']

    for epoch in range(epochs):
        # linear LR warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_lr = warmup_start_lr + (target_lr - warmup_start_lr) * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # training
        model.train()
        epoch_losses = []
        train_correct = 0
        train_total = 0

        shard_order = list(range(n_shards))
        np.random.shuffle(shard_order)

        for shard_num, shard_idx in enumerate(shard_order):
            shard_path = train_shard_paths[shard_idx]
            data = torch.load(str(shard_path), weights_only=True)
            shard_dataset = GWTensorDataset(data['signals'], data['labels'], augment=True)
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
                X_batch, y_batch = mixup_batch(X_batch, y_batch)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast("cuda", enabled=use_amp):
                    logits, branch_logits = model(X_batch)
                    loss = model.compute_loss(y_batch, logits, branch_logits, aux_loss_weight)
                scaler.scale(loss).backward()
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())
                with torch.no_grad():
                    pred_labels = (logits.float() >= 0.0).int().flatten()
                    train_correct += (pred_labels == y_batch.flatten().int()).sum().item()
                    train_total += len(y_batch)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            del data, shard_dataset, shard_loader

        train_loss = np.mean(epoch_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        history['train_loss'].append(train_loss)

        # validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch_float = y_batch.float().unsqueeze(1).to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(X_batch)
                    loss = model.compute_loss(y_batch_float, logits)
                val_losses.append(loss.item())

                pred_labels = (logits.float().cpu().numpy() >= 0.0).astype(int).flatten()
                val_correct += (pred_labels == y_batch.numpy()).sum()
                val_total += len(y_batch)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)

        # LR scheduling (skip during warmup to avoid premature reduction)
        if epoch >= warmup_epochs:
            scheduler.step(epoch - warmup_epochs)

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
                  f"LR: {current_lr:.2e}{improvement_marker}", flush=True)

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
    """Evaluate model on dataset, return probabilities and labels."""
    model.eval()
    y_true_all = []
    y_proba_all = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            proba = torch.sigmoid(logits).cpu().numpy().flatten()
            y_proba_all.append(proba)
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

def generate_plots(results, save_dir, base_name, max_plot_samples=10000):
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
    learning_rate = hyperparameters.get('learning_rate', 0.0001)
    dropout_rate = hyperparameters.get('dropout_rate', 0.5)
    weight_decay = hyperparameters.get('weight_decay', 1e-4)
    epochs = hyperparameters.get('epochs', 50)
    batch_size = hyperparameters.get('batch_size', 128)
    early_stopping_patience = hyperparameters.get('early_stopping_patience', 10)
    warmup_epochs = hyperparameters.get('warmup_epochs', 0)
    aux_loss_weight = hyperparameters.get('aux_loss_weight', 0.0)

    print(f"Signal length: {n_samples_config}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Aux loss weight: {aux_loss_weight}")
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
    n_channels = hyperparameters.get('n_channels', 32)
    model = DIYModel(n_channels=n_channels, dropout_rate=dropout_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DIYModel ({n_params:,} parameters)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # train
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    use_amp = hyperparameters.get('use_amp', False)
    clip_grad_norm = hyperparameters.get('clip_grad_norm', None)

    history = fit(
        model,
        train_shard_paths,
        val_loader,
        optimizer,
        device,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
        early_stopping_patience=early_stopping_patience,
        warmup_epochs=warmup_epochs,
        aux_loss_weight=aux_loss_weight,
        use_amp=use_amp,
        clip_grad_norm=clip_grad_norm,
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
#                        LR RANGE TEST
# =====================================================================

def lr_range_test(
    model,
    shard_paths,
    device,
    batch_size,
    use_amp=False,
    weight_decay=5e-4,
    lr_start=1e-5,
    lr_end=1e-1,
    n_steps=1000,
):
    """
    Exponentially increase LR over n_steps, record loss at each step.

    Parameters
    ----------
    model : nn.Module
        Model to test (weights will be modified).
    shard_paths : list[Path]
        Training data shard paths.
    device : torch.device
        Device to run on.
    batch_size : int
        Batch size.
    use_amp : bool
        Whether to use mixed precision.
    weight_decay : float
        Weight decay for AdamW.
    lr_start : float
        Starting learning rate.
    lr_end : float
        Ending learning rate.
    n_steps : int
        Number of optimization steps.

    Returns
    -------
    dict
        {'lrs': list, 'losses': list, 'smoothed': list, 'suggested_lr': float}
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)
    use_amp = use_amp and (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    lr_mult = (lr_end / lr_start) ** (1.0 / n_steps)
    lrs = []
    losses = []
    best_loss = float('inf')
    step = 0

    for shard_path in shard_paths:
        if step >= n_steps:
            break
        data = torch.load(str(shard_path), weights_only=True)
        dataset = GWTensorDataset(data['signals'], data['labels'], augment=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for X_batch, y_batch in loader:
            if step >= n_steps:
                break

            X_batch = X_batch.to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(X_batch)
                logits = output[0] if isinstance(output, tuple) else output
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lrs.append(optimizer.param_groups[0]['lr'])
            losses.append(loss.item())

            if step > 20 and loss.item() > 4 * best_loss:
                print(f"  Loss exploded at step {step}, stopping.")
                break
            best_loss = min(best_loss, loss.item())

            for pg in optimizer.param_groups:
                pg['lr'] *= lr_mult
            step += 1

        del data, dataset, loader
        if step > 20 and losses[-1] > 4 * best_loss:
            break

    if not losses:
        print("  No steps completed.")
        return {'lrs': [], 'losses': [], 'smoothed': [], 'suggested_lr': lr_start}

    # smooth losses (exponential moving average)
    smoothed = []
    avg = losses[0]
    for l in losses:
        avg = 0.98 * avg + 0.02 * l
        smoothed.append(avg)

    # suggested LR: where smoothed loss is minimum, divided by 5
    min_idx = int(np.argmin(smoothed))
    suggested_lr = lrs[min_idx] / 5

    print(f"\nLR Range Test: {len(lrs)} steps")
    print(f"  Best smoothed loss: {smoothed[min_idx]:.4f} at LR {lrs[min_idx]:.2e}")
    print(f"  Suggested LR: {suggested_lr:.2e}")

    return {'lrs': lrs, 'losses': losses, 'smoothed': smoothed, 'suggested_lr': suggested_lr}


def run_lr_range_test(data_dir, n_samples, hyperparameters):
    """
    Set up model and data, run LR range test, save plot.

    Parameters
    ----------
    data_dir : Path
        Directory containing tensor shards.
    n_samples : int
        Total number of samples in dataset.
    hyperparameters : dict
        Model and training configuration.
    """
    print("\n" + "=" * 60)
    print("LR RANGE TEST")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = hyperparameters.get('batch_size', 64)
    dropout_rate = hyperparameters.get('dropout_rate', 0.5)
    n_channels = hyperparameters.get('n_channels', 32)

    # find shards
    shard_files = sorted(data_dir.glob('shard_*.pt'))
    if not shard_files:
        single = data_dir / 'train.pt'
        if single.exists():
            shard_files = [single]
        else:
            raise FileNotFoundError(f"No data files in {data_dir}")

    # create model
    model = DIYModel(n_channels=n_channels, dropout_rate=dropout_rate).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DIYModel ({n_params:,} parameters)")
    print(f"Device: {device}")
    print(f"Shards: {len(shard_files)}")

    # run test
    use_amp = hyperparameters.get('use_amp', False)
    lr_data = lr_range_test(
        model, shard_files, device, batch_size,
        use_amp=use_amp,
        weight_decay=hyperparameters.get('weight_decay', 5e-4),
    )

    # save plot
    output_dir = get_output_dir()
    plot_path = output_dir / "lr_range_test.png"
    plot_lr_range_test(lr_data, save_path=str(plot_path))
    print(f"Plot saved: {plot_path}")

    return lr_data


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


def main(mode='train'):
    """
    Main execution flow for training or LR range test.

    Parameters
    ----------
    mode : str
        'train' for full training, 'lr_test' for LR range test.
    """

    # ========== GPU CHECK ==========
    has_gpu = check_gpu()

    # ========== HYPERPARAMETERS ==========
    HYPERPARAMETERS = {
        'n_channels': 32,
        'n_samples': 4096,
        'learning_rate': 1e-3,
        'dropout_rate': 0.5,
        'weight_decay': 5e-4,
        'epochs': 50,
        'batch_size': 64 if has_gpu else 32,
        'early_stopping_patience': 15,
        'warmup_epochs': 5,
        'aux_loss_weight': 0.2,
        'use_amp': True,
        'clip_grad_norm': 1.0,
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

    # ========== LR RANGE TEST MODE ==========
    if mode == 'lr_test':
        run_lr_range_test(data_dir, n_samples, HYPERPARAMETERS)
        return

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
