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
import random
import time
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import Dataset, DataLoader

# add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.g2net import is_kaggle, get_output_dir
from models.diy_model import DIYModel, SkyHeadModel
from sky_feasibility import SkyGeometry
from evaluation import (
    compute_confusion_values,
    metrics_from_confusion,
    confusion_matrix as compute_cm,
    evaluate_metrics,
    roc_curve,
    precision_recall_curve,
)
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


def set_seed(seed: int) -> None:
    """
    Seed every RNG a run depends on and put cuDNN in deterministic mode.

    Covers Python ``random``, NumPy, and torch (CPU + all CUDA devices). cuDNN
    is set deterministic with autotuning off so convolutions pick a fixed
    algorithm. ``torch.use_deterministic_algorithms(True)`` is deliberately not
    set: several 1-D pooling/conv backward kernels lack deterministic CUDA
    implementations and would raise on the P100. cuDNN-determinism plus full
    seeding leaves only sub-ULP atomic-add nondeterminism, far below the
    seed-to-seed AUC variance the analysis plan measures.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """DataLoader worker init: seed NumPy/random from the per-worker torch seed
    so on-the-fly augmentation is reproducible across runs."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(hyperparameters: dict, n_sky_features: int, device) -> nn.Module:
    """
    Construct the model for a run from its config flags.

    ``sky_head_only`` builds the CNN-free SkyHeadModel (sweep config 5).
    Otherwise a DIYModel is built with the sky readout, H1/L1 merge, and
    parameter-matching selected by config.
    """
    if hyperparameters.get('sky_head_only', False):
        model = SkyHeadModel(
            n_sky_features,
            hidden_dim=hyperparameters.get('sky_head_hidden', 128),
            dropout_rate=hyperparameters.get('dropout_rate', 0.5),
        )
    else:
        seed = hyperparameters.get('seed', 0)
        model = DIYModel(
            n_channels=hyperparameters.get('n_channels', 16),
            dropout_rate=hyperparameters.get('dropout_rate', 0.5),
            drop_path_rate=hyperparameters.get('drop_path_rate', 0.0),
            n_sky_features=n_sky_features,
            sky_readout=hyperparameters.get('sky_readout', 'mlp121'),
            h1l1_merge=hyperparameters.get('h1l1_merge', 'concat'),
            h1l1_pool=hyperparameters.get('h1l1_pool', 'mean'),
            match_params=hyperparameters.get('match_params', False),
            l_max=hyperparameters.get('sky_l_max', 10),
            l_bisp=hyperparameters.get('sky_l_bisp', 4),
            scramble_seed=hyperparameters.get('scramble_seed', seed),
        )
    return model.to(device)


# =====================================================================
#                           DATASET
# =====================================================================

class GWTensorDataset(Dataset):
    """Wraps signal and label tensors for DataLoader, with optional sky feature extraction."""

    def __init__(self, signals: torch.Tensor, labels: torch.Tensor, augment: bool = False, sky_geometry: SkyGeometry | None = None, precomputed_sh: torch.Tensor | None = None, aug_config: dict | None = None):
        self.signals = signals
        self.labels = labels
        self.augment = augment
        self.sky_geometry = sky_geometry
        self.precomputed_sh = precomputed_sh
        self.aug_config = aug_config or {
            'time_shift': True, 'noise': True,
            'spectral_dropout': True, 'channel_shuffle': True,
            'amplitude_scale': True,
        }

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = self.signals[idx].clone()

        # sky features: use precomputed if available, else compute on-the-fly
        # from the clean signal before augmentation -- time shifts and channel
        # shuffles break the physical time-delay relationships the sky map encodes
        if self.precomputed_sh is not None:
            sh_coeffs = self.precomputed_sh[idx]
        else:
            sh_coeffs = torch.tensor(self.sky_geometry.extract(x.numpy()), dtype=torch.float32)

        if self.augment:
            if self.aug_config['time_shift']:
                for ch in range(x.shape[0]):
                    shift = int(torch.randint(0, 21, (1,)).item())
                    x[ch] = torch.roll(x[ch], shift)
            if self.aug_config['noise']:
                noise_scale = (0.01 + 0.09 * torch.rand(1).item()) * x.std().item()
                x = x + torch.randn_like(x) * noise_scale
            if self.aug_config['spectral_dropout']:
                spec = torch.fft.rfft(x, dim=-1)
                x = torch.fft.irfft(spec * (torch.rand(spec.shape) > 0.05), n=x.shape[-1])
            if self.aug_config.get('amplitude_scale', False):
                scale = 0.8 + 0.4 * torch.rand(1).item()
                x = x * scale
            if self.aug_config['channel_shuffle'] and torch.rand(1).item() > 0.5:
                x = x[[1, 0, 2]]

        return x, self.labels[idx], sh_coeffs


# =====================================================================
#                         TRAINING LOOP
# =====================================================================

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    sky: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply mixup to a batch (signals, labels, and sky features).

    Parameters
    ----------
    x : torch.Tensor
        Input signals of shape (batch, channels, time).
    y : torch.Tensor
        Labels of shape (batch, 1), may be soft.
    sky : torch.Tensor
        Sky features of shape (batch, n_sky_features).
    alpha : float
        Beta distribution concentration parameter.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Mixed signals, soft labels, and dominant sample's sky features.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)

    # sky features encode geometric consistency for a specific source --
    # blending two unrelated sky maps produces a non-physical vector.
    # keep the dominant sample's coefficients intact.
    sky_out = sky if lam >= 0.5 else sky[perm]

    return (
        lam * x + (1 - lam) * x[perm],
        lam * y + (1 - lam) * y[perm],
        sky_out,
    )


def manifold_mixup_pooled(
    pooled: torch.Tensor,
    y: torch.Tensor,
    sky: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mix pooled CNN features (post-fusion, post-pool) instead of raw signals.

    Mixing happens after the convolutional path has condensed each sample to
    a single feature vector, so the mix never touches the time domain and
    cannot interfere with the cross-detector phase coherence the sky map
    encodes. Sky features still belong to a single physical source: we keep
    the dominant sample's coefficients (lam >= 0.5) intact, the same rule
    used by ``mixup_batch``.

    Parameters
    ----------
    pooled : torch.Tensor
        Pooled feature vectors of shape (batch, feat_dim).
    y : torch.Tensor
        Labels of shape (batch, 1), may be soft.
    sky : torch.Tensor
        Sky features of shape (batch, n_sky_features).
    alpha : float
        Beta distribution concentration. 0.4 puts more mass at the extremes
        than mixup's 0.2 default, which works better for feature-space mixing.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Mixed features, soft labels, dominant sample's sky features.
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(pooled.size(0), device=pooled.device)
    sky_out = sky if lam >= 0.5 else sky[perm]
    return (
        lam * pooled + (1 - lam) * pooled[perm],
        lam * y + (1 - lam) * y[perm],
        sky_out,
    )


def update_swa_bn(
    swa_model,
    train_shard_paths,
    device,
    batch_size,
    sky_geometry,
    verbose: bool = True,
    max_shards: int = 2,
):
    """
    Recompute BatchNorm running stats for an SWA-averaged model.

    PyTorch's stock ``torch.optim.swa_utils.update_bn`` assumes the loader
    yields tensors directly compatible with ``model(x)``. Our loaders yield
    ``(signals, labels, sky)`` triples and the model takes two named
    arguments, so we iterate the shards by hand. BatchNorm layers are reset
    to a cumulative average (momentum=None) so the new statistics depend
    only on this pass over the data, not on whatever they had during the
    last pre-SWA epoch.
    """
    bn_layers = [m for m in swa_model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return

    momenta = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None

    was_training = swa_model.training
    swa_model.train()

    # BN running stats converge after ~10k samples under cumulative
    # averaging (momentum=None); 2 shards (~120k) is overkill but cheap.
    # Using all 9 shards was both wasteful (~3min) and enlarged the window
    # for CUDA faults seen once on Kaggle P100.
    shards_to_use = list(train_shard_paths)[:max_shards]
    if verbose:
        print(f"Updating BN running stats over {len(shards_to_use)} shards "
              f"(of {len(train_shard_paths)} available)...")

    with torch.no_grad():
        for shard_path in shards_to_use:
            data = torch.load(str(shard_path), weights_only=True)
            precomputed_sh = data.get('sh_coeffs', None)
            ds = GWTensorDataset(
                data['signals'], data['labels'],
                augment=False, sky_geometry=sky_geometry,
                precomputed_sh=precomputed_sh,
            )
            loader = DataLoader(
                ds, batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=(device.type == 'cuda'),
            )
            for X_batch, _y_batch, sky_batch in loader:
                X_batch = X_batch.to(device)
                sky_batch = sky_batch.to(device)
                swa_model(X_batch, sky_features=sky_batch)
            del data, ds, loader

    for m, mom in zip(bn_layers, momenta):
        m.momentum = mom
    if not was_training:
        swa_model.eval()


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
    max_train_hours=None,
    sky_geometry=None,
    use_mixup=True,
    use_manifold_mixup=False,
    use_swa=False,
    swa_start_epoch=None,
    aug_config=None,
    label_smoothing=0.0,
    checkpoint_dir=None,
    seed=426425,
):
    """
    Train model by streaming shards from disk. Sky features are computed
    on-the-fly per sample (or loaded from precomputed shards).

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
    max_train_hours : float | None
        Wall-clock time budget for training. Stops after completing the
        current epoch if the next epoch would exceed the budget, leaving
        time for evaluation and saving. None disables.
    sky_geometry : SkyGeometry | None
        Precomputed sky geometry for on-the-fly SH coefficient extraction.
        None disables sky features.

    Returns
    -------
    dict
        Training history
    """
    # cosine schedule covers the pre-SWA window (or all of training when
    # SWA is disabled). After swa_start_epoch we hand off to SWALR.
    target_lr = optimizer.param_groups[0]['lr']
    if use_swa and swa_start_epoch is None:
        swa_start_epoch = max(warmup_epochs, int(epochs * 0.7))
    cosine_t_max = (swa_start_epoch if use_swa else epochs) - warmup_epochs
    cosine_t_max = max(cosine_t_max, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=min_lr)

    swa_model = None
    swa_scheduler = None
    if use_swa:
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=target_lr * 0.1,
            anneal_epochs=3,
            anneal_strategy='cos',
        )
        if verbose:
            print(f"SWA enabled: averaging from epoch {swa_start_epoch + 1} "
                  f"to epoch {epochs} at LR {target_lr * 0.1:.2e}")

    train_start = time.monotonic()

    # amp setup (CUDA only)
    use_amp = use_amp and (device.type == 'cuda')
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': [],
    }

    best_val_loss = float('inf')
    best_state = None
    epochs_without_improvement = 0

    # on-disk checkpoint paths. best.pt is overwritten whenever val_loss
    # improves so a crash inside SWA finalization still leaves usable
    # weights in /kaggle/working (which the kaggle runner commits even
    # on script failure). swa_pre_bn.pt is a snapshot of the averaged
    # weights taken before the BN recompute pass, so a CUDA fault in
    # update_swa_bn doesn't wipe out the SWA result.
    best_ckpt_path = None
    swa_pre_bn_path = None
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_ckpt_path = checkpoint_dir / "checkpoint_best.pt"
        swa_pre_bn_path = checkpoint_dir / "checkpoint_swa_pre_bn.pt"
    n_shards = len(train_shard_paths)
    nan_batches_total = 0
    nan_batches_consecutive = 0
    max_consecutive_nan = 100
    first_nan_reported = False

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
            precomputed_sh = data.get('sh_coeffs', None)
            shard_dataset = GWTensorDataset(data['signals'], data['labels'], augment=True, sky_geometry=sky_geometry, precomputed_sh=precomputed_sh, aug_config=aug_config)
            # reproducible-but-varying shuffle: generator seed depends on the run
            # seed, epoch, and shard so each pass differs yet repeats across runs.
            loader_gen = torch.Generator()
            loader_gen.manual_seed(seed * 100003 + epoch * 997 + shard_idx)
            shard_loader = DataLoader(
                shard_dataset, batch_size=batch_size, shuffle=True,
                num_workers=2, pin_memory=(device.type == 'cuda'),
                worker_init_fn=_seed_worker, generator=loader_gen,
            )

            desc = f"Epoch {epoch+1}/{epochs}"
            if n_shards > 1:
                desc += f" [shard {shard_num+1}/{n_shards}]"
            pbar = tqdm(shard_loader, desc=desc, disable=not verbose)

            for X_batch, y_batch, sky_batch in pbar:
                X_batch = X_batch.to(device)
                sky_batch = sky_batch.to(device)
                y_batch = y_batch.float().unsqueeze(1).to(device)
                if use_mixup:
                    X_batch, y_batch, sky_batch = mixup_batch(X_batch, y_batch, sky_batch)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if use_manifold_mixup:
                        # mix in feature space after pool. aux heads are
                        # bypassed because the per-branch tensors here are
                        # unmixed and would carry the wrong labels.
                        pooled, _ = model._extract_pooled(X_batch)
                        pooled, y_batch, sky_batch = manifold_mixup_pooled(pooled, y_batch, sky_batch)
                        logits = model.classify_from_pooled(pooled, sky_batch)
                        loss = model.compute_loss(y_batch, logits, None, 0.0, label_smoothing)
                    else:
                        logits, branch_logits = model(X_batch, sky_features=sky_batch)
                        loss = model.compute_loss(y_batch, logits, branch_logits, aux_loss_weight, label_smoothing)

                # NaN guard: skip the batch if the loss is not finite. logs
                # the first occurrence with context, counts consecutive NaNs,
                # and aborts training if they keep coming so we don't waste
                # compute on a collapsed model.
                loss_val = loss.item()
                if not np.isfinite(loss_val):
                    nan_batches_total += 1
                    nan_batches_consecutive += 1
                    if not first_nan_reported:
                        print(f"\n[NaN] first non-finite loss at epoch {epoch+1}, "
                              f"shard {shard_num+1}/{n_shards}, step in shard {pbar.n}. "
                              f"Skipping this batch.", flush=True)
                        first_nan_reported = True
                    if nan_batches_consecutive >= max_consecutive_nan:
                        print(f"\n[NaN] {nan_batches_consecutive} consecutive NaN batches, "
                              f"aborting training.", flush=True)
                        raise RuntimeError("training collapsed (NaN loss)")
                    continue

                nan_batches_consecutive = 0
                scaler.scale(loss).backward()
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss_val)
                with torch.no_grad():
                    pred_labels = (logits.float() >= 0.0).int().flatten()
                    train_correct += (pred_labels == y_batch.flatten().int()).sum().item()
                    train_total += len(y_batch)
                pbar.set_postfix(loss=f"{loss_val:.4f}")

            del data, shard_dataset, shard_loader

        train_loss = np.mean(epoch_losses)
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        history['train_loss'].append(train_loss)

        # validation
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        val_probas = []
        val_labels_all = []

        with torch.no_grad():
            for X_batch, y_batch, sky_batch in val_loader:
                X_batch = X_batch.to(device)
                sky_batch = sky_batch.to(device)
                y_batch_float = y_batch.float().unsqueeze(1).to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X_batch, sky_features=sky_batch)
                    loss = model.compute_loss(y_batch_float, logits)
                val_losses.append(loss.item())

                logits_np = logits.float().cpu().numpy().flatten()
                pred_labels = (logits_np >= 0.0).astype(int)
                y_np = y_batch.numpy()
                val_correct += (pred_labels == y_np).sum()
                val_total += len(y_batch)
                val_probas.append(1.0 / (1.0 + np.exp(-logits_np)))
                val_labels_all.append(y_np)

        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        try:
            val_auc = roc_auc_score(
                np.concatenate(val_labels_all),
                np.concatenate(val_probas),
            )
        except ValueError:
            val_auc = float('nan')

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(float(val_auc))
        history['train_acc'].append(train_acc)

        # LR scheduling (skip during warmup to avoid premature reduction).
        # SWA phase uses SWALR + parameter averaging instead of cosine.
        in_swa_phase = use_swa and epoch >= swa_start_epoch
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        elif epoch >= warmup_epochs:
            scheduler.step()

        # check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            improvement_marker = " *"
            if best_ckpt_path is not None:
                torch.save(
                    {
                        'model_state_dict': best_state,
                        'epoch': epoch + 1,
                        'val_loss': float(val_loss),
                        'val_acc': float(val_acc),
                    },
                    str(best_ckpt_path),
                )
        else:
            epochs_without_improvement += 1
            improvement_marker = ""

        if checkpoint_dir is not None:
            torch.save(history, str(Path(checkpoint_dir) / "history.pt"))

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - "
                  f"Val AUC: {val_auc:.4f} - "
                  f"LR: {current_lr:.2e}{improvement_marker}", flush=True)

        # early stopping (disabled during the SWA phase: we want a stable
        # window to average over, not an early bailout on noise)
        if not in_swa_phase and epochs_without_improvement >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping: val_loss hasn't improved for {early_stopping_patience} epochs")
            break

        # time budget check: stop if the next epoch would exceed the budget
        if max_train_hours is not None:
            elapsed_h = (time.monotonic() - train_start) / 3600
            h_per_epoch = elapsed_h / (epoch + 1)
            if elapsed_h + h_per_epoch > max_train_hours:
                if verbose:
                    print(f"\nTime budget: {elapsed_h:.2f}h elapsed, ~{h_per_epoch:.2f}h/epoch, "
                          f"stopping to stay within {max_train_hours}h limit")
                break

    # SWA: recompute BN stats over the training data and copy averaged
    # weights into the live model. SWA replaces best-weight restoration --
    # the averaged model is the final model regardless of its last
    # epoch's val loss.
    if use_swa and swa_model is not None and epoch >= swa_start_epoch:
        # snapshot the averaged weights to disk BEFORE the BN recompute
        # pass. a CUDA fault during update_swa_bn (seen once on Kaggle P100)
        # is otherwise fatal: the averaged params live only in GPU memory
        # inside swa_model.module, and the crash wipes everything.
        if swa_pre_bn_path is not None:
            torch.save(
                {'model_state_dict': swa_model.module.state_dict()},
                str(swa_pre_bn_path),
            )
        if verbose:
            print("\nFinalizing SWA: updating BN running stats over training data...")
        try:
            update_swa_bn(swa_model, train_shard_paths, device, batch_size, sky_geometry, verbose=verbose)
            model.load_state_dict(swa_model.module.state_dict())
            if verbose:
                print("SWA averaged weights copied into model.")
        except Exception as e:
            # BN pass failed (CUDA fault, OOM, etc). fall back to the best
            # per-epoch weights so the run still produces something usable.
            # the pre-BN SWA snapshot is on disk for manual recovery if the
            # user wants to retry the BN pass in a separate job.
            print(f"\n[SWA] update_swa_bn failed: {type(e).__name__}: {e}", flush=True)
            print("[SWA] Falling back to best per-epoch weights.", flush=True)
            if best_state is not None:
                model.load_state_dict(best_state)
    elif best_state is not None:
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
        for X_batch, y_batch, sky_batch in data_loader:
            X_batch = X_batch.to(device)
            sky_batch = sky_batch.to(device)
            logits = model(X_batch, sky_features=sky_batch)
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
    sky_list = []
    collected = 0
    for X_batch, y_batch, sky_batch in val_loader:
        X_list.append(X_batch.numpy())
        y_list.append(y_batch.numpy())
        sky_list.append(sky_batch.numpy())
        collected += len(y_batch)
        if collected >= n_plot:
            break

    X_val = np.concatenate(X_list, axis=0)[:n_plot]
    plot_y = np.concatenate(y_list, axis=0)[:n_plot]
    sky_val = np.concatenate(sky_list, axis=0)[:n_plot]

    saved_plots = {}

    # single prediction pass for all plots
    print("Computing predictions...")
    y_proba = model.predict_proba(X_val, sky_val)

    roc_data = roc_curve(y_proba, plot_y)
    pr_data = precision_recall_curve(y_proba, plot_y)
    cm_data = compute_cm(y_proba, plot_y)

    print("Generating plots...")

    # 1. Learning curves
    print("  - Learning curves")
    learning_path = plots_dir / f"{base_name}_learning_curves.png"
    plot_learning_curves(history, metrics=['loss', 'acc'], save_path=str(learning_path))
    saved_plots['learning_curves'] = learning_path

    # 2. ROC curve
    print("  - ROC curve")
    roc_path = plots_dir / f"{base_name}_roc_curve.png"
    plot_roc_curve(roc_data, save_path=str(roc_path))
    saved_plots['roc_curve'] = roc_path

    # 3. Precision-Recall curve
    print("  - Precision-Recall curve")
    pr_path = plots_dir / f"{base_name}_pr_curve.png"
    plot_precision_recall_curve(pr_data, save_path=str(pr_path))
    saved_plots['pr_curve'] = pr_path

    # 4. Confusion matrix
    print("  - Confusion matrix")
    cm_path = plots_dir / f"{base_name}_confusion_matrix.png"
    plot_confusion_matrix(cm_data, normalize=True, save_path=str(cm_path))
    saved_plots['confusion_matrix'] = cm_path

    # 5. Prediction distribution
    print("  - Prediction distribution")
    dist_path = plots_dir / f"{base_name}_prediction_dist.png"
    plot_prediction_distribution(y_proba, plot_y, save_path=str(dist_path))
    saved_plots['prediction_dist'] = dist_path

    # 6. Combined dashboard
    print("  - Combined dashboard")
    dashboard_path = plots_dir / f"{base_name}_dashboard.png"
    plot_all_metrics(y_proba, plot_y, history=history, save_path=str(dashboard_path))
    saved_plots['dashboard'] = dashboard_path

    print(f"Plots saved to: {plots_dir}")

    return saved_plots


# =====================================================================
#                       TENSOR LOADING
# =====================================================================

def train_from_tensors(data_dir, n_samples, hyperparameters, val_split=0.2, checkpoint_dir=None):
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

    seed = hyperparameters.get('seed', 0)
    set_seed(seed)

    n_samples_config = hyperparameters.get('n_samples', 4096)
    learning_rate = hyperparameters.get('learning_rate', 0.0001)
    dropout_rate = hyperparameters.get('dropout_rate', 0.5)
    weight_decay = hyperparameters.get('weight_decay', 1e-4)
    epochs = hyperparameters.get('epochs', 50)
    batch_size = hyperparameters.get('batch_size', 128)
    early_stopping_patience = hyperparameters.get('early_stopping_patience', 10)
    warmup_epochs = hyperparameters.get('warmup_epochs', 0)
    aux_loss_weight = hyperparameters.get('aux_loss_weight', 0.0)
    drop_path_rate = hyperparameters.get('drop_path_rate', 0.0)
    sky_n_pix = hyperparameters.get('sky_n_pix', 192)
    sky_l_max = hyperparameters.get('sky_l_max', 10)
    sky_readout = hyperparameters.get('sky_readout', 'mlp121')
    h1l1_merge = hyperparameters.get('h1l1_merge', 'concat')
    sky_head_only = hyperparameters.get('sky_head_only', False)
    use_mixup = hyperparameters.get('use_mixup', True)
    use_manifold_mixup = hyperparameters.get('use_manifold_mixup', False)
    if sky_head_only and use_manifold_mixup:
        raise ValueError("sky_head_only has no CNN pathway; disable use_manifold_mixup for this config")
    use_swa = hyperparameters.get('use_swa', False)
    swa_start_epoch = hyperparameters.get('swa_start_epoch', None)
    label_smoothing = hyperparameters.get('label_smoothing', 0.0)
    aug_config = {
        'time_shift': hyperparameters.get('aug_time_shift', True),
        'noise': hyperparameters.get('aug_noise', True),
        'spectral_dropout': hyperparameters.get('aug_spectral_dropout', True),
        'channel_shuffle': hyperparameters.get('aug_channel_shuffle', True),
        'amplitude_scale': hyperparameters.get('aug_amplitude_scale', False),
    }

    print(f"Signal length: {n_samples_config}")
    print(f"Learning rate: {learning_rate}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Aux loss weight: {aux_loss_weight}")
    print(f"Drop path rate: {drop_path_rate}")
    print(f"Label smoothing: {label_smoothing}")
    print(f"Sky: n_pix={sky_n_pix}, l_max={sky_l_max}")
    print(f"Seed: {seed}")
    print(f"Sky readout: {sky_readout}  H1/L1 merge: {h1l1_merge}  "
          f"match_params: {hyperparameters.get('match_params', False)}  sky_head_only: {sky_head_only}")
    print(f"Mixup: {use_mixup}")
    print(f"Manifold mixup: {use_manifold_mixup}")
    print(f"SWA: {use_swa} (start epoch: {swa_start_epoch})")
    print(f"Augmentations: {aug_config}")
    print(f"Total samples: {n_samples}")
    print("Mode: TENSOR (preprocessed data, shard streaming)")

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # split shards into val / train
    val_sh = None
    single_file = data_dir / "train.pt"
    if single_file.exists():
        # small dataset mode: load entirely into memory
        print(f"\nLoading single file: {single_file}")
        data = torch.load(str(single_file), weights_only=True)
        all_signals = data['signals']
        all_labels = data['labels']
        all_sh = data.get('sh_coeffs', None)

        n_val = int(len(all_labels) * val_split)
        n_train = len(all_labels) - n_val

        val_signals = all_signals[:n_val]
        val_labels = all_labels[:n_val]
        if all_sh is not None:
            val_sh = all_sh[:n_val]

        # save train split as a temporary shard so fit() streams it from disk
        train_shard = data_dir / "_train_split.pt"
        train_data = {'signals': all_signals[n_val:], 'labels': all_labels[n_val:]}
        if all_sh is not None:
            train_data['sh_coeffs'] = all_sh[n_val:]
        torch.save(train_data, str(train_shard))
        train_shard_paths = [train_shard]
        del all_signals, all_labels, all_sh
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
        val_sh_list = []
        for f in val_shard_paths:
            print(f"  {f.name}")
            data = torch.load(str(f), weights_only=True)
            val_signals_list.append(data['signals'])
            val_labels_list.append(data['labels'])
            if 'sh_coeffs' in data:
                val_sh_list.append(data['sh_coeffs'])
            del data

        val_signals = torch.cat(val_signals_list)
        val_labels = torch.cat(val_labels_list)
        if val_sh_list:
            if len(val_sh_list) != len(val_signals_list):
                raise ValueError(
                    f"Inconsistent shards: {len(val_sh_list)}/{len(val_signals_list)} "
                    f"have sh_coeffs. Regenerate all shards with create_tensors.py."
                )
            val_sh = torch.cat(val_sh_list)
        del val_signals_list, val_labels_list, val_sh_list

        n_val = len(val_labels)
        n_train = n_samples - n_val

        # optional train subsample: keep the first N shards. mmap avoids
        # pulling ~2.5 GB per shard off disk just to count labels.
        max_train_shards = hyperparameters.get('max_train_shards', None)
        if max_train_shards is not None and max_train_shards < len(train_shard_paths):
            train_shard_paths = train_shard_paths[:max_train_shards]
            n_train = 0
            for f in train_shard_paths:
                data = torch.load(str(f), weights_only=True, mmap=True)
                n_train += len(data['labels'])
                del data
            print(f"Train subsampled to {len(train_shard_paths)} shards ({n_train} samples)")

    print(f"Training shards: {len(train_shard_paths)} (streamed from disk)")
    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")

    # sky geometry (Phase 3, Step 2)
    print(f"\nInitializing S2 sky geometry (n_pix={sky_n_pix}, l_max={sky_l_max})...")
    sky_geo = SkyGeometry(n_pix=sky_n_pix, l_max=sky_l_max)
    n_sky_features = sky_geo.n_coeffs
    print(f"  SH coefficients per sample: {n_sky_features}")

    # validation DataLoader (always in memory)
    val_dataset = GWTensorDataset(val_signals, val_labels, sky_geometry=sky_geo, precomputed_sh=val_sh)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == 'cuda')
    )

    # initialize model (readout / merge / match_params selected from config)
    model = build_model(hyperparameters, n_sky_features, device)
    n_params = sum(p.numel() for p in model.parameters())
    cond_params = model.conditioning_param_count() if hasattr(model, 'conditioning_param_count') else 0
    print(f"Model: {type(model).__name__} ({n_params:,} parameters, "
          f"conditioning path {cond_params:,})")

    # split params: decay 2D+ weights (conv/linear matmul), skip biases,
    # norms, and any 1-D scalar (which includes GeM's learnable p).
    # Applying WD to GeM p was the cause of the epoch-5 divergence in the
    # previous runs -- p drifts each step and after a few epochs GeM becomes
    # numerically unstable regardless of the peak LR.
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ],
        lr=learning_rate,
    )
    print(f"Param groups: decay={sum(p.numel() for p in decay_params):,}, "
          f"no_decay={sum(p.numel() for p in no_decay_params):,}")

    # train
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60 + "\n")

    use_amp = hyperparameters.get('use_amp', False)
    clip_grad_norm = hyperparameters.get('clip_grad_norm', None)
    max_train_hours = hyperparameters.get('max_train_hours', None)

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
        max_train_hours=max_train_hours,
        sky_geometry=sky_geo,
        use_mixup=use_mixup,
        use_manifold_mixup=use_manifold_mixup,
        use_swa=use_swa,
        swa_start_epoch=swa_start_epoch,
        aug_config=aug_config,
        label_smoothing=label_smoothing,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )

    print("\nTraining complete.")

    # evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)

    y_val, y_val_proba = evaluate(model, val_loader, device)
    y_val_pred = (y_val_proba >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, y_val_proba)
    cm = compute_confusion_values(y_val_pred, y_val)
    val_metrics = metrics_from_confusion(cm, len(y_val))

    print("\nValidation Set:")
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
    sky_geometry=None,
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lr_mult = (lr_end / lr_start) ** (1.0 / n_steps)
    lrs = []
    losses = []
    best_loss = float('inf')
    step = 0

    for shard_path in shard_paths:
        if step >= n_steps:
            break
        data = torch.load(str(shard_path), weights_only=True)
        precomputed_sh = data.get('sh_coeffs', None)
        dataset = GWTensorDataset(data['signals'], data['labels'], augment=False, sky_geometry=sky_geometry, precomputed_sh=precomputed_sh)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        for X_batch, y_batch, sky_batch in loader:
            if step >= n_steps:
                break

            X_batch = X_batch.to(device)
            sky_batch = sky_batch.to(device)
            y_batch = y_batch.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(X_batch, sky_features=sky_batch)
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
    n_channels = hyperparameters.get('n_channels', 16)
    drop_path_rate = hyperparameters.get('drop_path_rate', 0.0)

    # find shards
    shard_files = sorted(data_dir.glob('shard_*.pt'))
    if not shard_files:
        single = data_dir / 'train.pt'
        if single.exists():
            shard_files = [single]
        else:
            raise FileNotFoundError(f"No data files in {data_dir}")

    # sky geometry
    sky_n_pix = hyperparameters.get('sky_n_pix', 192)
    sky_l_max = hyperparameters.get('sky_l_max', 10)
    sky_geo = SkyGeometry(n_pix=sky_n_pix, l_max=sky_l_max)
    n_sky_features = sky_geo.n_coeffs

    # create model
    model = DIYModel(n_channels=n_channels, dropout_rate=dropout_rate, drop_path_rate=drop_path_rate, n_sky_features=n_sky_features).to(device)
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
        sky_geometry=sky_geo,
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


# =====================================================================
#                         SWEEP + RUN LOGGING
# =====================================================================

# Fixed base for the NeurReps sky-readout sweep. Identical for every config;
# only the sky-readout flags below are overridden per config. Manifold mixup is
# OFF for the whole sweep: it cannot run on the sky-head-only config (no pooled
# CNN features to mix), and keeping the recipe identical across all configs is
# the integrity constraint from ANALYSIS_PLAN. See PHASE2_HANDOFF.md.
SWEEP_BASE = {
    'n_channels': 16,
    'n_samples': 4096,
    'learning_rate': 2e-3,
    'dropout_rate': 0.5,
    'weight_decay': 1e-3,
    'epochs': 20,
    'early_stopping_patience': 4,
    'warmup_epochs': 3,
    'aux_loss_weight': 0.0,
    'use_amp': True,
    'clip_grad_norm': 1.0,
    'drop_path_rate': 0.3,
    'max_train_hours': 2.0,
    # 2 shards = 100k train samples out of 410k. The sweep measures a
    # difference between readouts, not an absolute AUC, so every config pays
    # the same subsampling cost. Full-data runs cost ~8h each and 21 of them
    # do not fit the GPU quota before the deadline.
    'max_train_shards': 2,
    'sky_n_pix': 192,
    'sky_l_max': 10,
    'label_smoothing': 0.0,
    'use_mixup': False,
    'use_manifold_mixup': False,
    'use_swa': True,
    'swa_start_epoch': 14,
    'aug_time_shift': False,
    'aug_noise': True,
    'aug_spectral_dropout': False,
    'aug_channel_shuffle': True,
    'aug_amplitude_scale': False,
    'sky_readout': 'none',
    'h1l1_merge': 'concat',
    'h1l1_pool': 'mean',
    'match_params': False,
    'sky_head_only': False,
}

# Pre-registered configs (ANALYSIS_PLAN section 2). Tier 1: 1-5, Tier 2: 6-7,
# Tier 3: 8. Config 4 is bit-identical to config 2 by construction (match_params
# targets mlp121 @ hidden=128, so matching mlp121 to itself is a no-op) -- the
# logged param counts make the identity explicit.
SWEEP_CONFIGS = {
    1: ('none_concat',             {'sky_readout': 'none',       'h1l1_merge': 'concat',    'match_params': False}),
    2: ('mlp121_concat',           {'sky_readout': 'mlp121',     'h1l1_merge': 'concat',    'match_params': False}),
    3: ('power_concat_match',      {'sky_readout': 'power',      'h1l1_merge': 'concat',    'match_params': True}),
    4: ('mlp121_concat_match',     {'sky_readout': 'mlp121',     'h1l1_merge': 'concat',    'match_params': True}),
    5: ('sky_head_only',           {'sky_head_only': True}),
    6: ('scramble_concat',         {'sky_readout': 'scramble',   'h1l1_merge': 'concat',    'match_params': False}),
    7: ('power_symmetric_match',   {'sky_readout': 'power',      'h1l1_merge': 'symmetric', 'match_params': True}),
    8: ('bispectrum_concat_match', {'sky_readout': 'bispectrum', 'h1l1_merge': 'concat',    'match_params': True}),
}

# Config 4 is deliberately absent: its parameter count was measured identical
# to config 2 (147,954 both), so running it would spend GPU quota reproducing
# config 2 under a different label. The comparator reads as config 3 (power,
# matched) vs config 2 (mlp121). Tiers run in order; Tier 1 gates the rest.
TIER1_RUNS = [(c, s) for s in (0, 1, 2) for c in (1, 2, 3, 5)]
TIER2_RUNS = [(c, s) for s in (0, 1, 2) for c in (6, 7)]
TIER3_RUNS = [(c, s) for s in (0, 1, 2) for c in (8,)]


def read_git_hash() -> str:
    """
    Commit hash for run provenance.

    On Kaggle the code runs from the uploaded dataset with no ``.git``, so the
    hash is stamped into ``src/_version.txt`` before upload. Locally, fall back
    to ``git rev-parse``. Returns ``"unknown"`` if neither is available.
    """
    version_file = Path(__file__).parent / "_version.txt"
    if version_file.exists():
        return version_file.read_text().strip()
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).parent),
            stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def find_tensor_data_dir(output_dir):
    """Locate the directory holding shard_*.pt or train.pt."""
    candidates = [
        Path("D:/Programming/g2net-preprocessed"),
        output_dir / "tensors",
        Path("/kaggle/input/g2net-preprocessed-tfrecords"),
    ]
    for c in candidates:
        if c.exists() and (list(c.glob("shard_*.pt")) or (c / "train.pt").exists()):
            return c
    raise FileNotFoundError(
        "Tensor data not found. Expected shard_*.pt or train.pt in one of:\n" +
        "\n".join(f"  - {p}" for p in candidates)
    )


def load_n_samples(data_dir):
    """Read sample count from metadata.json, else fall back to an estimate."""
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)['n_samples']
    return 560000


def append_run_log(log_path, row: dict):
    """
    Append one run's record as a JSONL line. Never overwrites or drops rows;
    tables and figures are generated from this file, never hand-transcribed.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(row) + "\n")


def build_run_log_row(config_id, seed, hyperparameters, results, wall_clock_s, git_hash):
    """Assemble one sweep-log row from a finished run's results."""
    model = results['model']
    vm = results['val_metrics']
    prec, rec = vm['precision'], vm['recall']
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    total_params = sum(p.numel() for p in model.parameters())
    cond_fn = getattr(model, 'conditioning_param_count', None)
    cond_params = cond_fn() if callable(cond_fn) else 0
    # 'auc' is the final (SWA-averaged) model. Runs can be cut short by the
    # wall-clock guard, which makes that number partly a function of how much
    # GPU time the run got -- log the best epoch too so the comparison rests
    # on a quantity the clock can't move.
    val_auc_hist = [float(a) for a in results['history']['val_auc']]
    best_epoch = max(range(len(val_auc_hist)), key=lambda i: val_auc_hist[i])
    return {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config_id': config_id,
        'seed': seed,
        'git_hash': git_hash,
        'sky_readout': hyperparameters.get('sky_readout'),
        'h1l1_merge': hyperparameters.get('h1l1_merge'),
        'match_params': hyperparameters.get('match_params'),
        'sky_head_only': hyperparameters.get('sky_head_only', False),
        'sky_l_max': hyperparameters.get('sky_l_max'),
        'scramble_seed': hyperparameters.get('scramble_seed'),
        'total_params': total_params,
        'conditioning_params': cond_params,
        'auc': vm['auc'],
        'best_auc': val_auc_hist[best_epoch],
        'best_epoch': best_epoch + 1,
        'epochs_run': len(val_auc_hist),
        'accuracy': vm['accuracy'],
        'precision': prec,
        'recall': rec,
        'specificity': vm['specificity'],
        'f1': f1,
        'final_train_loss': float(results['history']['train_loss'][-1]),
        'final_val_loss': float(results['history']['val_loss'][-1]),
        'n_train': results['n_train'],
        'n_val': results['n_val'],
        'wall_clock_s': round(wall_clock_s, 1),
        'hyperparameters': hyperparameters,
    }


def run_sweep(run_list, log_path=None, kernel_budget_hours=8.5):
    """
    Run a slice of the pre-registered sweep: each (config_id, seed) pair, in order.

    One row per completed run is appended to the JSONL log. Runs are never
    dropped or selected post hoc; a run that would not finish within the
    remaining kernel budget is left unrun (not logged) for a later kernel, so
    two concurrent Kaggle sessions can split the work by passing disjoint lists.

    Parameters
    ----------
    run_list : list[tuple[int, int]]
        (config_id, seed) pairs to run in this kernel.
    log_path : str | Path | None
        JSONL destination. Defaults to ``<output>/sweep_results.jsonl``.
    kernel_budget_hours : float
        Wall-clock ceiling for the whole kernel, below Kaggle's 9h hard limit.
    """
    has_gpu = check_gpu()
    output_dir = get_output_dir()
    data_dir = find_tensor_data_dir(output_dir)
    n_samples = load_n_samples(data_dir)
    if log_path is None:
        log_path = output_dir / "sweep_results.jsonl"
    git_hash = read_git_hash()
    models_dir = output_dir / "models" / "saved"

    print(f"Data directory: {data_dir}")
    print(f"Log: {log_path}")
    print(f"Git hash: {git_hash}")
    print(f"Runs this kernel: {run_list}")

    start = time.time()
    for config_id, seed in run_list:
        elapsed_h = (time.time() - start) / 3600
        remaining = kernel_budget_hours - elapsed_h
        # a run must fit its full training budget plus data loading, or not
        # start at all. Starting one that the clock will cut short produces a
        # row whose AUC reflects the queue rather than the config.
        needed = SWEEP_BASE['max_train_hours'] + 0.35
        if remaining < needed:
            print(f"Kernel budget spent ({elapsed_h:.2f}h used, {remaining:.2f}h left, "
                  f"{needed:.2f}h needed); leaving config {config_id} seed {seed} "
                  f"for a later kernel.")
            break

        name, overrides = SWEEP_CONFIGS[config_id]
        hp = dict(SWEEP_BASE)
        hp.update(overrides)
        hp['seed'] = seed
        hp['scramble_seed'] = seed
        hp['batch_size'] = 64 if has_gpu else 32
        config_label = f"{config_id}_{name}"

        print("\n" + "#"*60)
        print(f"SWEEP RUN  config {config_label}  seed {seed}  (budget left {remaining:.2f}h)")
        print("#"*60)

        run_start = time.time()
        results = train_from_tensors(
            data_dir, n_samples, hyperparameters=hp, val_split=0.2, checkpoint_dir=models_dir,
        )
        wall = time.time() - run_start

        row = build_run_log_row(config_label, seed, hp, results, wall, git_hash)
        append_run_log(log_path, row)
        print(f"Logged {config_label} seed {seed}: AUC {row['auc']:.4f}  "
              f"best {row['best_auc']:.4f} @ epoch {row['best_epoch']}/{row['epochs_run']}  "
              f"cond_params {row['conditioning_params']}  ({wall/3600:.2f}h)")

    print(f"\nSweep slice complete. Log: {log_path}")


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
    # Phase 3 Step 2 + sky-compatible regularization stack.
    # Sky-incompatible aug (time-domain mixup, spectral dropout, time shift)
    # remain off because they break cross-detector phase/timing coherence.
    # New regularizers added: manifold mixup at the pooled-feature level
    # (sky-compatible -- mixing happens after the conv path collapses time),
    # SWA over the last ~15 epochs, drop_path_rate bumped to 0.3.
    # sky_l_max bumped to 10 -> 121 SH coefficients (was 81). Requires
    # regenerating sh_coeffs in the shards via create_tensors.py --add-sky.
    HYPERPARAMETERS = {
        'n_channels': 16,
        'n_samples': 4096,
        'learning_rate': 2e-3,
        'dropout_rate': 0.5,
        'weight_decay': 1e-3,
        'epochs': 50,
        'batch_size': 64 if has_gpu else 32,
        'early_stopping_patience': 10,
        'warmup_epochs': 7,
        'aux_loss_weight': 0.0,
        'use_amp': True,
        'clip_grad_norm': 1.0,
        'drop_path_rate': 0.3,
        'max_train_hours': 8.0,
        'sky_n_pix': 192,
        'sky_l_max': 10,
        'label_smoothing': 0.0,
        'use_mixup': False,
        'use_manifold_mixup': True,
        'use_swa': True,
        'swa_start_epoch': 15,
        'aug_time_shift': False,
        'aug_noise': True,
        'aug_spectral_dropout': False,
        'aug_channel_shuffle': True,
        'aug_amplitude_scale': False,
    }

    # ========== SETUP PATHS ==========
    print("="*60)
    print("SETUP (TENSOR MODE)")
    print("="*60)

    output_dir = get_output_dir()
    print(f"Output directory: {output_dir}")
    print(f"Environment: {'Kaggle' if is_kaggle() else 'Local'}")
    print("Mode: TENSOR (preprocessed data)")

    data_dir = find_tensor_data_dir(output_dir)
    print(f"Data directory: {data_dir}")

    n_samples = load_n_samples(data_dir)
    print(f"Total samples: {n_samples}")

    # ========== LR RANGE TEST MODE ==========
    if mode == 'lr_test':
        run_lr_range_test(data_dir, n_samples, HYPERPARAMETERS)
        return

    models_dir = output_dir / "models" / "saved"
    print(f"Models directory: {models_dir}")

    # ========== TRAIN MODEL ==========
    # checkpoint_dir is /kaggle/working/models/saved on kaggle; the kernel
    # runner commits this directory to the output bundle even when the
    # script exits with an error, so best.pt survives a late-stage crash.
    run_start = time.time()
    results = train_from_tensors(
        data_dir,
        n_samples,
        hyperparameters=HYPERPARAMETERS,
        val_split=0.2,
        checkpoint_dir=models_dir,
    )

    # log this run to the same JSONL the sweep uses, so single runs and sweep
    # runs share one results table
    append_run_log(
        output_dir / "sweep_results.jsonl",
        build_run_log_row(
            'single', HYPERPARAMETERS.get('seed', 0), HYPERPARAMETERS,
            results, time.time() - run_start, read_git_hash(),
        ),
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
    print("\nFinal Validation Metrics:")
    print(f"  Accuracy:    {results['val_metrics']['accuracy']:.4f}")
    print(f"  AUC:         {results['val_metrics']['auc']:.4f}")
    print(f"  Precision:   {results['val_metrics']['precision']:.4f}")
    print(f"  Recall:      {results['val_metrics']['recall']:.4f}")
    print(f"  Specificity: {results['val_metrics']['specificity']:.4f}")
    print("\nFiles saved:")
    print(f"  Weights: {saved_paths['weights'].name}")
    print(f"  Config:  {saved_paths['config'].name}")
    print(f"  Metrics: {saved_paths['metrics'].name}")
    print("\nPlots saved:")
    for name, path in saved_plots.items():
        print(f"  {name}: {path.name}")


if __name__ == "__main__":
    main()
