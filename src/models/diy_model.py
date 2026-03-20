"""
Deep residual 1D CNN for G2Net gravitational wave detection.

Phase 3, Step 1: ~10 residual blocks with progressive downsampling.
LIGO H1/L1 share extractor weights; Virgo has a separate extractor.
Residual backbone is shared across all 3 branches.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================================
#                                        POOLING
# ============================================================================================

class GeM(nn.Module):
    """
    Generalized Mean pooling with learnable exponent.

    Computes (mean(x^p))^(1/p) with learnable p (init=3). Interpolates between
    average pooling (p=1) and max pooling (p->inf). Negative activations from
    BatchNorm are clamped to eps. Forces float32 to prevent NaN under AMP.

    Parameters
    ----------
    kernel_size : int
        Pooling window size.
    p : float
        Initial value for the learnable exponent.
    eps : float
        Clamping floor for negative/zero activations.
    """

    def __init__(self, kernel_size: int, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            x = x.clamp(min=self.eps).pow(self.p)
            x = F.avg_pool1d(x, self.kernel_size)
            return x.pow(1.0 / self.p)


class AdaptiveConcatPool1d(nn.Module):
    """Concatenation of adaptive average pooling and adaptive max pooling."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            F.adaptive_avg_pool1d(x, 1),
            F.adaptive_max_pool1d(x, 1),
        ], dim=1)


# ============================================================================================
#                                    RESIDUAL BLOCK
# ============================================================================================

class ResBlock(nn.Module):
    """
    Residual block with two Conv1d layers and optional GeM downsampling (part of ablation study).

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Kernel size for both convolutions.
    downsample_factor : int | None
        Spatial downsampling factor via GeM pooling. None for identity blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, downsample_factor: int | None = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5)
        self.act = nn.SiLU()
        self.pool = GeM(downsample_factor) if downsample_factor else None

        # shortcut: project channels and/or downsample to match main path
        needs_proj = (in_channels != out_channels)
        layers = []
        if needs_proj:
            layers.append(nn.Conv1d(in_channels, out_channels, 1))
            layers.append(nn.BatchNorm1d(out_channels, momentum=0.1, eps=1e-5))
        if downsample_factor:
            layers.append(GeM(downsample_factor))
        self.shortcut = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))         # first conv + BN + activation
        out = self.bn2(self.conv2(out))                 # second conv + BN (no activation yet)
        if self.pool is not None:                       # ablation (GeM)
            out = self.pool(out)                    
        return self.act(out + residual)                 # add residual and apply final activation


# ============================================================================================
#                                      DIY MODEL
# ============================================================================================

class DIYModel(nn.Module):
    """
    Deep residual 1D CNN for binary classification of GW signals.

    Architecture (Phase 3, Step 1):
    - Separate extractors for LIGO (H1/L1 shared) and Virgo
    - 10 residual blocks with GeM downsampling and channel widening
    - AdaptiveConcatPool1d (avg + max) for global pooling
    - 3-layer classifier head
    - Auxiliary per-branch heads for training supervision
    - All outputs are raw logits (no sigmoid); use BCEWithLogitsLoss
    - Optimizer: AdamW (configured externally in model_runs.py)

    Temporal:  4096 -> 2048 (extractor) -> 512 -> 128 -> 32; 
    Channels:  1 -> n -> n -> 2n -> 4n -> 4n

    Input shape: (batch_size, 3, 4096) - 3 detectors, 4096 time samples
    """

    def __init__(self, n_channels: int = 32, dropout_rate: float = 0.5):
        """
        Parameters
        ----------
        n_channels : int
            Base channel width (n). Channels progress as n -> 2n -> 4n.
        dropout_rate : float
            Dropout rate for classifier head.
        """
        super().__init__()
        n = n_channels

        # extractors: Conv(1,n,64) -> BN -> SiLU -> Conv(n,n,64) -> GeM(2)
        # H1/L1 share weights (same instrument type); Virgo is separate
        self.ligo_extractor = nn.Sequential(
            nn.Conv1d(1, n, 64, padding='same'),
            nn.BatchNorm1d(n, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(n, n, 64, padding='same'),
            GeM(2),
        )
        self.virgo_extractor = nn.Sequential(
            nn.Conv1d(1, n, 64, padding='same'),
            nn.BatchNorm1d(n, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Conv1d(n, n, 64, padding='same'),
            GeM(2),
        )

        # residual backbone (shared by all 3 branches)
        # 5 groups x 2 blocks = 10 residual blocks (20 conv layers)
        # (out_channels, kernel_size, downsample_factor)
        # first block in each group gets the downsample; second is identity
        groups = [
            (n,     31, 4),     # group 1: 2048 -> 512
            (n,     31, None),  # group 2: 512 (no downsample)
            (2 * n, 15, 4),     # group 3: 512 -> 128, widen to 2n
            (4 * n,  7, 4),     # group 4: 128 -> 32, widen to 4n
            (4 * n,  7, None),  # group 5: 32 (no downsample)
        ]

        blocks = []
        in_ch = n
        for out_ch, k, ds in groups:
            blocks.append(ResBlock(in_ch, out_ch, k, downsample_factor=ds))
            in_ch = out_ch
            blocks.append(ResBlock(in_ch, out_ch, k))

        self.backbone = nn.Sequential(*blocks)

        # global pooling: (batch, 4n, T) -> (batch, 8n) via concat of avg + max
        self.global_pool = AdaptiveConcatPool1d()

        # auxiliary per-branch classification heads (raw logits)
        self.branch_head = nn.Linear(8 * n, 1)

        # classifier head: 3 branches * 8n = 24n features
        feat_dim = 3 * 8 * n
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, momentum=0.1, eps=1e-5),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

        # kaiming initialization
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming normal initialization to conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Parameters
        ----------
        X : torch.Tensor
            Input of shape (batch_size, 3, 4096).

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]
            Inference: logits (batch_size, 1).
            Training: (logits, [h1_logits, l1_logits, v1_logits]).
        """
        # per-detector extraction
        h1 = self.ligo_extractor(X[:, 0:1, :])
        l1 = self.ligo_extractor(X[:, 1:2, :])
        v1 = self.virgo_extractor(X[:, 2:3, :])

        # shared backbone -> global pool -> (batch, 8n)
        h1_feat = self.global_pool(self.backbone(h1)).squeeze(-1)
        l1_feat = self.global_pool(self.backbone(l1)).squeeze(-1)
        v1_feat = self.global_pool(self.backbone(v1)).squeeze(-1)

        # classify
        combined = torch.cat([h1_feat, l1_feat, v1_feat], dim=-1)
        main_logits = self.classifier(combined)

        if self.training:
            branch_logits = [
                self.branch_head(h1_feat),
                self.branch_head(l1_feat),
                self.branch_head(v1_feat),
            ]
            return main_logits, branch_logits

        return main_logits

    def compute_loss(
        self,
        y_true: torch.Tensor,
        logits: torch.Tensor,
        branch_logits: list[torch.Tensor] | None = None,
        aux_loss_weight: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute total loss: main BCE + weighted auxiliary per-branch BCE.

        All inputs are raw logits (pre-sigmoid). Uses BCEWithLogitsLoss for
        numerical stability.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels of shape (batch_size, 1).
        logits : torch.Tensor
            Main logits of shape (batch_size, 1).
        branch_logits : list[torch.Tensor] | None
            Per-detector logits, each of shape (batch_size, 1).
        aux_loss_weight : float
            Weight for auxiliary branch losses (lambda). 0 disables.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        main_loss = F.binary_cross_entropy_with_logits(logits, y_true)

        if branch_logits is not None and aux_loss_weight > 0:
            aux_loss = sum(
                F.binary_cross_entropy_with_logits(bl, y_true) for bl in branch_logits
            ) / len(branch_logits)
            return main_loss + aux_loss_weight * aux_loss

        return main_loss

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Predict probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, 3, n_time_samples).
        batch_size : int
            Batch size for inference to avoid OOM on large inputs.

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        was_training = self.training
        self.eval()
        device = next(self.parameters()).device

        n_samples = X.shape[0]
        if n_samples <= batch_size:
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            logits = self.forward(X_t)
            result = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            all_predictions = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
                logits = self.forward(X_batch)
                all_predictions.append(torch.sigmoid(logits).cpu().numpy())
            result = np.concatenate(all_predictions, axis=0).flatten()

        if was_training:
            self.train()
        return result

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels for input samples. 1 = BH merger, 0 = not

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features).
        threshold : float
            Classification threshold.

        Returns
        -------
        np.ndarray
            Predicted binary labels of shape (n_samples,).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def _compute_confusion_values(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
        """
        Compute confusion matrix values from predictions and labels.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted binary labels.
        y_true : np.ndarray
            True binary labels.

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts.
        """
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))
        return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

    def confusion_matrix(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Compute confusion matrix components.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        threshold : float
            Classification threshold.

        Returns
        -------
        dict
            Dictionary containing 'TP', 'TN', 'FP', 'FN' counts.
        """
        y_pred = self.predict(X, threshold=threshold)
        return self._compute_confusion_values(y_pred, y)

    def _metrics_from_confusion(self, cm: dict, n_samples: int) -> dict:
        """
        Compute all metrics from confusion matrix values.

        Parameters
        ----------
        cm : dict
            Confusion matrix dict with 'TP', 'TN', 'FP', 'FN' keys.
        n_samples : int
            Total number of samples.

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, specificity, f1.
        """
        TP, TN, FP, FN = cm['TP'], cm['TN'], cm['FP'], cm['FN']

        accuracy = (TP + TN) / n_samples
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1': float(f1)
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
        """
        Evaluate model statistics.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        threshold : float
            Classification threshold.

        Returns
        -------
        dict
            Dictionary containing accuracy, precision, recall, specificity, f1.
        """
        cm = self.confusion_matrix(X, y, threshold=threshold)
        return self._metrics_from_confusion(cm, len(y))

    def roc_curve(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 100) -> dict:
        """
        Compute ROC curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        n_thresholds : int
            Number of threshold points to evaluate.

        Returns
        -------
        dict
            Dictionary containing 'fpr', 'tpr', 'thresholds', 'auc'.
        """
        y_proba = self.predict_proba(X)
        thresholds = np.linspace(0, 1, n_thresholds)

        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = self._compute_confusion_values(y_pred, y)

            tpr = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
            fpr = cm['FP'] / (cm['FP'] + cm['TN']) if (cm['FP'] + cm['TN']) > 0 else 0.0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        fpr_arr = np.array(fpr_list)
        tpr_arr = np.array(tpr_list)

        sorted_indices = np.argsort(fpr_arr)
        fpr_sorted = fpr_arr[sorted_indices]
        tpr_sorted = tpr_arr[sorted_indices]
        auc = np.trapezoid(tpr_sorted, fpr_sorted)

        return {'fpr': fpr_arr, 'tpr': tpr_arr, 'thresholds': thresholds, 'auc': float(auc)}

    def precision_recall_curve(self, X: np.ndarray, y: np.ndarray, n_thresholds: int = 100) -> dict:
        """
        Compute precision-recall curve data at multiple thresholds.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            True labels.
        n_thresholds : int
            Number of threshold points to evaluate.

        Returns
        -------
        dict
            Dictionary containing 'precision', 'recall', 'thresholds', 'ap'.
        """
        y_proba = self.predict_proba(X)
        thresholds = np.linspace(0, 1, n_thresholds)

        precision_list = []
        recall_list = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            cm = self._compute_confusion_values(y_pred, y)

            precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 1.0
            recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0

            precision_list.append(precision)
            recall_list.append(recall)

        precision_arr = np.array(precision_list)
        recall_arr = np.array(recall_list)

        sorted_indices = np.argsort(recall_arr)
        recall_sorted = recall_arr[sorted_indices]
        precision_sorted = precision_arr[sorted_indices]
        ap = np.trapezoid(precision_sorted, recall_sorted)

        return {'precision': precision_arr, 'recall': recall_arr, 'thresholds': thresholds, 'ap': float(ap)}

    def save_weights(self, filepath: str) -> None:
        """
        Save model weights to a file.

        Parameters
        ----------
        filepath : str
            Path to save weights (.pt file).
        """
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath: str) -> None:
        """
        Load model weights from a file.

        Parameters
        ----------
        filepath : str
            Path to load weights from (.pt file).
        """
        state_dict = torch.load(filepath, map_location='cpu', weights_only=True)
        self.load_state_dict(state_dict)
