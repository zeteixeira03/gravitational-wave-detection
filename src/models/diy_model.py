"""
1D CNN Model for G2Net gravitational wave detection.

Custom PyTorch implementation using 1D convolutions on whitened signals.
This architecture processes the 3 detector signals through shared conv layers,
then concatenates features for classification.
"""

from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================================
#                                      GEM POOLING
# ============================================================================================

class GeMPool1d(nn.Module):
    """Generalized Mean Pooling over the time dimension."""

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor([p]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, channels, time).

        Returns
        -------
        torch.Tensor
            Pooled output of shape (batch, channels).
        """
        p = self.p.clamp(1.0, 10.0)
        eps = 1e-6
        x = x.clamp(min=eps)
        return x.pow(p).mean(dim=2).pow(1.0 / p)


# ============================================================================================
#                                      DIY MODEL
# ============================================================================================

class DIYModel(nn.Module):
    """
    1D Convolutional Neural Network for binary classification of GW signals.

    Architecture:
    - Shared 1D conv layers process each detector independently
    - Features concatenated and passed through dense classifier
    - Loss: Binary Cross-Entropy
    - Optimizer: AdamW (configured externally in model_runs.py)

    Input shape: (batch_size, 3, 4096) - 3 detectors, 4096 time samples
    """

    def __init__(self, n_samples: int = 4096, dropout_rate: float = 0.5):
        """
        Initialize the DIY 1D CNN model.

        Parameters
        ----------
        n_samples : int
            Number of time samples per detector (4096 for 2s at 2048Hz).
        dropout_rate : float
            Dropout rate for regularization.
        """
        super().__init__()
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

        # convolutional layer parameters: (filters, kernel_size, pool_size)
        self.conv_config = [
            (32, 64, 4),   # large kernel to capture low-freq patterns
            (64, 32, 4),
            (128, 16, 4),
            (256, 8, 4),
        ]

        # shared conv blocks (process each detector independently)
        self.conv_blocks = nn.ModuleList()
        in_ch = 1
        for filters, kernel_size, pool_size in self.conv_config:
            block = nn.Sequential(
                nn.Conv1d(in_ch, filters, kernel_size, padding='same'),
                nn.BatchNorm1d(filters, momentum=0.1, eps=1e-5),
                nn.SiLU(),
                nn.MaxPool1d(pool_size),
            )
            self.conv_blocks.append(block)
            in_ch = filters

        # GeM pooling
        self.gem_pool = GeMPool1d(p=3.0)

        # classifier head: 256 features per detector * 3 detectors = 768
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64, momentum=0.1, eps=1e-5),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # kaiming initialization
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming normal initialization to conv and linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _process_detector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process a single detector through conv layers.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, time_steps).

        Returns
        -------
        torch.Tensor
            Features of shape (batch, 256).
        """
        # (batch, time) -> (batch, 1, time) for Conv1d
        x = x.unsqueeze(1)

        for block in self.conv_blocks:
            x = block(x)

        # GeM pooling: (batch, 256, reduced_time) -> (batch, 256)
        x = self.gem_pool(x)
        return x

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, 3, n_samples).

        Returns
        -------
        torch.Tensor
            Output predictions of shape (batch_size, 1) with sigmoid applied.
        """
        # split detectors: X shape is (batch, 3, 4096)
        h1 = X[:, 0, :]  # (batch, 4096)
        l1 = X[:, 1, :]
        v1 = X[:, 2, :]

        # process each detector through shared conv layers
        h1_feat = self._process_detector(h1)  # (batch, 256)
        l1_feat = self._process_detector(l1)
        v1_feat = self._process_detector(v1)

        # concatenate features from all detectors
        combined = torch.cat([h1_feat, l1_feat, v1_feat], dim=-1)  # (batch, 768)

        # classifier head
        return self.classifier(combined)

    def compute_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        y_true : torch.Tensor
            True labels of shape (batch_size, 1).
        y_pred : torch.Tensor
            Predicted probabilities of shape (batch_size, 1).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        epsilon = 1e-7
        y_pred = y_pred.clamp(epsilon, 1 - epsilon)
        bce = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        return bce.mean()

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
            predictions = self.forward(X_t)
            result = predictions.cpu().numpy().flatten()
        else:
            all_predictions = []
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = torch.tensor(X[start_idx:end_idx], dtype=torch.float32, device=device)
                batch_pred = self.forward(X_batch)
                all_predictions.append(batch_pred.cpu().numpy())
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
