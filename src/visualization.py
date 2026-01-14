"""
Visualization utilities for model performance assessment.

Provides plotting functions for:
- Learning curves (loss, accuracy, metrics over epochs)
- ROC curve
- Precision-Recall curve
- Confusion matrix heatmap
- Prediction distribution histogram
"""

from __future__ import annotations
from typing import Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(
    history: Dict[str, list],
    metrics: Optional[list] = None,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning curves from training history.

    Parameters
    ----------
    history : dict
        Training history dict from model.fit() containing keys like
        'train_loss', 'val_loss', 'train_acc', 'val_acc', etc.
    metrics : list, optional
        List of metric names to plot. If None, plots all available.
        Options: 'loss', 'acc', 'prec', 'recall', 'spec', 'f1'
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # determine which metrics to plot
    available_metrics = set()
    for key in history.keys():
        # extract metric name from 'train_loss' -> 'loss'
        if key.startswith('train_'):
            available_metrics.add(key.replace('train_', ''))

    if metrics is None:
        metrics = sorted(available_metrics)
    else:
        metrics = [m for m in metrics if m in available_metrics]

    n_metrics = len(metrics)
    if n_metrics == 0:
        raise ValueError("No valid metrics found in history")

    # create subplots
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    epochs = range(1, len(history[f'train_{metrics[0]}']) + 1)

    metric_labels = {
        'loss': 'Loss',
        'acc': 'Accuracy',
        'prec': 'Precision',
        'recall': 'Recall',
        'spec': 'Specificity',
        'f1': 'F1 Score'
    }

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2)
        if val_key in history:
            ax.plot(epochs, history[val_key], 'r-', label='Validation', linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_labels.get(metric, metric.capitalize()))
        ax.set_title(f'{metric_labels.get(metric, metric.capitalize())} over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_roc_curve(
    roc_data: Dict[str, Any],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve from model.roc_curve() output.

    Parameters
    ----------
    roc_data : dict
        Output from model.roc_curve() containing 'fpr', 'tpr', 'auc'
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    auc = roc_data['auc']

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_precision_recall_curve(
    pr_data: Dict[str, Any],
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve from model.precision_recall_curve() output.

    Parameters
    ----------
    pr_data : dict
        Output from model.precision_recall_curve() containing
        'precision', 'recall', 'ap'
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    precision = pr_data['precision']
    recall = pr_data['recall']
    ap = pr_data['ap']

    # sort by recall for proper curve display
    sorted_idx = np.argsort(recall)
    recall_sorted = recall[sorted_idx]
    precision_sorted = precision[sorted_idx]

    ax.plot(recall_sorted, precision_sorted, 'b-', linewidth=2,
            label=f'PR Curve (AP = {ap:.4f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm_data: Dict[str, int],
    figsize: tuple = (8, 6),
    normalize: bool = False,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap from model.confusion_matrix() output.

    Parameters
    ----------
    cm_data : dict
        Output from model.confusion_matrix() containing 'TP', 'TN', 'FP', 'FN'
    figsize : tuple
        Figure size (width, height)
    normalize : bool
        If True, normalize values to percentages
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    TP, TN, FP, FN = cm_data['TP'], cm_data['TN'], cm_data['FP'], cm_data['FN']

    # confusion matrix layout: [[TN, FP], [FN, TP]]
    # rows = actual, cols = predicted
    cm = np.array([[TN, FP], [FN, TP]])

    if normalize:
        cm_display = cm.astype(float) / cm.sum() * 100
        fmt = '.1f'
        title = 'Confusion Matrix (Normalized %)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    im = ax.imshow(cm_display, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)

    # labels
    classes = ['Noise (0)', 'Signal (1)']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    # annotate cells
    thresh = cm_display.max() / 2
    for i in range(2):
        for j in range(2):
            if normalize:
                text = f'{cm_display[i, j]:.1f}%\n({cm[i, j]})'
            else:
                text = f'{cm_display[i, j]}'
            ax.text(j, i, text, ha='center', va='center',
                    color='white' if cm_display[i, j] > thresh else 'black',
                    fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_distribution(
    y_proba: np.ndarray,
    y_true: np.ndarray,
    bins: int = 50,
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of predicted probabilities for each class.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities from model.predict_proba()
    y_true : np.ndarray
        True labels
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # separate predictions by true class
    proba_noise = y_proba[y_true == 0]
    proba_signal = y_proba[y_true == 1]

    ax.hist(proba_noise, bins=bins, alpha=0.6, label=f'Noise (n={len(proba_noise)})',
            color='blue', density=True)
    ax.hist(proba_signal, bins=bins, alpha=0.6, label=f'Signal (n={len(proba_signal)})',
            color='red', density=True)

    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=1, label='Threshold (0.5)')

    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    history: Optional[Dict[str, list]] = None,
    figsize: tuple = (16, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard with all performance plots.

    Parameters
    ----------
    model : DIYModel
        Trained model instance with evaluate methods
    X : np.ndarray
        Input features
    y : np.ndarray
        True labels
    history : dict, optional
        Training history for learning curves. If None, skips learning curves.
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # compute all metrics
    roc_data = model.roc_curve(X, y)
    pr_data = model.precision_recall_curve(X, y)
    cm_data = model.confusion_matrix(X, y)
    y_proba = model.predict_proba(X)
    metrics = model.evaluate(X, y)

    # determine layout
    has_history = history is not None and len(history) > 0
    n_plots = 5 if has_history else 4

    if has_history:
        fig = plt.figure(figsize=figsize)
        # 3 rows: learning curves (top, spans 2 cols), then 2x2 grid below
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
        ax_learning = fig.add_subplot(gs[0, :])
        ax_roc = fig.add_subplot(gs[1, 0])
        ax_pr = fig.add_subplot(gs[1, 1])
        ax_cm = fig.add_subplot(gs[2, 0])
        ax_dist = fig.add_subplot(gs[2, 1])
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_roc, ax_pr = axes[0]
        ax_cm, ax_dist = axes[1]

    # learning curves (if history provided)
    if has_history:
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        epochs = range(1, len(train_loss) + 1)

        ax_learning.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
        if val_loss:
            ax_learning.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
        ax_learning.set_xlabel('Epoch')
        ax_learning.set_ylabel('Loss')
        ax_learning.set_title('Learning Curves')
        ax_learning.legend()
        ax_learning.grid(True, alpha=0.3)

    # ROC curve
    ax_roc.plot(roc_data['fpr'], roc_data['tpr'], 'b-', linewidth=2,
                label=f'AUC = {roc_data["auc"]:.4f}')
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc='lower right')
    ax_roc.grid(True, alpha=0.3)

    # PR curve
    sorted_idx = np.argsort(pr_data['recall'])
    ax_pr.plot(pr_data['recall'][sorted_idx], pr_data['precision'][sorted_idx],
               'b-', linewidth=2, label=f'AP = {pr_data["ap"]:.4f}')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(loc='lower left')
    ax_pr.grid(True, alpha=0.3)

    # confusion matrix
    cm = np.array([[cm_data['TN'], cm_data['FP']],
                   [cm_data['FN'], cm_data['TP']]])
    im = ax_cm.imshow(cm, cmap='Blues')
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Noise', 'Signal'])
    ax_cm.set_yticklabels(['Noise', 'Signal'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                      color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=12)

    # prediction distribution
    proba_noise = y_proba[y == 0]
    proba_signal = y_proba[y == 1]
    ax_dist.hist(proba_noise, bins=50, alpha=0.6, label='Noise', color='blue', density=True)
    ax_dist.hist(proba_signal, bins=50, alpha=0.6, label='Signal', color='red', density=True)
    ax_dist.axvline(x=0.5, color='black', linestyle='--', linewidth=1)
    ax_dist.set_xlabel('Predicted Probability')
    ax_dist.set_ylabel('Density')
    ax_dist.set_title('Prediction Distribution')
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    # add metrics summary as text
    metrics_text = (f"Accuracy: {metrics['accuracy']:.4f}  |  "
                   f"Precision: {metrics['precision']:.4f}  |  "
                   f"Recall: {metrics['recall']:.4f}  |  "
                   f"F1: {metrics['f1']:.4f}")
    fig.suptitle(metrics_text, fontsize=11, y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
