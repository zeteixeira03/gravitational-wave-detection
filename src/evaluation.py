"""
Evaluation metrics for binary classification.

Standalone functions operating on numpy arrays of predictions and labels.
No model dependency -- call model.predict_proba() first, then pass results here.
"""

from __future__ import annotations

import numpy as np


# ============================================================================================
#                                    confusion matrix
# ============================================================================================

def compute_confusion_values(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Compute confusion matrix counts from binary predictions and labels.

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


def metrics_from_confusion(cm: dict, n_samples: int) -> dict:
    """
    Compute classification metrics from confusion matrix counts.

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
        'f1': float(f1),
    }


# ============================================================================================
#                                    threshold-based metrics
# ============================================================================================

def confusion_matrix(y_proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute confusion matrix from predicted probabilities and labels.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        True binary labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Dictionary containing 'TP', 'TN', 'FP', 'FN' counts.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return compute_confusion_values(y_pred, y_true)


def evaluate_metrics(y_proba: np.ndarray, y_true: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute all classification metrics from predicted probabilities.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        True binary labels.
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Dictionary containing accuracy, precision, recall, specificity, f1.
    """
    cm = confusion_matrix(y_proba, y_true, threshold)
    return metrics_from_confusion(cm, len(y_true))


# ============================================================================================
#                                       curves
# ============================================================================================

def roc_curve(y_proba: np.ndarray, y_true: np.ndarray, n_thresholds: int = 100) -> dict:
    """
    Compute ROC curve data at multiple thresholds.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        True binary labels.
    n_thresholds : int
        Number of threshold points to evaluate.

    Returns
    -------
    dict
        Dictionary containing 'fpr', 'tpr', 'thresholds', 'auc'.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = compute_confusion_values(y_pred, y_true)
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


def precision_recall_curve(y_proba: np.ndarray, y_true: np.ndarray, n_thresholds: int = 100) -> dict:
    """
    Compute precision-recall curve data at multiple thresholds.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities.
    y_true : np.ndarray
        True binary labels.
    n_thresholds : int
        Number of threshold points to evaluate.

    Returns
    -------
    dict
        Dictionary containing 'precision', 'recall', 'thresholds', 'ap'.
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        cm = compute_confusion_values(y_pred, y_true)
        prec = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 1.0
        rec = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
        precision_list.append(prec)
        recall_list.append(rec)

    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    sorted_indices = np.argsort(recall_arr)
    recall_sorted = recall_arr[sorted_indices]
    precision_sorted = precision_arr[sorted_indices]
    ap = np.trapezoid(precision_sorted, recall_sorted)

    return {'precision': precision_arr, 'recall': recall_arr, 'thresholds': thresholds, 'ap': float(ap)}
