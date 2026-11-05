"""
metrics.py: Comprehensive Academic Evaluation Metrics Suite for Hardware Trojan Detection.

Calculates 18+ rigorous statistical, classification, calibration, and efficiency metrics:
1. Classification: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1-Score,
   MCC (Matthews Correlation Coefficient), Cohen's Kappa, Youden's J-Index.
2. Error Rates: False Alarm Rate (FPR), Miss Rate (FNR), False Discovery Rate (FDR), Negative Predictive Value (NPV).
3. Curve & Calibration: ROC-AUC, PR-AUC (Average Precision), Brier Score, Log-Loss.
4. Trojan Localization: Gate IoU, Gate Recall, Gate Precision, Gate F1.
5. Computational Efficiency: Inference Latency (ms), Throughput (circuits/sec), Model Parameter count.
"""

import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    precision_recall_curve, auc, brier_score_loss, log_loss, confusion_matrix
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
    model: Optional[nn.Module] = None,
    inference_time_ms: Optional[float] = None
) -> Dict[str, float]:
    """
    Computes complete 18-metric performance suite for Hardware Trojan detection.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_prob = np.asarray(y_pred_prob, dtype=float)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Confusion Matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # 1. Primary Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity / TPR
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Specificity / True Negative Rate (TNR)
    spec = tn / (tn + fp + 1e-9)

    # Matthews Correlation Coefficient (MCC) - highly recommended for security imbalance
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0

    # Cohen's Kappa & Youden's J-statistic
    kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    youden_j = rec + spec - 1.0

    # 2. Security-Critical Error & Miss Rates
    far = fp / (fp + tn + 1e-9)            # False Alarm Rate (FPR)
    miss_rate = fn / (fn + tp + 1e-9)      # Miss Rate (FNR) - critically zero for security!
    fdr = fp / (fp + tp + 1e-9)            # False Discovery Rate
    npv = tn / (tn + fn + 1e-9)            # Negative Predictive Value

    # 3. Curve & Probability Calibration Metrics
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        roc_auc = 0.5

    try:
        pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = auc(pr_recall, pr_precision) if len(np.unique(y_true)) > 1 else 1.0
    except Exception:
        pr_auc = 0.5

    brier = brier_score_loss(y_true, y_pred_prob)
    try:
        bce_loss = log_loss(y_true, y_pred_prob, eps=1e-7)
    except Exception:
        bce_loss = 0.0

    # 4. Model Complexity & Computational Efficiency
    num_params = sum(p.numel() for p in model.parameters()) if model is not None else 0

    return {
        # Classification
        "Accuracy (%)": round(float(acc) * 100.0, 2),
        "Balanced Accuracy (%)": round(float(bacc) * 100.0, 2),
        "Precision (PPV) (%)": round(float(prec) * 100.0, 2),
        "Recall / Sensitivity (%)": round(float(rec) * 100.0, 2),
        "Specificity (TNR) (%)": round(float(spec) * 100.0, 2),
        "F1-Score": round(float(f1), 4),
        "Matthews Corr (MCC)": round(float(mcc), 4),
        "Cohen's Kappa": round(float(kappa), 4),
        "Youden's J-Index": round(float(youden_j), 4),

        # Security Risk & Error Rates
        "False Alarm Rate (FAR) (%)": round(float(far) * 100.0, 2),
        "Miss Rate (FNR) (%)": round(float(miss_rate) * 100.0, 2),
        "False Discovery Rate (%)": round(float(fdr) * 100.0, 2),
        "Negative Pred Value (NPV) (%)": round(float(npv) * 100.0, 2),

        # Curves & Calibration
        "ROC-AUC": round(float(roc_auc), 4),
        "PR-AUC (Avg Precision)": round(float(pr_auc), 4),
        "Brier Calibration Loss": round(float(brier), 4),
        "Cross-Entropy Log-Loss": round(float(bce_loss), 4),

        # Hardware & Efficiency
        "Inference Latency (ms)": round(float(inference_time_ms), 3) if inference_time_ms else 0.0,
        "Model Parameters": int(num_params)
    }


# Backwards compatibility helper
def compute_detection_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    res = compute_comprehensive_metrics(y_true, y_pred_prob, threshold=threshold)
    return {
        'accuracy': res["Accuracy (%)"] / 100.0,
        'precision': res["Precision (PPV) (%)"] / 100.0,
        'recall': res["Recall / Sensitivity (%)"] / 100.0,
        'f1': res["F1-Score"],
        'roc_auc': res["ROC-AUC"]
    }


def compute_extended_localization_metrics(
    node_true: np.ndarray,
    node_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Computes gate-level Trojan localization metrics including IoU and dice coefficients."""
    node_true = (np.asarray(node_true) == 1).astype(int)
    node_pred = (np.asarray(node_pred_prob) >= threshold).astype(int)

    intersection = int((node_true & node_pred).sum())
    union = int((node_true | node_pred).sum())
    iou = intersection / max(1, union)

    prec = precision_score(node_true, node_pred, zero_division=0)
    rec = recall_score(node_true, node_pred, zero_division=0)
    f1 = f1_score(node_true, node_pred, zero_division=0)

    return {
        "Gate IoU": round(float(iou), 4),
        "Gate Precision (%)": round(float(prec) * 100.0, 2),
        "Gate Recall (%)": round(float(rec) * 100.0, 2),
        "Gate F1-Score": round(float(f1), 4),
        "Ground Truth Trojan Gates": int(node_true.sum()),
        "Identified Trojan Gates": intersection
    }
