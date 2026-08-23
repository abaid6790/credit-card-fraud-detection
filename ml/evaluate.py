"""
Model evaluation metrics. No accuracy-only shortcuts: fraud detection is
evaluated on precision, recall, F1, ROC-AUC, and PR-AUC, plus a full
confusion matrix breakdown.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve,
)


def evaluate_predictions(y_true, y_proba, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positive_rate": round(float(fpr), 4),
        "false_negative_rate": round(float(fnr), 4),
        "specificity": round(float(specificity), 4),
    }


def roc_curve_data(y_true, y_proba) -> dict:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4)}


def pr_curve_data(y_true, y_proba) -> dict:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    return {"precision": precision.tolist(), "recall": recall.tolist(),
            "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4)}
