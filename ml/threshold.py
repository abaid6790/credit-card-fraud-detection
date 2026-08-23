"""
Classification threshold optimization.

Thresholds are selected using the VALIDATION set only. The test set is used
solely for final, unbiased evaluation after the threshold is locked in.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

SUPPORTED_OBJECTIVES = ("max_f1", "prioritize_recall", "balanced")


def optimize_threshold(y_true, y_proba, objective: str = "max_f1",
                        min_recall: float = 0.8) -> dict:
    """
    Sweep thresholds and pick the best one according to `objective`:
      - 'max_f1': threshold that maximizes F1 score.
      - 'prioritize_recall': highest-precision threshold subject to
        recall >= min_recall.
      - 'balanced': threshold minimizing |precision - recall|.

    Returns dict with the chosen threshold and the metrics at that threshold.
    """
    if objective not in SUPPORTED_OBJECTIVES:
        raise ValueError(f"Unknown objective '{objective}'. Choose from {SUPPORTED_OBJECTIVES}")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns len(thresholds) == len(precisions) - 1
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    if len(thresholds) == 0:
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if objective == "max_f1":
        f1s = np.where((precisions + recalls) > 0,
                        2 * precisions * recalls / (precisions + recalls + 1e-12), 0)
        best_idx = int(np.argmax(f1s))
    elif objective == "prioritize_recall":
        eligible = np.where(recalls >= min_recall)[0]
        if len(eligible) == 0:
            best_idx = int(np.argmax(recalls))
        else:
            best_idx = eligible[int(np.argmax(precisions[eligible]))]
    else:  # balanced
        diff = np.abs(precisions - recalls)
        best_idx = int(np.argmin(diff))

    best_threshold = float(thresholds[best_idx])
    y_pred = (y_proba >= best_threshold).astype(int)

    return {
        "threshold": round(best_threshold, 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "objective": objective,
    }
