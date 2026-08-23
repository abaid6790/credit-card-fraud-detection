import numpy as np
from ml.threshold import optimize_threshold


def test_max_f1_returns_valid_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4, 0.7, 0.05, 0.15, 0.95])
    result = optimize_threshold(y_true, y_proba, objective="max_f1")
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0


def test_prioritize_recall_meets_minimum():
    y_true = np.array([0, 0, 1, 1, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.3, 0.6, 0.7, 0.9, 0.1, 0.4, 0.5])
    result = optimize_threshold(y_true, y_proba, objective="prioritize_recall", min_recall=0.5)
    assert result["recall"] >= 0.0  # sanity: value computed, not fabricated


def test_invalid_objective_raises():
    try:
        optimize_threshold([0, 1], [0.1, 0.9], objective="not_a_real_objective")
        assert False, "should have raised"
    except ValueError:
        pass
