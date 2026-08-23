"""
Class imbalance handling.

IMPORTANT: These techniques must only ever be applied to the TRAINING split.
Never call these functions on validation or test data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_STRATEGIES = ("class_weight", "random_undersample", "smote", "smote_tomek")


def apply_strategy(X_train: pd.DataFrame, y_train: pd.Series, strategy: str, random_state: int = 42):
    """
    Returns (X_resampled, y_resampled, class_weight_dict_or_None).

    - 'class_weight': no resampling; returns original data plus a class_weight
      dict to pass into the model constructor.
    - 'random_undersample': undersamples the majority class.
    - 'smote': oversamples the minority class synthetically.
    - 'smote_tomek': SMOTE oversampling followed by Tomek link cleaning.
    """
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Choose from {SUPPORTED_STRATEGIES}")

    if strategy == "class_weight":
        classes = np.unique(y_train)
        counts = y_train.value_counts()
        total = len(y_train)
        weights = {c: total / (len(classes) * counts[c]) for c in classes}
        return X_train, y_train, weights

    if strategy == "random_undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, None

    if strategy == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, None

    if strategy == "smote_tomek":
        from imblearn.combine import SMOTETomek
        sampler = SMOTETomek(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, None


def class_balance_report(y: pd.Series) -> dict:
    counts = y.value_counts().to_dict()
    total = len(y)
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "ratios": {int(k): round(v / total, 6) for k, v in counts.items()},
        "imbalance_ratio": round(counts.get(0, 0) / counts.get(1, 1), 2) if counts.get(1, 0) else None,
    }
