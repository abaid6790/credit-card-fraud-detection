"""
Preprocessing pipeline: feature/target split and scaling.

The scaler is always fit on training data only, then reused (never refit)
on validation, test, or live prediction data.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Class"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c != TARGET_COLUMN]


def split_features_target(df: pd.DataFrame):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y, feature_cols


def stratified_split(X: pd.DataFrame, y: pd.Series, val_size: float = 0.15,
                      test_size: float = 0.15, random_state: int = 42):
    """
    70/15/15 stratified split by default. Test set is carved out first and
    must never be touched again until final evaluation.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: StandardScaler, X: pd.DataFrame) -> pd.DataFrame:
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)
