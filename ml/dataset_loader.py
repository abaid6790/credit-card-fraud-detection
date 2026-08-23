"""
Dataset loading and validation for the Credit Card Fraud Detection dataset.

No fake data, no silent fallbacks: if the dataset is missing or malformed,
this module raises a clear error instead of guessing.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
    "Class",
]


class DatasetValidationError(Exception):
    """Raised when the dataset does not match the expected structure."""


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV from disk. Raises FileNotFoundError with a clear message."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it "
            "at that path (see data/README.md)."
        )
    df = pd.read_csv(csv_path)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate dataset structure and quality. Raises DatasetValidationError on
    any problem so the training pipeline fails loudly rather than training on
    bad data.
    """
    errors = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    unexpected_cols = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if unexpected_cols:
        errors.append(f"Unexpected columns found: {unexpected_cols}")

    if errors:
        # Can't safely continue checks below if columns are wrong.
        raise DatasetValidationError("; ".join(errors))

    if df.empty:
        raise DatasetValidationError("Dataset is empty.")

    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        cols_with_nan = nan_counts[nan_counts > 0].to_dict()
        errors.append(f"NaN values found: {cols_with_nan}")

    numeric_df = df.select_dtypes(include=[np.number])
    inf_mask = np.isinf(numeric_df.to_numpy()).any(axis=0)
    inf_cols = numeric_df.columns[inf_mask].tolist()
    if inf_cols:
        errors.append(f"Infinite values found in columns: {inf_cols}")

    # NOTE: duplicate rows are intentionally NOT treated as a validation
    # failure. This dataset (and real transaction data generally) is
    # expected to contain some exact-duplicate rows; they are removed by
    # clean_dataset() as a normal cleaning step, not a structural defect.
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        print(f"[validate_dataset] Note: {dup_count} duplicate rows found; "
              "these will be removed during cleaning.")

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        errors.append(f"Constant (zero-variance) columns found: {constant_cols}")

    invalid_classes = set(df["Class"].unique()) - {0, 1}
    if invalid_classes:
        errors.append(f"Class column contains invalid values: {invalid_classes}")

    if errors:
        raise DatasetValidationError("Dataset validation failed:\n- " + "\n- ".join(errors))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply safe, well-understood cleaning steps:
    - drop exact duplicate rows
    - drop rows with NaNs (should be none if validate_dataset passed on raw data,
      but this is defensive for re-use of this function elsewhere)
    """
    cleaned = df.drop_duplicates().reset_index(drop=True)
    cleaned = cleaned.dropna().reset_index(drop=True)
    return cleaned


def get_class_distribution(df: pd.DataFrame) -> dict:
    total = len(df)
    fraud = int((df["Class"] == 1).sum())
    legit = int((df["Class"] == 0).sum())
    return {
        "total_transactions": total,
        "fraud_transactions": fraud,
        "legitimate_transactions": legit,
        "fraud_percentage": round((fraud / total) * 100, 4) if total else 0.0,
    }


def load_and_validate(csv_path: str) -> pd.DataFrame:
    """Convenience wrapper: load, validate, clean."""
    df = load_dataset(csv_path)
    validate_dataset(df)
    df = clean_dataset(df)
    return df