import pandas as pd
import pytest
from ml.dataset_loader import validate_dataset, DatasetValidationError, REQUIRED_COLUMNS


def _valid_df(n=50):
    import numpy as np
    rng = np.random.default_rng(1)
    data = {c: rng.normal(size=n) for c in REQUIRED_COLUMNS if c not in ("Class",)}
    data["Class"] = [0] * (n - 5) + [1] * 5
    return pd.DataFrame(data)


def test_valid_dataset_passes():
    df = _valid_df()
    validate_dataset(df)  # should not raise


def test_missing_column_raises():
    df = _valid_df().drop(columns=["V1"])
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_unexpected_column_raises():
    df = _valid_df()
    df["ExtraColumn"] = 1
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_nan_values_raise():
    df = _valid_df()
    df.loc[0, "Amount"] = None
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)


def test_invalid_class_values_raise():
    df = _valid_df()
    df.loc[0, "Class"] = 5
    with pytest.raises(DatasetValidationError):
        validate_dataset(df)
