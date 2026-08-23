import pandas as pd
from ml.preprocessing import split_features_target, stratified_split, fit_scaler, transform_with_scaler


def make_df(n=200):
    import numpy as np
    rng = np.random.default_rng(0)
    data = {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Time"] = rng.integers(0, 100000, size=n)
    data["Amount"] = rng.uniform(1, 500, size=n)
    data["Class"] = [0] * (n - 10) + [1] * 10
    return pd.DataFrame(data)


def test_split_features_target():
    df = make_df()
    X, y, cols = split_features_target(df)
    assert "Class" not in cols
    assert len(X) == len(y) == len(df)


def test_stratified_split_preserves_ratio():
    df = make_df()
    X, y, _ = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    # Each split should contain at least one fraud case given stratification
    assert y_train.sum() >= 1
    assert y_test.sum() >= 1


def test_scaler_fit_transform_shapes():
    df = make_df()
    X, y, _ = split_features_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    scaler = fit_scaler(X_train)
    scaled = transform_with_scaler(scaler, X_train)
    assert scaled.shape == X_train.shape
