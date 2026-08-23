"""
Full training pipeline for the fraud detection system.

Run with:
    python ml/train.py

Steps: load -> validate -> clean -> analyze imbalance -> split -> preprocess ->
train candidate models -> evaluate on validation -> optimize threshold on
validation -> select best model -> final unbiased evaluation on test ->
persist model + pipeline + metadata.

The test set is touched exactly once, at the very end, for final metrics only.
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.dataset_loader import load_and_validate, get_class_distribution, DatasetValidationError
from ml.preprocessing import split_features_target, stratified_split, fit_scaler, transform_with_scaler
from ml.imbalance import apply_strategy, class_balance_report
from ml.evaluate import evaluate_predictions
from ml.threshold import optimize_threshold

DATA_PATH = os.path.join("data", "creditcard.csv")
MODELS_DIR = "models"

# ---- Configuration (edit these to change training behavior) ----
IMBALANCE_STRATEGY = "class_weight"   # class_weight | random_undersample | smote | smote_tomek
THRESHOLD_OBJECTIVE = "max_f1"        # max_f1 | prioritize_recall | balanced
PRIMARY_METRIC = "pr_auc"             # used to pick the best model on validation
RANDOM_STATE = 42


def build_models(class_weight_dict):
    """Instantiate all candidate models. class_weight_dict is only applied
    where the imbalance strategy is 'class_weight'; otherwise models train on
    already-resampled data with default weighting."""
    cw = "balanced" if class_weight_dict is not None else None

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight=cw, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight=cw,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=12, class_weight=cw,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBClassifier
        scale_pos_weight = 1.0
        if class_weight_dict is not None and 1 in class_weight_dict and 0 in class_weight_dict:
            scale_pos_weight = class_weight_dict[1] / class_weight_dict[0]
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight if class_weight_dict is not None else 1.0,
            eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
        )
    except ImportError:
        print("WARNING: xgboost not installed, skipping XGBoost model.")

    return models


def main():
    print("=" * 70)
    print("FRAUD DETECTION TRAINING PIPELINE")
    print("=" * 70)

    # 1-3. Load, validate, clean
    print(f"\n[1/9] Loading dataset from {DATA_PATH} ...")
    try:
        df = load_and_validate(DATA_PATH)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except DatasetValidationError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    print(f"Loaded {len(df):,} rows after cleaning.")

    dist = get_class_distribution(df)
    print(f"Class distribution: {dist}")

    # 4. Feature/target split
    print("\n[2/9] Splitting features and target ...")
    X, y, feature_cols = split_features_target(df)
    print(f"{len(feature_cols)} features: {feature_cols[:5]} ... {feature_cols[-2:]}")

    # 5. Train/val/test split (stratified, 70/15/15)
    print("\n[3/9] Creating stratified train/validation/test split (70/15/15) ...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    print(f"Train class balance: {class_balance_report(y_train)}")

    # 6. Feature scaling (fit on train only)
    print("\n[4/9] Fitting StandardScaler on training data only ...")
    scaler = fit_scaler(X_train)
    X_train_scaled = transform_with_scaler(scaler, X_train)
    X_val_scaled = transform_with_scaler(scaler, X_val)
    X_test_scaled = transform_with_scaler(scaler, X_test)

    # 7. Imbalance handling (train only, never val/test)
    print(f"\n[5/9] Applying imbalance strategy: {IMBALANCE_STRATEGY} (train set only) ...")
    X_train_res, y_train_res, class_weights = apply_strategy(
        X_train_scaled, y_train, IMBALANCE_STRATEGY, random_state=RANDOM_STATE
    )
    print(f"Resampled train class balance: {class_balance_report(y_train_res)}")

    # 8. Train + evaluate candidate models on VALIDATION set
    print("\n[6/9] Training candidate models ...")
    models = build_models(class_weights)
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"  Training {name} ...")
        t0 = time.time()
        model.fit(X_train_res, y_train_res)
        elapsed = time.time() - t0

        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        val_metrics = evaluate_predictions(y_val, val_proba, threshold=0.5)
        results[name] = val_metrics
        trained_models[name] = model
        print(f"    done in {elapsed:.1f}s | val PR-AUC={val_metrics['pr_auc']} "
              f"ROC-AUC={val_metrics['roc_auc']} F1={val_metrics['f1']}")

    # 9. Select best model on VALIDATION performance (never test)
    print(f"\n[7/9] Selecting best model by validation {PRIMARY_METRIC} ...")
    best_name = max(results, key=lambda n: results[n][PRIMARY_METRIC])
    best_model = trained_models[best_name]
    print(f"Best model: {best_name} (val {PRIMARY_METRIC}={results[best_name][PRIMARY_METRIC]})")

    # Optimize threshold on VALIDATION probabilities for the best model
    print(f"\n[8/9] Optimizing threshold (objective: {THRESHOLD_OBJECTIVE}) on validation set ...")
    best_val_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    threshold_result = optimize_threshold(y_val, best_val_proba, objective=THRESHOLD_OBJECTIVE)
    chosen_threshold = threshold_result["threshold"]
    print(f"Chosen threshold: {chosen_threshold} -> {threshold_result}")

    # Final unbiased evaluation on TEST set, using the selected model + threshold
    print("\n[9/9] Final evaluation on held-out TEST set ...")
    test_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_proba, threshold=chosen_threshold)
    print(f"Final test metrics: {test_metrics}")

    # Persist everything
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "fraud_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    # Feature importance if supported
    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(sorted(
            zip(feature_cols, best_model.feature_importances_.tolist()),
            key=lambda kv: kv[1], reverse=True,
        ))
    elif hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_[0])
        feature_importance = dict(sorted(
            zip(feature_cols, coefs.tolist()), key=lambda kv: kv[1], reverse=True,
        ))

    metadata = {
        "model_name": best_name,
        "task": "binary_classification",
        "threshold": chosen_threshold,
        "threshold_objective": THRESHOLD_OBJECTIVE,
        "imbalance_strategy": IMBALANCE_STRATEGY,
        "features": feature_cols,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_stats": dist,
        "validation_metrics": results,
        "metrics": test_metrics,  # final, unbiased test-set metrics
        "feature_importance": feature_importance,
        "model_comparison": {
            name: {k: v for k, v in m.items() if k in
                   ("precision", "recall", "f1", "roc_auc", "pr_auc")}
            for name, m in results.items()
        },
    }

    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model, scaler, feature list, and metadata to '{MODELS_DIR}/'.")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
