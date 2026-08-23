"""
Loads the persisted model/scaler/features and runs predictions.
Used by both the Flask app and offline scripts. Contains no fake logic —
every prediction comes from the actual trained model.
"""
from __future__ import annotations

import json
import os
import joblib
import numpy as np
import pandas as pd

MODELS_DIR = "models"


class ModelNotTrainedError(Exception):
    pass


class FraudPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        model_path = os.path.join(self.models_dir, "fraud_model.pkl")
        scaler_path = os.path.join(self.models_dir, "preprocessing_pipeline.pkl")
        features_path = os.path.join(self.models_dir, "feature_columns.pkl")
        metadata_path = os.path.join(self.models_dir, "model_metadata.json")

        missing = [p for p in (model_path, scaler_path, features_path, metadata_path)
                   if not os.path.exists(p)]
        if missing:
            raise ModelNotTrainedError(
                "Trained model artifacts not found. Run 'python ml/train.py' first "
                f"after placing data/creditcard.csv. Missing: {missing}"
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_columns = joblib.load(features_path)
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        self._loaded = True
        return self

    def get_threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    def validate_input(self, transaction: dict) -> list[str]:
        """Return a list of validation error strings (empty = valid)."""
        errors = []
        missing = [c for c in self.feature_columns if c not in transaction]
        if missing:
            errors.append(f"Missing required feature(s): {missing}")
        for k, v in transaction.items():
            if k in self.feature_columns:
                try:
                    float(v)
                except (TypeError, ValueError):
                    errors.append(f"Feature '{k}' must be numeric, got: {v!r}")
        return errors

    def _to_frame(self, transaction: dict) -> pd.DataFrame:
        row = {c: float(transaction[c]) for c in self.feature_columns}
        return pd.DataFrame([row], columns=self.feature_columns)

    def predict_one(self, transaction: dict) -> dict:
        if not self._loaded:
            self.load()
        errors = self.validate_input(transaction)
        if errors:
            raise ValueError("; ".join(errors))

        X = self._to_frame(transaction)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_columns)
        proba = float(self.model.predict_proba(X_scaled)[0, 1])
        threshold = self.get_threshold()
        is_fraud = proba >= threshold

        from ml.risk import probability_to_risk  # local import avoids cycles
        risk_score, risk_level = probability_to_risk(proba)

        return {
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "fraud_probability": round(proba, 4),
            "risk_score": risk_score,
            "risk_level": risk_level,
            "threshold_used": threshold,
            "model_name": self.metadata.get("model_name"),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._loaded:
            self.load()
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s) in uploaded file: {missing}")

        X = df[self.feature_columns].copy()
        for c in self.feature_columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        if X.isna().any().any():
            bad_rows = X[X.isna().any(axis=1)].index.tolist()
            raise ValueError(f"Non-numeric or missing values found in rows: {bad_rows[:20]}")

        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.feature_columns)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        threshold = self.get_threshold()

        from ml.risk import probability_to_risk
        risk_pairs = [probability_to_risk(p) for p in probs]

        result = df.copy()
        result["fraud_probability"] = np.round(probs, 4)
        result["prediction"] = np.where(probs >= threshold, "FRAUD", "LEGITIMATE")
        result["risk_score"] = [rp[0] for rp in risk_pairs]
        result["risk_level"] = [rp[1] for rp in risk_pairs]
        return result
