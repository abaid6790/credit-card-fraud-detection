"""
Explainability via SHAP. If SHAP or the model type isn't supported, the app
must keep working — callers should treat a None/empty result as
"explanation unavailable", never as an error that breaks prediction.
"""
from __future__ import annotations

import pandas as pd

_explainer_cache = {}


def is_shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except ImportError:
        return False


def explain_prediction(model, scaler, feature_columns: list[str], transaction: dict, top_n: int = 8):
    """
    Returns a list of {feature, contribution} dicts sorted by absolute
    contribution, or None if explanation isn't available for this model.
    """
    if not is_shap_available():
        return None

    import shap

    model_key = id(model)
    if model_key not in _explainer_cache:
        try:
            _explainer_cache[model_key] = shap.TreeExplainer(model)
        except Exception:
            try:
                _explainer_cache[model_key] = shap.Explainer(model)
            except Exception:
                return None

    explainer = _explainer_cache[model_key]

    row = {c: float(transaction[c]) for c in feature_columns}
    X = pd.DataFrame([row], columns=feature_columns)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_columns)

    try:
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(shap_values, list):
            # binary classifiers: take the "fraud" class (index 1) if present
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            values = shap_values[0]
            if hasattr(values, "ndim") and values.ndim > 1:
                values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
    except Exception:
        return None

    contributions = [
        {"feature": f, "contribution": round(float(v), 4)}
        for f, v in zip(feature_columns, values)
    ]
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)
    return contributions[:top_n]
