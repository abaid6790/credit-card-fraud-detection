from ml.explain import explain_prediction, is_shap_available
from app.services.model_service import get_predictor


def get_explanation(transaction: dict):
    """Returns a list of {feature, contribution} or None if unavailable.
    Never raises — explanation is best-effort and must not block prediction."""
    if not is_shap_available():
        return None
    predictor = get_predictor()
    if not predictor or not predictor.is_loaded():
        return None
    try:
        return explain_prediction(
            predictor.model, predictor.scaler, predictor.feature_columns, transaction
        )
    except Exception:
        return None
