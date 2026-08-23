from flask import current_app


def get_predictor():
    return current_app.predictor


def model_is_ready() -> bool:
    return getattr(current_app, "model_ready", False)


def get_metadata() -> dict:
    predictor = get_predictor()
    if predictor and predictor.is_loaded():
        return predictor.metadata
    return {}


def get_feature_columns() -> list[str]:
    predictor = get_predictor()
    if predictor and predictor.is_loaded():
        return predictor.feature_columns
    return []
