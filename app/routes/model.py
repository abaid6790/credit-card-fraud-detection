from flask import Blueprint, jsonify

from app.services.model_service import get_metadata, model_is_ready

model_bp = Blueprint("model_api", __name__, url_prefix="/api")


@model_bp.route("/model")
def api_model():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable", "message": "Model not trained yet."}), 503
    metadata = get_metadata()
    return jsonify({
        "model_name": metadata.get("model_name"),
        "threshold": metadata.get("threshold"),
        "threshold_objective": metadata.get("threshold_objective"),
        "imbalance_strategy": metadata.get("imbalance_strategy"),
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics"),
        "feature_importance": metadata.get("feature_importance"),
    })


@model_bp.route("/model/comparison")
def api_model_comparison():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable", "message": "Model not trained yet."}), 503
    metadata = get_metadata()
    return jsonify(metadata.get("model_comparison", {}))
