from flask import Blueprint, jsonify

from app.services.analytics_service import dashboard_summary
from app.services.model_service import model_is_ready

analytics_bp = Blueprint("analytics", __name__, url_prefix="/api")


@analytics_bp.route("/dashboard")
def api_dashboard():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable",
                         "message": "Model not trained yet."}), 503
    return jsonify(dashboard_summary())


@analytics_bp.route("/analytics")
def api_analytics():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable",
                         "message": "Model not trained yet."}), 503
    return jsonify(dashboard_summary())
