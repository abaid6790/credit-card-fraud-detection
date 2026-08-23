from flask import Blueprint, render_template, jsonify

from app import get_db
from app.services.analytics_service import prediction_history_stats
from app.services.model_service import model_is_ready

monitoring_bp = Blueprint("monitoring", __name__)


def _recent_high_risk(limit=20):
    db = get_db()
    rows = db.execute(
        """
        SELECT id, timestamp, prediction, fraud_probability, risk_score, risk_level, amount
        FROM transactions
        WHERE risk_level IN ('HIGH', 'CRITICAL')
        ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


@monitoring_bp.route("/monitoring")
def monitoring_page():
    stats = prediction_history_stats()
    recent = _recent_high_risk()
    return render_template("monitoring.html", stats=stats, recent=recent, ready=model_is_ready())


@monitoring_bp.route("/api/alerts")
def api_alerts():
    recent = _recent_high_risk(limit=50)
    return jsonify({"alerts": recent, "count": len(recent)})
