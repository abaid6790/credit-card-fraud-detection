from app import get_db
from app.services.model_service import get_metadata


def prediction_history_stats() -> dict:
    """All numbers here come from the transactions table — real logged
    predictions, not hardcoded figures."""
    db = get_db()
    row = db.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN prediction = 'FRAUD' THEN 1 ELSE 0 END) AS fraud_count,
            AVG(risk_score) AS avg_risk_score,
            AVG(amount) AS avg_amount,
            SUM(CASE WHEN prediction = 'FRAUD' THEN amount ELSE 0 END) AS fraud_amount,
            SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS high_risk_count,
            SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END) AS critical_count
        FROM transactions
        """
    ).fetchone()

    total = row["total"] or 0
    fraud_count = row["fraud_count"] or 0

    return {
        "total_predictions": total,
        "fraud_predictions": fraud_count,
        "fraud_rate": round((fraud_count / total) * 100, 4) if total else 0.0,
        "average_risk_score": round(row["avg_risk_score"], 2) if row["avg_risk_score"] else 0.0,
        "average_amount": round(row["avg_amount"], 2) if row["avg_amount"] else 0.0,
        "total_fraud_amount": round(row["fraud_amount"], 2) if row["fraud_amount"] else 0.0,
        "high_risk_count": row["high_risk_count"] or 0,
        "critical_count": row["critical_count"] or 0,
    }


def risk_level_distribution() -> dict:
    db = get_db()
    rows = db.execute(
        "SELECT risk_level, COUNT(*) as cnt FROM transactions GROUP BY risk_level"
    ).fetchall()
    return {r["risk_level"]: r["cnt"] for r in rows}


def dashboard_summary() -> dict:
    """Combines dataset-level metadata (from training) with live prediction
    history (from the database)."""
    metadata = get_metadata()
    history = prediction_history_stats()
    dataset_stats = metadata.get("dataset_stats", {})
    metrics = metadata.get("metrics", {})

    return {
        "dataset": dataset_stats,
        "model_metrics": metrics,
        "prediction_history": history,
        "risk_distribution": risk_level_distribution(),
        "model_name": metadata.get("model_name"),
        "threshold": metadata.get("threshold"),
    }
