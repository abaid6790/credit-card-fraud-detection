import json
from datetime import datetime, timezone

from app import get_db
from app.services.model_service import get_predictor


def log_prediction(result: dict, transaction: dict) -> int:
    """Store a prediction in SQLite. Only feature values are stored (no PII),
    and the raw payload is kept minimal per the privacy requirements."""
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO transactions
            (timestamp, prediction, fraud_probability, risk_score, risk_level,
             amount, model_name, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            result["prediction"],
            result["fraud_probability"],
            result["risk_score"],
            result["risk_level"],
            transaction.get("Amount"),
            result.get("model_name"),
            json.dumps(transaction),
        ),
    )
    db.commit()
    return cur.lastrowid


def predict_single(transaction: dict) -> dict:
    predictor = get_predictor()
    result = predictor.predict_one(transaction)
    row_id = log_prediction(result, transaction)
    result["id"] = row_id
    return result


def predict_batch_df(df):
    predictor = get_predictor()
    result_df = predictor.predict_batch(df)
    return result_df
