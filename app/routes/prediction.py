import os
import uuid

import pandas as pd
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename

from app import limiter
from app.services.model_service import model_is_ready, get_feature_columns
from app.services.prediction_service import predict_single, predict_batch_df
from app.services.explanation_service import get_explanation

prediction_bp = Blueprint("prediction", __name__, url_prefix="/api")

ALLOWED_EXTENSIONS = {"csv"}
MAX_BATCH_ROWS = 50000


def _model_unavailable_response():
    return jsonify({"error": "Service Unavailable",
                     "message": "Fraud detection model is not trained yet. "
                                "Run 'python ml/train.py' after placing the dataset."}), 503


@prediction_bp.route("/predict", methods=["POST"])
@limiter.limit("60 per minute")
def predict():
    if not model_is_ready():
        return _model_unavailable_response()

    payload = request.get_json(silent=True)
    if not payload or not isinstance(payload, dict):
        return jsonify({"error": "Bad Request", "message": "Invalid transaction data."}), 400

    try:
        result = predict_single(payload)
    except ValueError as e:
        return jsonify({"error": "Unprocessable Entity", "message": str(e)}), 422
    except Exception:
        current_app.logger.exception("Prediction failed")
        return jsonify({"error": "Internal Server Error",
                         "message": "Fraud detection service temporarily unavailable."}), 500

    include_explanation = request.args.get("explain", "false").lower() == "true"
    if include_explanation:
        explanation = get_explanation(payload)
        result["explanation"] = explanation

    return jsonify(result), 200


@prediction_bp.route("/predict/batch", methods=["POST"])
@limiter.limit("10 per minute")
def predict_batch():
    if not model_is_ready():
        return _model_unavailable_response()

    if "file" not in request.files:
        return jsonify({"error": "Bad Request", "message": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Bad Request", "message": "Empty filename."}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Bad Request", "message": "Only CSV files are supported."}), 400

    safe_name = secure_filename(f"{uuid.uuid4().hex}_{file.filename}")
    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
    file.save(upload_path)

    try:
        df = pd.read_csv(upload_path)
        if len(df) > MAX_BATCH_ROWS:
            return jsonify({"error": "Unprocessable Entity",
                             "message": f"File exceeds max of {MAX_BATCH_ROWS} rows."}), 422

        result_df = predict_batch_df(df)

        export_name = f"batch_predictions_{uuid.uuid4().hex}.csv"
        export_path = os.path.join(current_app.config["EXPORT_FOLDER"], export_name)
        result_df.to_csv(export_path, index=False)

        summary = {
            "total_transactions": int(len(result_df)),
            "potential_fraud": int((result_df["prediction"] == "FRAUD").sum()),
            "fraud_rate": round(float((result_df["prediction"] == "FRAUD").mean()) * 100, 4),
            "average_risk_score": round(float(result_df["risk_score"].mean()), 2),
            "high_risk_count": int((result_df["risk_level"] == "HIGH").sum()),
            "critical_risk_count": int((result_df["risk_level"] == "CRITICAL").sum()),
            "average_amount": round(float(result_df["Amount"].mean()), 2) if "Amount" in result_df else None,
            "total_fraud_amount": round(float(
                result_df.loc[result_df["prediction"] == "FRAUD", "Amount"].sum()
            ), 2) if "Amount" in result_df else None,
            "download_token": export_name,
        }
        preview = result_df.head(50).to_dict(orient="records")
        return jsonify({"summary": summary, "preview": preview}), 200

    except ValueError as e:
        return jsonify({"error": "Unprocessable Entity", "message": str(e)}), 422
    except pd.errors.ParserError:
        return jsonify({"error": "Bad Request", "message": "Malformed CSV file."}), 400
    except Exception:
        current_app.logger.exception("Batch prediction failed")
        return jsonify({"error": "Internal Server Error",
                         "message": "Fraud detection service temporarily unavailable."}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@prediction_bp.route("/batch/download/<token>", methods=["GET"])
def download_batch_results(token):
    safe_token = secure_filename(token)
    path = os.path.join(current_app.config["EXPORT_FOLDER"], safe_token)
    if not os.path.abspath(path).startswith(os.path.abspath(current_app.config["EXPORT_FOLDER"])):
        return jsonify({"error": "Bad Request", "message": "Invalid file token."}), 400
    if not os.path.exists(path):
        return jsonify({"error": "Not Found", "message": "Export not found or expired."}), 404
    return send_file(path, as_attachment=True, download_name="batch_predictions.csv")


@prediction_bp.route("/model/features", methods=["GET"])
def model_features():
    if not model_is_ready():
        return _model_unavailable_response()
    return jsonify({"features": get_feature_columns()}), 200
