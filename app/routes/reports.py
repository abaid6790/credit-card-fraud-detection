from flask import Blueprint, send_file, jsonify

from app.services.report_service import generate_csv_report, generate_pdf_report
from app.services.model_service import model_is_ready

reports_bp = Blueprint("reports", __name__, url_prefix="/api/reports")


@reports_bp.route("/csv", methods=["POST"])
def download_csv_report():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable", "message": "Model not trained yet."}), 503
    path = generate_csv_report()
    return send_file(path, as_attachment=True, download_name="fraud_report.csv")


@reports_bp.route("/pdf", methods=["POST"])
def download_pdf_report():
    if not model_is_ready():
        return jsonify({"error": "Service Unavailable", "message": "Model not trained yet."}), 503
    path = generate_pdf_report()
    return send_file(path, as_attachment=True, download_name="fraud_report.pdf")
