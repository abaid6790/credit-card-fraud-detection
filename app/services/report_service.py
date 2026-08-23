import csv
import io
import os
from datetime import datetime, timezone

from app.services.analytics_service import dashboard_summary
from app.services.model_service import get_metadata


def generate_csv_report() -> str:
    summary = dashboard_summary()
    metadata = get_metadata()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Fraud Detection Report", datetime.now(timezone.utc).isoformat()])
    writer.writerow([])
    writer.writerow(["Summary"])
    for k, v in summary["prediction_history"].items():
        writer.writerow([k, v])
    writer.writerow([])
    writer.writerow(["Model"])
    writer.writerow(["model_name", metadata.get("model_name")])
    writer.writerow(["threshold", metadata.get("threshold")])
    for k, v in metadata.get("metrics", {}).items():
        if k != "confusion_matrix":
            writer.writerow([k, v])

    export_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = os.path.join(export_dir, filename)
    with open(path, "w", newline="") as f:
        f.write(buf.getvalue())
    return path


def generate_pdf_report() -> str:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch

    summary = dashboard_summary()
    metadata = get_metadata()

    export_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(export_dir, exist_ok=True)
    filename = f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    path = os.path.join(export_dir, filename)

    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - inch

    def line(text, size=11, gap=18, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(inch, y, text)
        y -= gap

    line("Fraud Transaction Detection - Report", size=16, bold=True, gap=28)
    line(f"Generated: {datetime.now(timezone.utc).isoformat()}", size=9, gap=24)

    line("Summary", size=13, bold=True)
    for k, v in summary["prediction_history"].items():
        line(f"  {k}: {v}")

    y -= 10
    line("Model", size=13, bold=True)
    line(f"  model_name: {metadata.get('model_name')}")
    line(f"  threshold: {metadata.get('threshold')}")
    for k, v in metadata.get("metrics", {}).items():
        if k != "confusion_matrix":
            line(f"  {k}: {v}")

    y -= 10
    line("Note: This is a machine-learning-based fraud-risk estimate, not a", size=9)
    line("guaranteed determination of fraud. See /about for full limitations.", size=9)

    c.save()
    return path
