from flask import Blueprint, render_template, current_app

from app.services.model_service import get_feature_columns, get_metadata, model_is_ready
from app.services.analytics_service import dashboard_summary

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def dashboard():
    ready = model_is_ready()
    summary = dashboard_summary() if ready else {}
    return render_template("dashboard.html", ready=ready, summary=summary,
                            error=getattr(current_app, "model_load_error", None))


@main_bp.route("/analyze")
def analyze():
    features = get_feature_columns()
    pca_features = [f for f in features if f.startswith("V")]
    other_features = [f for f in features if not f.startswith("V")]
    return render_template("analyze.html", ready=model_is_ready(),
                            pca_features=pca_features, other_features=other_features)


@main_bp.route("/batch")
def batch():
    return render_template("batch.html", ready=model_is_ready())


@main_bp.route("/analytics")
def analytics_page():
    summary = dashboard_summary() if model_is_ready() else {}
    return render_template("analytics.html", ready=model_is_ready(), summary=summary)


@main_bp.route("/model")
def model_page():
    metadata = get_metadata()
    return render_template("model.html", ready=model_is_ready(), metadata=metadata)


@main_bp.route("/explain")
def explain_page():
    features = get_feature_columns()
    pca_features = [f for f in features if f.startswith("V")]
    other_features = [f for f in features if not f.startswith("V")]
    return render_template("explain.html", ready=model_is_ready(),
                            pca_features=pca_features, other_features=other_features)


@main_bp.route("/reports")
def reports_page():
    return render_template("reports.html", ready=model_is_ready())


@main_bp.route("/settings")
def settings_page():
    return render_template("settings.html")


@main_bp.route("/about")
def about_page():
    metadata = get_metadata()
    return render_template("about.html", metadata=metadata)
