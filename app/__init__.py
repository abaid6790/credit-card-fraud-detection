import os
import sqlite3
from flask import Flask, g
from flask_wtf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

load_dotenv()

csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address, default_limits=["200 per hour"])

DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join("instance", "app.db"))


def get_db():
    if "db" not in g:
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    with app.app_context():
        os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
        conn = sqlite3.connect(DATABASE_PATH)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prediction TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                amount REAL,
                model_name TEXT,
                features_json TEXT
            )
            """
        )
        conn.commit()
        conn.close()


def create_app(test_config: dict | None = None):
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key-change-in-production"),
        MAX_CONTENT_LENGTH=int(os.environ.get("MAX_UPLOAD_MB", 10)) * 1024 * 1024,
        UPLOAD_FOLDER=os.path.join(os.getcwd(), "uploads"),
        EXPORT_FOLDER=os.path.join(os.getcwd(), "exports"),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if test_config:
        app.config.update(test_config)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["EXPORT_FOLDER"], exist_ok=True)

    csrf.init_app(app)
    limiter.init_app(app)
    app.teardown_appcontext(close_db)

    init_db(app)

    from ml.predict import FraudPredictor
    app.predictor = FraudPredictor()
    try:
        app.predictor.load()
        app.model_ready = True
    except Exception as e:
        app.model_ready = False
        app.model_load_error = str(e)

    from app.routes.main import main_bp
    from app.routes.prediction import prediction_bp
    from app.routes.transactions import transactions_bp
    from app.routes.analytics import analytics_bp
    from app.routes.monitoring import monitoring_bp
    from app.routes.model import model_bp
    from app.routes.reports import reports_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(transactions_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(monitoring_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(reports_bp)

    # JSON API blueprints are exempted from CSRF (they're not browser-form based);
    # they still go through rate limiting, input validation, and JSON-only parsing.
    csrf.exempt(prediction_bp)
    csrf.exempt(transactions_bp)
    csrf.exempt(analytics_bp)
    csrf.exempt(monitoring_bp)
    csrf.exempt(model_bp)
    csrf.exempt(reports_bp)

    @app.errorhandler(400)
    def bad_request(e):
        return {"error": "Bad Request", "message": "Invalid request data."}, 400

    @app.errorhandler(413)
    def too_large(e):
        return {"error": "Payload Too Large", "message": "Uploaded file exceeds the allowed size."}, 413

    @app.errorhandler(404)
    def not_found(e):
        return {"error": "Not Found", "message": "The requested resource does not exist."}, 404

    @app.errorhandler(500)
    def server_error(e):
        return {"error": "Internal Server Error",
                "message": "Fraud detection service temporarily unavailable."}, 500

    return app
