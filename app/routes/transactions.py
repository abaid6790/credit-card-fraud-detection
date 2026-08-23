import json
from flask import Blueprint, render_template, request, jsonify, abort

from app import get_db

transactions_bp = Blueprint("transactions", __name__)


@transactions_bp.route("/transactions")
def transactions_page():
    filter_type = request.args.get("filter", "all")
    page = max(int(request.args.get("page", 1)), 1)
    per_page = 25
    offset = (page - 1) * per_page

    db = get_db()
    where_clause, params = _build_filter(filter_type)

    rows = db.execute(
        f"SELECT * FROM transactions {where_clause} ORDER BY id DESC LIMIT ? OFFSET ?",
        (*params, per_page, offset),
    ).fetchall()
    total = db.execute(f"SELECT COUNT(*) as cnt FROM transactions {where_clause}", params).fetchone()["cnt"]

    return render_template("transactions.html", rows=rows, filter_type=filter_type,
                            page=page, total_pages=max((total + per_page - 1) // per_page, 1))


@transactions_bp.route("/transactions/<int:tx_id>")
def transaction_detail(tx_id):
    db = get_db()
    row = db.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
    if row is None:
        abort(404)
    features = json.loads(row["features_json"]) if row["features_json"] else {}
    return render_template("transaction_detail.html", tx=row, features=features)


@transactions_bp.route("/api/transactions")
def api_transactions():
    filter_type = request.args.get("filter", "all")
    limit = min(int(request.args.get("limit", 50)), 500)
    db = get_db()
    where_clause, params = _build_filter(filter_type)
    rows = db.execute(
        f"SELECT id, timestamp, prediction, fraud_probability, risk_score, risk_level, amount "
        f"FROM transactions {where_clause} ORDER BY id DESC LIMIT ?",
        (*params, limit),
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@transactions_bp.route("/api/transactions/<int:tx_id>")
def api_transaction_detail(tx_id):
    db = get_db()
    row = db.execute("SELECT * FROM transactions WHERE id = ?", (tx_id,)).fetchone()
    if row is None:
        return jsonify({"error": "Not Found", "message": "Transaction not found."}), 404
    result = dict(row)
    result["features"] = json.loads(result.pop("features_json") or "{}")
    return jsonify(result)


@transactions_bp.route("/api/transactions/clear", methods=["POST"])
def clear_history():
    """Privacy control: allow clearing stored prediction history."""
    db = get_db()
    db.execute("DELETE FROM transactions")
    db.commit()
    return jsonify({"message": "Prediction history cleared."}), 200


def _build_filter(filter_type: str):
    if filter_type == "legitimate":
        return "WHERE prediction = 'LEGITIMATE'", ()
    if filter_type == "fraud":
        return "WHERE prediction = 'FRAUD'", ()
    if filter_type == "high":
        return "WHERE risk_level = 'HIGH'", ()
    if filter_type == "critical":
        return "WHERE risk_level = 'CRITICAL'", ()
    return "", ()
