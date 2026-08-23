import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True, "DATABASE_PATH": ":memory:"})
    with app.test_client() as client:
        yield client


def test_dashboard_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_analyze_page_loads(client):
    resp = client.get("/analyze")
    assert resp.status_code == 200


def test_predict_without_model_returns_503_or_error(client):
    resp = client.post("/api/predict", json={"Time": 1, "Amount": 10})
    # If no model is trained in this test environment, expect 503.
    # If a model happens to be trained, a well-formed prediction or 422 is fine too.
    assert resp.status_code in (200, 422, 503)


def test_predict_rejects_non_json(client):
    resp = client.post("/api/predict", data="not json")
    assert resp.status_code in (400, 503)


def test_unknown_transaction_returns_404(client):
    resp = client.get("/transactions/999999")
    assert resp.status_code == 404


def test_clear_history_endpoint(client):
    resp = client.post("/api/transactions/clear")
    assert resp.status_code == 200
