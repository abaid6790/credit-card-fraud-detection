import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app({"TESTING": True, "DATABASE_PATH": ":memory:"})
    with app.test_client() as client:
        yield client


def test_no_stack_trace_on_error(client):
    resp = client.get("/transactions/999999")
    body = resp.get_data(as_text=True)
    assert "Traceback" not in body
    assert "File \"" not in body


def test_path_traversal_blocked_on_download(client):
    resp = client.get("/api/batch/download/..%2F..%2Fetc%2Fpasswd")
    assert resp.status_code in (400, 404)


def test_batch_predict_requires_file(client):
    resp = client.post("/api/predict/batch", data={})
    assert resp.status_code in (400, 503)


def test_batch_predict_rejects_non_csv(client):
    from io import BytesIO
    data = {"file": (BytesIO(b"not a csv"), "test.txt")}
    resp = client.post("/api/predict/batch", data=data, content_type="multipart/form-data")
    assert resp.status_code in (400, 503)
