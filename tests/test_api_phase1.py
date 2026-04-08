from fastapi.testclient import TestClient

from src.api.main import app


def test_health() -> None:
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_overview_shape() -> None:
    client = TestClient(app)
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert "kpis" in body
    assert "last_run" in body

