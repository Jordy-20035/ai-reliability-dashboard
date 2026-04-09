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


def test_ops_stats_shape() -> None:
    client = TestClient(app)
    r = client.get("/api/ops/stats")
    assert r.status_code == 200
    body = r.json()
    assert "orchestration" in body
    assert "total_runs" in body["orchestration"]
    assert "trigger_rate" in body["orchestration"]


def test_incoming_csv_requires_path_for_check_once() -> None:
    client = TestClient(app)
    r = client.post("/api/orchestration/check-once?scenario=incoming_csv")
    assert r.status_code == 400


def test_inference_requires_production_model(monkeypatch) -> None:
    class FakeLifecycle:
        def get_production_model_id(self):
            return None

    monkeypatch.setattr("src.api.main.default_lifecycle_service", lambda: FakeLifecycle())
    client = TestClient(app)
    r = client.post(
        "/api/inference/predict",
        json={"rows": [_adult_row()]},
    )
    assert r.status_code == 409


def test_inference_predict_success(monkeypatch, tmp_path) -> None:
    class FakeModelRow:
        id = 7
        version_num = 42
        artifact_path = str(tmp_path / "model.joblib")

    class FakeLifecycle:
        def get_production_model_id(self):
            return 7

        def get_model_by_id(self, model_row_id: int):
            return FakeModelRow() if model_row_id == 7 else None

    class FakeModel:
        def predict(self, X):  # type: ignore[no-untyped-def]
            return [1 for _ in range(len(X))]

        def predict_proba(self, X):  # type: ignore[no-untyped-def]
            return [[0.1, 0.9] for _ in range(len(X))]

    (tmp_path / "model.joblib").write_bytes(b"fake")
    monkeypatch.setattr("src.api.main.default_lifecycle_service", lambda: FakeLifecycle())
    monkeypatch.setattr("src.api.main._load_model_from_artifact", lambda p: FakeModel())
    client = TestClient(app)
    r = client.post("/api/inference/predict", json={"rows": [_adult_row(), _adult_row()]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_row_id"] == 7
    assert body["n_rows"] == 2
    assert body["predictions"] == [1, 1]
    assert body["predicted_income_class"] == [">50K", ">50K"]
    assert body["positive_class_probability"] == [0.9, 0.9]


def _adult_row() -> dict:
    return {
        "age": 37,
        "fnlwgt": 120000,
        "education.num": 10,
        "capital.gain": 0,
        "capital.loss": 0,
        "hours.per.week": 40,
        "workclass": "Private",
        "education": "HS-grad",
        "marital.status": "Married-civ-spouse",
        "occupation": "Craft-repair",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "native.country": "United-States",
    }

