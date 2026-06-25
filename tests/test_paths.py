from pathlib import Path

from src.paths import resolve_artifact_path


def test_resolve_docker_artifact_path(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "proj"
    models = root / "artifacts" / "models"
    models.mkdir(parents=True)
    artifact = models / "model_v1.joblib"
    artifact.write_bytes(b"model")

    monkeypatch.setattr("src.paths.project_root", lambda: root)

    resolved = resolve_artifact_path("/app/artifacts/models/model_v1.joblib")
    assert resolved == artifact
    assert resolved.is_file()
