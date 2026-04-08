from src.orchestration.config import OrchestratorConfig


def test_config_reads_env_toggles(monkeypatch) -> None:
    monkeypatch.setenv("ORCH_ALERT_WEBHOOK_URL", "https://example.test/webhook")
    monkeypatch.setenv("ORCH_ENABLE_AUTO_RETRAIN", "false")
    monkeypatch.setenv("ORCH_SCHEDULER_INTERVAL", "123")
    monkeypatch.setenv("ORCH_CURRENT_CSV_PATH", "/tmp/incoming.csv")

    cfg = OrchestratorConfig()
    assert cfg.alert_webhook_url == "https://example.test/webhook"
    assert cfg.enable_auto_retrain is False
    assert cfg.scheduler_interval_seconds == 123
    assert cfg.current_csv_path == "/tmp/incoming.csv"
