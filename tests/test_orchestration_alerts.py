import pandas as pd

from src.drift_detection.report import DriftReport
from src.orchestration.actions import PipelineContext
from src.orchestration.alerts import WebhookAlertAction, build_alert_payload


def _ctx(*, triggered: bool) -> PipelineContext:
    report = DriftReport(
        feature_results=pd.DataFrame(),
        summary={"n_features_high_psi": 1},
    )
    return PipelineContext(
        report=report,
        policy_triggered=triggered,
        trigger_reasons=["high_psi count 1 > 0"] if triggered else [],
        metadata={"scenario": "age_shift"},
    )


def test_build_alert_payload_contains_core_fields() -> None:
    payload = build_alert_payload(_ctx(triggered=True))
    assert payload["event"] == "drift_policy_triggered"
    assert payload["scenario"] == "age_shift"
    assert payload["trigger_reasons"]
    assert "summary" in payload


def test_webhook_action_skips_when_not_triggered(monkeypatch) -> None:
    called = {"n": 0}

    def fake_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["n"] += 1

    monkeypatch.setattr("src.orchestration.alerts.post_json_webhook", fake_post)
    action = WebhookAlertAction("https://example.test/webhook")
    action.run(_ctx(triggered=False))
    assert called["n"] == 0
