import pandas as pd

from src.drift_detection.report import DriftReport
from src.orchestration.policies import DriftThresholdPolicy


def test_policy_triggers_when_high_psi_exceeds():
    df = pd.DataFrame(
        [
            {"feature": "a", "kind": "numeric", "psi_band": "high"},
            {"feature": "b", "kind": "numeric", "psi_band": "stable"},
        ]
    )
    summary = {
        "n_features_high_psi": 2,
        "n_numeric_ks_significant": 0,
        "n_categorical_chi2_significant": 0,
    }
    report = DriftReport(feature_results=df, summary=summary)
    policy = DriftThresholdPolicy(max_high_psi_features=1)
    triggered, reasons = policy.evaluate(report)
    assert triggered
    assert any("high_psi" in r for r in reasons)


def test_policy_calm_when_within_bounds():
    df = pd.DataFrame()
    summary = {
        "n_features_high_psi": 0,
        "n_numeric_ks_significant": 0,
        "n_categorical_chi2_significant": 1,
    }
    report = DriftReport(feature_results=df, summary=summary)
    policy = DriftThresholdPolicy(
        max_high_psi_features=0,
        max_ks_significant_numeric=2,
        max_chi2_significant_categorical=3,
    )
    triggered, reasons = policy.evaluate(report)
    assert not triggered
    assert reasons == []
