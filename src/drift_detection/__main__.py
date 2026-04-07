"""Demo: random split (low drift) vs age-shifted current (visible drift)."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from .baseline import build_baseline
from .load import load_adult_csv
from .report import run_drift_analysis
from .schema import ADULT_CATEGORICAL_FEATURES, ADULT_NUMERIC_FEATURES


def main() -> None:
    df = load_adult_csv()
    feature_cols = list(ADULT_NUMERIC_FEATURES) + list(ADULT_CATEGORICAL_FEATURES)
    X = df[feature_cols]

    ref, cur_same = train_test_split(X, test_size=0.3, random_state=42)
    baseline = build_baseline(
        ref,
        list(ADULT_NUMERIC_FEATURES),
        list(ADULT_CATEGORICAL_FEATURES),
        psi_bins=10,
        metadata={"scenario": "random_split"},
    )

    print("=== Scenario A: reference vs holdout (same distribution) ===")
    report_a = run_drift_analysis(ref, cur_same, baseline)
    print(report_a.summary)
    print(report_a.feature_results[["feature", "kind", "psi", "psi_band", "ks_pvalue"]].head(8))
    print()

    cur_shifted = df[df["age"] >= 40][feature_cols]
    print("=== Scenario B: reference vs current = age >= 40 (distribution shift) ===")
    report_b = run_drift_analysis(ref, cur_shifted, baseline)
    print(report_b.summary)
    print(report_b.feature_results[["feature", "kind", "psi", "psi_band", "ks_pvalue", "chi2_pvalue"]].head(12))


if __name__ == "__main__":
    main()
