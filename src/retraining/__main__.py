"""Manual retrain run (same labeled split as orchestration)."""

from __future__ import annotations

import argparse

from src.orchestration.data_context import split_labeled_reference_current

from .pipeline import run_retrain_pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Run retrain pipeline once")
    p.add_argument(
        "--scenario",
        choices=["random_holdout", "age_shift"],
        default="random_holdout",
    )
    args = p.parse_args()
    ref, cur = split_labeled_reference_current(
        test_size=0.3,
        random_state=42,
        scenario=args.scenario,
    )
    result = run_retrain_pipeline(ref, cur)
    print("version", result.version)
    print("promoted", result.promoted, result.promote_reason)
    print("metrics", result.metrics)
    print("artifact", result.artifact_path)


if __name__ == "__main__":
    main()
