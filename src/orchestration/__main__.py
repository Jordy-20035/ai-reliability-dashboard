"""CLI: init baseline, single run, scheduled loop, optional HTTP API."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import OrchestratorConfig
from .data_context import fit_or_load_baseline, load_feature_matrix, split_reference_current
from .engine import Orchestrator
from .scheduler import start_interval_scheduler
from .store import RunStore


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def cmd_init_baseline(args: argparse.Namespace) -> None:
    cfg = OrchestratorConfig()
    X = load_feature_matrix()
    ref, _ = split_reference_current(
        X,
        test_size=args.test_size,
        random_state=args.random_state,
        scenario="random_holdout",
    )
    assert cfg.baseline_path is not None
    fit_or_load_baseline(ref, cfg.baseline_path, psi_bins=args.psi_bins)
    print(f"Baseline saved to {cfg.baseline_path}")


def cmd_check_once(args: argparse.Namespace) -> None:
    cfg = OrchestratorConfig(
        scenario=args.scenario,
        max_high_psi_features=args.max_high_psi,
        max_ks_significant_numeric=args.max_ks,
        max_chi2_significant_categorical=args.max_chi2,
    )
    orch = Orchestrator(cfg)
    result = orch.run_pipeline()
    print("policy_triggered:", result.policy_triggered)
    print("trigger_reasons:", result.trigger_reasons)
    print("summary:", result.summary)
    print("run_id:", result.run_id)


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = OrchestratorConfig(
        scenario=args.scenario,
        max_high_psi_features=args.max_high_psi,
        max_ks_significant_numeric=args.max_ks,
        max_chi2_significant_categorical=args.max_chi2,
    )
    orch = Orchestrator(cfg)

    def job() -> None:
        orch.run_pipeline()

    start_interval_scheduler(job, interval_seconds=args.interval)
    print(f"Scheduler running every {args.interval}s. Ctrl+C to stop.")
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Stopped.")


def cmd_serve_http(args: argparse.Namespace) -> None:
    import uvicorn

    cfg = OrchestratorConfig(
        scenario=args.scenario,
        max_high_psi_features=args.max_high_psi,
        max_ks_significant_numeric=args.max_ks,
        max_chi2_significant_categorical=args.max_chi2,
    )
    orch = Orchestrator(cfg)
    from .api import create_app

    app = create_app(orch)
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_history(args: argparse.Namespace) -> None:
    cfg = OrchestratorConfig()
    assert cfg.sqlite_path is not None
    store = RunStore(cfg.sqlite_path)
    for r in store.recent(limit=args.limit):
        print(
            r.id,
            r.started_at,
            r.scenario,
            "TRIGGER" if r.policy_triggered else "ok",
            r.trigger_reasons,
            r.summary,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MLOps orchestration (drift checks + actions)")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-baseline", help="Fit and save baseline profile to artifacts/")
    p_init.add_argument("--test-size", type=float, default=0.3)
    p_init.add_argument("--random-state", type=int, default=42)
    p_init.add_argument("--psi-bins", type=int, default=10)
    p_init.set_defaults(func=cmd_init_baseline)

    p_once = sub.add_parser("check-once", help="Run drift pipeline once")
    p_once.add_argument(
        "--scenario",
        choices=["random_holdout", "age_shift"],
        default="random_holdout",
    )
    p_once.add_argument("--max-high-psi", type=int, default=0)
    p_once.add_argument("--max-ks", type=int, default=2)
    p_once.add_argument("--max-chi2", type=int, default=3)
    p_once.set_defaults(func=cmd_check_once)

    p_srv = sub.add_parser("serve", help="Run drift check on an interval (background scheduler)")
    p_srv.add_argument("--interval", type=int, default=60, help="Seconds between runs")
    p_srv.add_argument("--scenario", choices=["random_holdout", "age_shift"], default="random_holdout")
    p_srv.add_argument("--max-high-psi", type=int, default=0)
    p_srv.add_argument("--max-ks", type=int, default=2)
    p_srv.add_argument("--max-chi2", type=int, default=3)
    p_srv.set_defaults(func=cmd_serve)

    p_http = sub.add_parser("serve-http", help="HTTP API + POST /run/drift-check")
    p_http.add_argument("--host", default="127.0.0.1")
    p_http.add_argument("--port", type=int, default=8000)
    p_http.add_argument("--scenario", choices=["random_holdout", "age_shift"], default="random_holdout")
    p_http.add_argument("--max-high-psi", type=int, default=0)
    p_http.add_argument("--max-ks", type=int, default=2)
    p_http.add_argument("--max-chi2", type=int, default=3)
    p_http.set_defaults(func=cmd_serve_http)

    p_hist = sub.add_parser("history", help="Show recent runs from SQLite")
    p_hist.add_argument("--limit", type=int, default=20)
    p_hist.set_defaults(func=cmd_history)

    return p


def main() -> None:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
