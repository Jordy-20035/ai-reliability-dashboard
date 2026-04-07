"""CLI: list experiments/models, promote stages."""

from __future__ import annotations

import argparse

from .service import default_lifecycle_service
from .stages import DeploymentStage


def cmd_list_experiments(args: argparse.Namespace) -> None:
    svc = default_lifecycle_service()
    for e in svc.list_experiments(limit=args.limit):
        print(e.id, e.created_at, e.name, e.scenario, e.git_sha or "-")


def cmd_list_models(args: argparse.Namespace) -> None:
    svc = default_lifecycle_service()
    stage = DeploymentStage(args.stage) if args.stage else None
    for m in svc.list_models(stage=stage):
        print(
            m.id,
            f"v{m.version_num}",
            m.stage.value,
            m.created_at[:19],
            m.artifact_path,
        )


def cmd_promote(args: argparse.Namespace) -> None:
    svc = default_lifecycle_service()
    svc.promote_stage(args.model_id, DeploymentStage(args.to))
    print("OK")


def cmd_production(args: argparse.Namespace) -> None:
    svc = default_lifecycle_service()
    mid = svc.get_production_model_id()
    print("production_model_row_id:", mid)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Model lifecycle (experiments + stages)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("list-experiments", help="List recent experiments")
    pe.add_argument("--limit", type=int, default=30)
    pe.set_defaults(func=cmd_list_experiments)

    pm = sub.add_parser("list-models", help="List model versions")
    pm.add_argument(
        "--stage",
        choices=[s.value for s in DeploymentStage],
        default=None,
    )
    pm.set_defaults(func=cmd_list_models)

    pp = sub.add_parser("promote", help="Move a model row to a new stage")
    pp.add_argument("model_id", type=int)
    pp.add_argument(
        "--to",
        required=True,
        choices=[s.value for s in DeploymentStage],
    )
    pp.set_defaults(func=cmd_promote)

    sub.add_parser("production-id", help="Show settings: production model row id").set_defaults(
        func=cmd_production
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
