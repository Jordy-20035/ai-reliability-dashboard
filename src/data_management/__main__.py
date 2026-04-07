"""CLI: register raw CSV, list datasets / baselines / provenance."""

from __future__ import annotations

import argparse

from src.drift_detection.load import default_adult_csv_path

from .service import default_data_management_service


def cmd_register_raw(args: argparse.Namespace) -> None:
    dm = default_data_management_service()
    path = args.path or default_adult_csv_path()
    did = dm.register_dataset_from_file(path, name=args.name, notes=args.notes)
    print("dataset_version_id:", did)


def cmd_list_datasets(args: argparse.Namespace) -> None:
    dm = default_data_management_service()
    rows = dm.list_datasets(limit=args.limit)
    if not rows:
        print("(no dataset versions — run: python -m src.data_management register-raw)")
        return
    for d in rows:
        print(d.id, d.content_hash[:12], d.row_count, d.name, d.kind)


def cmd_list_baselines(args: argparse.Namespace) -> None:
    dm = default_data_management_service()
    rows = dm.list_baselines(limit=args.limit)
    if not rows:
        print(
            "(no baseline snapshots — run: python -m src.orchestration init-baseline "
            "so baseline_profile.json exists and is registered, or run a retrain)"
        )
        return
    for b in rows:
        print(b.id, b.content_hash[:12], b.artifact_path, b.dataset_version_id)


def cmd_list_provenance(args: argparse.Namespace) -> None:
    dm = default_data_management_service()
    rows = dm.list_provenance(limit=args.limit)
    if not rows:
        print(
            "(no provenance rows — these are written when you run "
            "python -m src.retraining … or retrain via orchestration after drift triggers)"
        )
        return
    for p in rows:
        print(p)


def main() -> None:
    p = argparse.ArgumentParser(description="Data management (SQLite)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("register-raw", help="Register a CSV file by content hash")
    pr.add_argument("--path", type=str, default=None)
    pr.add_argument("--name", type=str, default="adult_raw")
    pr.add_argument("--notes", type=str, default=None)
    pr.set_defaults(func=cmd_register_raw)

    ld = sub.add_parser("list-datasets")
    ld.add_argument("--limit", type=int, default=50)
    ld.set_defaults(func=cmd_list_datasets)

    lb = sub.add_parser("list-baselines")
    lb.add_argument("--limit", type=int, default=50)
    lb.set_defaults(func=cmd_list_baselines)

    lp = sub.add_parser("list-provenance")
    lp.add_argument("--limit", type=int, default=50)
    lp.set_defaults(func=cmd_list_provenance)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
