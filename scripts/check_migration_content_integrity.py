#!/usr/bin/env python3
"""Inspect Atlas migration-source provenance without mutating the database.

Run this from the exact Atlas release-candidate worktree before a
migration-bearing cutover. It reads the configured Atlas database target through
the existing typed `ATLAS_DB_*` settings and emits JSON only. It never invokes
the migration runner, evolves `schema_migrations`, or executes migration SQL.

Exit status:
  0: the configured target was displayed, or the check completed without
     mismatched or missing-source evidence.
     Legacy null hashes remain explicitly visible as `legacy_unverified`.
  2: mismatched or missing-source evidence requires reconciliation.
  3: the check could not determine the state (for example, connection failure).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas_brain.storage.config import db_settings  # noqa: E402
from atlas_brain.storage.migrations import (  # noqa: E402
    MIGRATIONS_DIR,
    MigrationContentIntegrityReport,
    migration_content_integrity_report,
)
from atlas_brain.storage.migrations.reconciliation import (  # noqa: E402
    attest_known_historical_migration_reconciliations,
    known_historical_reconciliation_names,
)


UNRESOLVED_DRIFT_EXIT = 2
COULD_NOT_DETERMINE_EXIT = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--show-target",
        action="store_true",
        help="Print the configured log-safe target label without connecting.",
    )
    target_group.add_argument(
        "--expected-target",
        help="Exact log-safe target label required before the catalog query runs.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for operator consistency; output is always JSON.",
    )
    parser.add_argument(
        "--attest-known-reconciliations",
        action="store_true",
        help=(
            "Include read-only catalog evidence for reviewed historical source "
            "gaps; requires --expected-target."
        ),
    )
    args = parser.parse_args(argv)
    if args.show_target and args.attest_known_reconciliations:
        parser.error("--attest-known-reconciliations requires --expected-target")
    return args


def _status(report: MigrationContentIntegrityReport) -> tuple[str, int]:
    if report.mismatched or report.missing_source:
        return "unresolved_drift", UNRESOLVED_DRIFT_EXIT
    if report.legacy_unverified:
        return "legacy_unverified", 0
    return "verified", 0


def _report_payload(
    report: MigrationContentIntegrityReport,
    *,
    reconciliation_evidence: list[dict[str, object]] | None = None,
) -> tuple[int, dict[str, object]]:
    status, exit_code = _status(report)
    categories = {
        "verified": list(report.verified),
        "legacy_unverified": list(report.legacy_unverified),
        "mismatched": list(report.mismatched),
        "missing_source": list(report.missing_source),
    }
    payload: dict[str, object] = {
        "check_completed": True,
        "status": status,
        "exit_code": exit_code,
        "report": categories,
        "counts": {name: len(values) for name, values in categories.items()},
    }
    if reconciliation_evidence is not None:
        payload["known_reconciliation_evidence"] = reconciliation_evidence
    return exit_code, payload


def _failure_payload(exc: Exception) -> dict[str, object]:
    """Return diagnosable failure metadata without leaking connection details."""
    return {
        "check_completed": False,
        "status": "could_not_determine",
        "exit_code": COULD_NOT_DETERMINE_EXIT,
        "error_type": exc.__class__.__name__,
    }


def _target_confirmation_payload(
    *,
    target_label: str,
    status: str,
) -> dict[str, object]:
    return {
        "check_completed": False,
        "status": status,
        "exit_code": COULD_NOT_DETERMINE_EXIT,
        "database_target": target_label,
    }


async def _connect_read_only() -> Any:
    if not db_settings.enabled:
        raise RuntimeError("Atlas database persistence is disabled")
    try:
        import asyncpg
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError("asyncpg is required for the migration integrity preflight") from exc

    connection_kwargs = dict(db_settings.connection_kwargs())
    existing_settings = connection_kwargs.get("server_settings")
    server_settings = dict(existing_settings) if isinstance(existing_settings, Mapping) else {}
    server_settings["default_transaction_read_only"] = "on"
    connection_kwargs["server_settings"] = server_settings
    return await asyncpg.connect(**connection_kwargs)


async def run_migration_content_integrity_preflight(
    connection: Any,
    *,
    migrations_dir: Path = MIGRATIONS_DIR,
    attest_known_reconciliations: bool = False,
) -> tuple[int, dict[str, object]]:
    """Read the packaged catalog and ledger in a transaction that rejects writes."""
    migration_files = sorted(migrations_dir.glob("*.sql"))
    reconciliation_evidence: list[dict[str, object]] | None = None
    async with connection.transaction(readonly=True):
        report = await migration_content_integrity_report(connection, migration_files)
        if attest_known_reconciliations:
            reported_names = frozenset(report.mismatched) | frozenset(
                report.missing_source
            )
            attestations = await attest_known_historical_migration_reconciliations(
                connection,
                migration_files,
                candidate_names=(
                    reported_names & known_historical_reconciliation_names()
                ),
            )
            reconciliation_evidence = [attestation.as_payload() for attestation in attestations]
    return _report_payload(report, reconciliation_evidence=reconciliation_evidence)


async def _main(
    *,
    migrations_dir: Path = MIGRATIONS_DIR,
    database_target: str | None = None,
    attest_known_reconciliations: bool = False,
) -> int:
    connection: Any | None = None
    exit_code = COULD_NOT_DETERMINE_EXIT
    payload: dict[str, object]
    target_label = db_settings.target_label if database_target is None else database_target
    try:
        connection = await _connect_read_only()
        exit_code, payload = await run_migration_content_integrity_preflight(
            connection,
            migrations_dir=migrations_dir,
            attest_known_reconciliations=attest_known_reconciliations,
        )
    except Exception as exc:
        payload = _failure_payload(exc)
    finally:
        if connection is not None:
            try:
                await connection.close()
            except Exception as exc:
                if exit_code == 0:
                    exit_code = COULD_NOT_DETERMINE_EXIT
                    payload = _failure_payload(exc)
    payload["database_target"] = target_label
    print(json.dumps(payload, sort_keys=True))
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    target_label = db_settings.target_label
    if args.show_target:
        print(json.dumps({"database_target": target_label, "status": "target_displayed"}, sort_keys=True))
        return 0
    if args.expected_target is None:
        print(json.dumps(
            _target_confirmation_payload(
                target_label=target_label,
                status="target_confirmation_required",
            ),
            sort_keys=True,
        ))
        return COULD_NOT_DETERMINE_EXIT
    if args.expected_target != target_label:
        print(json.dumps(
            _target_confirmation_payload(
                target_label=target_label,
                status="target_confirmation_mismatch",
            ),
            sort_keys=True,
        ))
        return COULD_NOT_DETERMINE_EXIT
    return asyncio.run(
        _main(
            database_target=target_label,
            attest_known_reconciliations=args.attest_known_reconciliations,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
