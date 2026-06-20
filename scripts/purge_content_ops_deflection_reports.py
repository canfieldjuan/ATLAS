#!/usr/bin/env python3
"""Purge stored FAQ deflection report rows past an explicit retention window."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.deflection_report_access import (  # noqa: E402
    PostgresDeflectionReportArtifactStore,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database-url",
        help=(
            "Postgres DSN for local/disposable runs. Prefer --database-url-env "
            "or --database-url-file for production so the DSN is not exposed on argv."
        ),
    )
    parser.add_argument(
        "--database-url-env",
        help="Name of the environment variable containing the Postgres DSN.",
    )
    parser.add_argument(
        "--database-url-file",
        type=Path,
        help="File containing the Postgres DSN.",
    )
    parser.add_argument(
        "--retention-days",
        required=True,
        type=int,
        help="Delete rows with created_at older than this many days.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to delete in this run.",
    )
    parser.add_argument(
        "--confirm-delete",
        action="store_true",
        help="Actually delete rows. Omit for dry-run count-only mode.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Accepted for operator consistency; output is always JSON.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    _validate_database_url_source(args)
    try:
        retention_days = int(args.retention_days)
    except (TypeError, ValueError) as exc:
        raise SystemExit("--retention-days must be an integer") from exc
    if retention_days < 1:
        raise SystemExit("--retention-days must be greater than 0")
    if args.limit is not None:
        try:
            limit = int(args.limit)
        except (TypeError, ValueError) as exc:
            raise SystemExit("--limit must be an integer") from exc
        if limit < 1:
            raise SystemExit("--limit must be greater than 0")


def _validate_database_url_source(args: argparse.Namespace) -> None:
    sources = (
        bool(_clean(getattr(args, "database_url", None))),
        bool(_clean(getattr(args, "database_url_env", None))),
        getattr(args, "database_url_file", None) is not None,
    )
    source_count = sum(1 for source in sources if source)
    if source_count != 1:
        raise SystemExit(
            "provide exactly one database URL source: --database-url, "
            "--database-url-env, or --database-url-file"
        )


def _resolve_database_url(args: argparse.Namespace) -> str:
    _validate_database_url_source(args)
    if _clean(getattr(args, "database_url", None)):
        return _clean(args.database_url)
    env_name = _clean(getattr(args, "database_url_env", None))
    if env_name:
        value = _clean(os.environ.get(env_name))
        if not value:
            raise SystemExit(f"--database-url-env {env_name!r} is not set or is empty")
        return value
    file_path = getattr(args, "database_url_file", None)
    if file_path is None:
        raise SystemExit("database URL source is required")
    try:
        value = _clean(Path(file_path).read_text())
    except OSError as exc:
        raise SystemExit(f"could not read --database-url-file: {exc}") from exc
    if not value:
        raise SystemExit("--database-url-file is empty")
    return value


def _preflight_output(output: Path | None) -> None:
    if output is None:
        return
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8"):
            pass
    except OSError as exc:
        raise SystemExit(f"could not prepare --output path: {exc}") from exc


async def _create_pool(database_url: str) -> Any:
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to purge deflection report retention rows"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def run_deflection_report_retention_purge(
    args: argparse.Namespace,
    pool: Any,
    *,
    store: PostgresDeflectionReportArtifactStore | None = None,
    now: datetime | None = None,
) -> tuple[int, dict[str, Any]]:
    """Run the guarded purge, returning an exit code and JSON-safe payload."""

    _validate_args(args)
    resolved_now = _aware_utc(now or datetime.now(timezone.utc))
    retention_days = int(args.retention_days)
    limit = int(args.limit) if args.limit is not None else None
    cutoff = resolved_now - timedelta(days=retention_days)
    artifact_store = store or PostgresDeflectionReportArtifactStore(pool=pool)

    candidate_count = await artifact_store.count_reports_older_than(cutoff=cutoff)
    dry_run = not bool(args.confirm_delete)
    deleted_count = 0
    if not dry_run:
        deleted_count = await artifact_store.delete_reports_older_than(
            cutoff=cutoff,
            limit=limit,
        )

    return (
        0,
        {
            "ok": True,
            "dry_run": dry_run,
            "retention_days": retention_days,
            "cutoff": cutoff.isoformat(),
            "candidate_count": int(candidate_count),
            "deleted_count": int(deleted_count),
            "limit": limit,
        },
    )


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    _preflight_output(args.output)
    database_url = _resolve_database_url(args)
    pool = await _create_pool(database_url)
    code = 1
    payload: dict[str, Any] = {"ok": False}
    try:
        code, payload = await run_deflection_report_retention_purge(args, pool)
    finally:
        await pool.close()
    _emit_payload_after_purge(payload, output=args.output)
    return code


def _emit_payload(payload: Mapping[str, Any], *, output: Path | None = None) -> None:
    body = json.dumps(dict(payload), sort_keys=True)
    if output is None:
        print(body)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body + "\n")


def _emit_payload_after_purge(
    payload: Mapping[str, Any], *, output: Path | None = None
) -> None:
    try:
        _emit_payload(payload, output=output)
    except OSError as exc:
        fallback = dict(payload)
        fallback["output_error"] = f"failed to write --output: {exc.__class__.__name__}"
        print(json.dumps(fallback, sort_keys=True))


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _clean(value: Any) -> str:
    return str(value or "").strip()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
