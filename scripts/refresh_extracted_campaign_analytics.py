#!/usr/bin/env python3
"""Refresh extracted campaign analytics materialized views."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_postgres_analytics import (  # noqa: E402
    refresh_campaign_analytics_from_postgres,
)
from extracted_content_pipeline.campaign_visibility import (  # noqa: E402
    JsonlVisibilitySink,
    OPERATION_COMPLETED_EVENT,
    OPERATION_FAILED_EVENT,
    OPERATION_STARTED_EVENT,
    REPORTED_ERROR_TYPE,
    emit_operation_event,
    visibility_result_summary,
)


DATABASE_URL_ENV = ("EXTRACTED_DATABASE_URL", "DATABASE_URL")


def _env(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value not in (None, ""):
            return value
    return default


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh campaign analytics for the extracted product database."
    )
    parser.add_argument(
        "--database-url",
        default=_env(*DATABASE_URL_ENV),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of a concise text summary.",
    )
    parser.add_argument(
        "--visibility-jsonl",
        type=Path,
        help="Append campaign operation visibility events to this JSONL file.",
    )
    return parser.parse_args(argv)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required to refresh campaign analytics; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


async def _main() -> int:
    args = _parse_args()
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")

    visibility = JsonlVisibilitySink(args.visibility_jsonl) if args.visibility_jsonl else None
    await emit_operation_event(
        visibility,
        OPERATION_STARTED_EVENT,
        "analytics_refresh",
        {},
    )
    pool = None
    try:
        pool = await _create_pool(args.database_url)
        result = await refresh_campaign_analytics_from_postgres(pool)
    except Exception as exc:
        await emit_operation_event(
            visibility,
            OPERATION_FAILED_EVENT,
            "analytics_refresh",
            {"error_type": type(exc).__name__},
        )
        raise
    finally:
        if pool is not None and (close := getattr(pool, "close", None)) is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    summary = result.as_dict()
    if not result.refreshed:
        await emit_operation_event(
            visibility,
            OPERATION_FAILED_EVENT,
            "analytics_refresh",
            {
                "error_type": REPORTED_ERROR_TYPE,
                "result": visibility_result_summary(summary),
            },
        )
    else:
        await emit_operation_event(
            visibility,
            OPERATION_COMPLETED_EVENT,
            "analytics_refresh",
            {"result": visibility_result_summary(summary)},
        )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        text = "refreshed={refreshed}".format(**summary)
        if summary.get("error"):
            text = f"{text} error={summary['error']}"
        print(text)
    return 0 if result.refreshed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
