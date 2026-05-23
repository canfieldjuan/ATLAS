#!/usr/bin/env python3
"""Smoke-test support-ticket FAQ generation through review/publish lifecycle."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ticket_faq_export import export_ticket_faq_drafts  # noqa: E402
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    TicketFAQMarkdownService,
)
from extracted_content_pipeline.ticket_faq_postgres import (  # noqa: E402
    PostgresTicketFAQRepository,
)
from content_ops_faq_smoke_profile import (  # noqa: E402
    console_input_profile,
    empty_input_profile,
    input_profile_error,
    input_profile_from_loaded,
    raw_row_profile_or_error,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - host dependency
    load_dotenv = None


DEFAULT_SOURCE_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
DEFAULT_REVIEW_STATUS = "published"


def _default_database_url() -> str | None:
    raw = os.getenv("EXTRACTED_DATABASE_URL") or os.getenv("DATABASE_URL")
    if raw:
        return raw
    try:
        from atlas_brain.storage.config import db_settings
    except Exception:
        return None
    dsn = str(getattr(db_settings, "dsn", "") or "").strip()
    return dsn or None


def _load_dotenv_files() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")
        load_dotenv(ROOT / ".env.local", override=True)


async def _create_pool(database_url: str):
    try:
        import asyncpg  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - host dependency
        raise RuntimeError(
            "asyncpg is required for the FAQ lifecycle smoke; install it in the host app"
        ) from exc
    return await asyncpg.create_pool(dsn=database_url, min_size=1, max_size=2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test support-ticket source rows through FAQ Markdown "
            "generation, Postgres persistence, export, and review status update."
        )
    )
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--source-format", choices=("auto", "json", "jsonl", "csv"), default="csv")
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--title", default="Customer Ticket FAQ")
    parser.add_argument("--account-id", required=True)
    parser.add_argument("--user-id", default=None)
    parser.add_argument("--min-source-rows", type=int, default=1)
    parser.add_argument("--min-saved-faqs", type=int, default=1)
    parser.add_argument("--review-status", default=DEFAULT_REVIEW_STATUS)
    parser.add_argument("--export-limit", type=int, default=20)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--allow-ingestion-warnings", action="store_true")
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help="Fallback metadata applied to every source row. Repeat as key=value.",
    )
    parser.add_argument("--output-result", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--database-url", default=_default_database_url())
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.min_source_rows) < 1:
        raise SystemExit("--min-source-rows must be positive")
    if int(args.min_saved_faqs) < 1:
        raise SystemExit("--min-saved-faqs must be positive")
    if int(args.export_limit) < 1:
        raise SystemExit("--export-limit must be positive")
    if int(args.max_text_chars) < 1:
        raise SystemExit("--max-text-chars must be positive")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if not str(args.review_status or "").strip():
        raise SystemExit("--review-status must be non-empty")
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")


async def run_faq_lifecycle_smoke(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    default_fields = parse_default_fields_or_exit(args.default_field)
    source_path = Path(args.path)
    pool = await _create_pool(str(args.database_url))
    errors: list[str] = []
    source_rows = 0
    saved_ids: list[str] = []
    draft_export: dict[str, Any] | None = None
    reviewed_export: dict[str, Any] | None = None
    generation: dict[str, Any] | None = None
    normalization_warnings = _normalization_warning_summary([])
    input_profile = empty_input_profile(status="not_started")
    try:
        try:
            if not await _relation_exists(pool, "ticket_faq_markdown"):
                errors.append(
                    "required Content Ops table missing: ticket_faq_markdown. "
                    "Run python scripts/run_extracted_content_pipeline_migrations.py "
                    "--database-url \"$EXTRACTED_DATABASE_URL\" before this smoke."
                )
            raw_profile = raw_row_profile_or_error(
                source_path,
                str(args.source_format),
            )
            input_profile = empty_input_profile()
            input_profile.update(raw_profile)
            try:
                loaded = load_source_campaign_opportunities_from_file(
                    source_path,
                    file_format=str(args.source_format),
                    target_mode=str(args.target_mode),
                    max_text_chars=int(args.max_text_chars),
                    default_fields=default_fields,
                )
            except Exception as exc:
                input_profile = input_profile_error(exc, raw_profile=raw_profile)
                raise
            source_rows = len(loaded.opportunities)
            loaded_warnings = loaded.warning_dicts()
            normalization_warnings = _normalization_warning_summary(loaded_warnings)
            input_profile = input_profile_from_loaded(
                loaded,
                raw_profile=raw_profile,
            )
            if source_rows < int(args.min_source_rows):
                errors.append(
                    f"expected at least {int(args.min_source_rows)} source row(s), got {source_rows}"
                )
            if loaded_warnings and not bool(args.allow_ingestion_warnings):
                warning_counts = _warning_counts_label(normalization_warnings)
                errors.append(
                    f"source normalization produced warnings ({warning_counts}); "
                    "provide --default-field bindings or pass --allow-ingestion-warnings"
                )
            if not errors:
                repo = PostgresTicketFAQRepository(pool)
                service = TicketFAQMarkdownService(ticket_faqs=repo)
                result = await service.generate(
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                    target_mode=str(args.target_mode),
                    source_material={"opportunities": loaded.opportunities},
                    title=str(args.title),
                    max_text_chars=int(args.max_text_chars),
                )
                generation = result.as_dict()
                saved_ids = [str(item) for item in result.saved_ids if str(item).strip()]
                if len(saved_ids) < int(args.min_saved_faqs):
                    errors.append(
                        f"expected at least {int(args.min_saved_faqs)} saved FAQ(s), got {len(saved_ids)}"
                    )
            if not errors:
                repo = PostgresTicketFAQRepository(pool)
                draft_result = await export_ticket_faq_drafts(
                    repo,
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                    status="draft",
                    target_mode=str(args.target_mode),
                    limit=int(args.export_limit),
                )
                draft_export = draft_result.as_dict()
                errors.extend(_export_errors(draft_export, saved_ids, expected_status="draft"))
            if not errors:
                repo = PostgresTicketFAQRepository(pool)
                updated = await repo.update_status(
                    saved_ids[0],
                    str(args.review_status),
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                )
                if not updated:
                    errors.append(f"review status update missed saved FAQ id: {saved_ids[0]}")
            if not errors:
                repo = PostgresTicketFAQRepository(pool)
                reviewed_result = await export_ticket_faq_drafts(
                    repo,
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                    status=str(args.review_status),
                    target_mode=str(args.target_mode),
                    limit=int(args.export_limit),
                )
                reviewed_export = reviewed_result.as_dict()
                errors.extend(
                    _export_errors(
                        reviewed_export,
                        saved_ids[:1],
                        expected_status=str(args.review_status),
                    )
                )
        except Exception as exc:  # pragma: no cover - exercised by live hosts
            errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable
    payload = {
        "ok": not errors,
        "source": str(source_path),
        "source_format": str(args.source_format),
        "source_rows": source_rows,
        "input_profile": input_profile,
        "saved_ids": saved_ids,
        "generation": generation,
        "draft_export": draft_export,
        "reviewed_export": reviewed_export,
        "normalization_warnings": normalization_warnings,
        "review_status": str(args.review_status),
        "errors": errors,
    }
    payload["lifecycle_summary"] = _lifecycle_summary(payload)
    if args.output_result:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return (0 if not errors else 1), payload


def _normalization_warning_summary(warnings: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(warning.get("code") or "unknown") for warning in warnings)
    return {
        "warning_count": len(warnings),
        "warnings_by_code": dict(sorted(counts.items())),
        "warning_sample": [dict(warning) for warning in warnings[:10]],
        "warnings_truncated": len(warnings) > 10,
    }


def _warning_counts_label(summary: Mapping[str, Any]) -> str:
    counts = summary.get("warnings_by_code")
    if not isinstance(counts, Mapping) or not counts:
        return "unknown"
    return ", ".join(
        f"{code}={count}"
        for code, count in sorted(counts.items())
    )


async def _relation_exists(pool: Any, table_name: str) -> bool:
    value = await pool.fetchval("SELECT to_regclass($1)::text", table_name)
    return bool(value)


def _export_errors(
    export_payload: Mapping[str, Any],
    saved_ids: list[str],
    *,
    expected_status: str,
) -> list[str]:
    rows = export_payload.get("rows")
    if not isinstance(rows, list):
        return ["export rows missing"]
    by_id = {
        str(row.get("id") or ""): row
        for row in rows
        if isinstance(row, Mapping)
    }
    errors: list[str] = []
    for saved_id in saved_ids:
        row = by_id.get(saved_id)
        if row is None:
            errors.append(f"export missing saved FAQ id: {saved_id}")
            continue
        if str(row.get("status") or "") != expected_status:
            errors.append(
                f"exported FAQ {saved_id} had status {row.get('status')!r}, expected {expected_status!r}"
            )
        if not str(row.get("markdown") or "").strip():
            errors.append(f"exported FAQ {saved_id} missing markdown")
    return errors


def _lifecycle_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    generation = payload.get("generation")
    generation_payload = generation if isinstance(generation, Mapping) else {}
    items = generation_payload.get("items")
    output_checks = generation_payload.get("output_checks")
    return {
        "status": "ok" if payload.get("ok") else "failed",
        "source": payload.get("source"),
        "source_format": payload.get("source_format"),
        "source_rows": payload.get("source_rows"),
        "input_profile": _mapping_or_none(payload.get("input_profile")),
        "source_count": generation_payload.get("source_count"),
        "ticket_source_count": generation_payload.get("ticket_source_count"),
        "generated_item_count": len(items) if isinstance(items, list) else None,
        "output_checks": _mapping_or_none(output_checks),
        "saved_faq_count": len(payload.get("saved_ids") or []),
        "draft_export_count": _export_row_count(payload.get("draft_export")),
        "reviewed_export_count": _export_row_count(payload.get("reviewed_export")),
        "review_status": payload.get("review_status"),
        "error_count": len(payload.get("errors") or []),
        "errors": list(payload.get("errors") or []),
    }


def _export_row_count(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    rows = value.get("rows")
    return len(rows) if isinstance(rows, list) else None


def _mapping_or_none(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    profile = console_input_profile(payload.get("input_profile"))
    if payload.get("ok"):
        print(
            "Content Ops FAQ lifecycle smoke passed: "
            f"{profile} "
            f"saved_faqs={len(payload.get('saved_ids') or [])} "
            f"review_status={payload.get('review_status')}"
        )
        return
    print(f"Content Ops FAQ lifecycle smoke failed: {profile}", file=sys.stderr)
    for error in payload.get("errors") or []:
        print(f"- {error}", file=sys.stderr)


async def _main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = _parse_args(argv)
    _validate_args(args)
    code, payload = await run_faq_lifecycle_smoke(args)
    _print_payload(payload, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
