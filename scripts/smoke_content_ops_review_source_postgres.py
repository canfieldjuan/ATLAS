#!/usr/bin/env python3
"""Smoke-test review-source export through Postgres-backed draft persistence."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_ports import TenantScope  # noqa: E402
from extracted_content_pipeline.campaign_postgres_import import (  # noqa: E402
    import_campaign_opportunities,
)
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ingestion_diagnostics import (  # noqa: E402
    inspect_ingestion_file,
)

def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_source_postgres_helpers = _load_script_module(
    "content_ops_source_postgres_smoke_helpers",
    ROOT / "scripts" / "content_ops_source_postgres_smoke_helpers.py",
)
_source_smoke = _load_script_module(
    "content_ops_review_source_generation_smoke",
    ROOT / "scripts" / "smoke_content_ops_review_source_generation.py",
)

DEFAULT_CHANNELS = _source_smoke.DEFAULT_CHANNELS
DEFAULT_FORBIDDEN_PHRASES = _source_smoke.DEFAULT_FORBIDDEN_PHRASES
DEFAULT_PHRASE_FIELDS = _source_smoke.DEFAULT_PHRASE_FIELDS
DEFAULT_POLARITIES = _source_smoke.DEFAULT_POLARITIES
DEFAULT_SUMMARY_SOURCES = _source_smoke.DEFAULT_SUMMARY_SOURCES
_create_pool = _source_smoke._create_pool
_csv_list = _source_smoke._csv_list
_default_database_url = _source_smoke._default_database_url
_load_dotenv_files = _source_smoke._load_dotenv_files
_row_for_source = _source_smoke._row_for_source
_write_jsonl = _source_smoke._write_jsonl
fetch_review_source_rows = _source_smoke.fetch_review_source_rows
fetch_review_source_summary = _source_smoke.fetch_review_source_summary
draft_errors = _source_postgres_helpers.draft_errors
fetch_saved_drafts = _source_postgres_helpers.fetch_saved_drafts
generate_imported_target_drafts = _source_postgres_helpers.generate_imported_target_drafts
generation_errors = _source_postgres_helpers.generation_errors
saved_draft_target_errors = _source_postgres_helpers.saved_draft_target_errors
schema_readiness_errors = _source_postgres_helpers.schema_readiness_errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Atlas review-source rows through Content Ops source "
            "export, Postgres import, and DB-backed offline draft generation."
        )
    )
    parser.add_argument("--source", default="g2")
    parser.add_argument("--vendor", default=None, help="Optional vendor_name filter.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--channels", default=",".join(DEFAULT_CHANNELS))
    parser.add_argument("--min-drafts", type=int, default=None)
    parser.add_argument("--min-quote-grade-rows", type=int, default=1)
    parser.add_argument("--min-review-text-chars", type=int, default=80)
    parser.add_argument("--phrase-limit", type=int, default=5)
    parser.add_argument("--polarities", default=",".join(DEFAULT_POLARITIES))
    parser.add_argument("--phrase-fields", default=",".join(DEFAULT_PHRASE_FIELDS))
    parser.add_argument("--summary-sources", default=",".join(DEFAULT_SUMMARY_SOURCES))
    parser.add_argument("--allow-missing-source-url", action="store_true")
    parser.add_argument("--allow-ingestion-warnings", action="store_true")
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help=(
            "Fallback metadata applied to every exported review source row. "
            "Repeat as key=value."
        ),
    )
    parser.add_argument(
        "--account-id",
        required=True,
        help="Tenant/account id used to scope imported opportunities and drafts.",
    )
    parser.add_argument("--user-id", default=None)
    parser.add_argument(
        "--opportunity-table",
        default="campaign_opportunities",
        help="Target opportunity table for import and generation.",
    )
    parser.add_argument(
        "--keep-existing-opportunities",
        action="store_true",
        help="Append imported opportunities instead of replacing matching target ids.",
    )
    parser.add_argument(
        "--forbidden-phrase",
        action="append",
        default=list(DEFAULT_FORBIDDEN_PHRASES),
        help="Fail if generated draft bodies contain this phrase. Repeatable.",
    )
    parser.add_argument("--output-source-rows", type=Path, default=None)
    parser.add_argument("--output-result", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--database-url",
        default=_default_database_url(),
        help="Postgres DSN. Defaults to EXTRACTED_DATABASE_URL or DATABASE_URL.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.limit) < 1:
        raise SystemExit("--limit must be positive")
    if int(args.min_quote_grade_rows) < 1:
        raise SystemExit("--min-quote-grade-rows must be positive")
    if int(args.min_review_text_chars) < 1:
        raise SystemExit("--min-review-text-chars must be positive")
    if int(args.phrase_limit) < 1:
        raise SystemExit("--phrase-limit must be positive")
    if args.min_drafts is not None and int(args.min_drafts) < 1:
        raise SystemExit("--min-drafts must be positive")
    if not _csv_list(args.channels):
        raise SystemExit("--channels must include at least one channel")
    if not str(args.account_id or "").strip():
        raise SystemExit("--account-id is required")
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")


async def _fetch_review_inputs(
    pool: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_rows = await fetch_review_source_summary(
        pool,
        sources=_csv_list(args.summary_sources),
        min_review_text_chars=int(args.min_review_text_chars),
        allowed_polarities=_csv_list(args.polarities),
        allowed_fields=_csv_list(args.phrase_fields),
        require_review_url=not bool(args.allow_missing_source_url),
    )
    source_summary = _row_for_source(summary_rows, str(args.source))
    if int(source_summary.get("quote_grade_rows") or 0) < int(args.min_quote_grade_rows):
        return source_summary, []
    source_rows = await fetch_review_source_rows(
        pool,
        source=str(args.source),
        vendor_name=args.vendor,
        limit=int(args.limit),
        min_review_text_chars=int(args.min_review_text_chars),
        phrase_limit=int(args.phrase_limit),
        allowed_polarities=_csv_list(args.polarities),
        allowed_fields=_csv_list(args.phrase_fields),
        require_review_url=not bool(args.allow_missing_source_url),
    )
    return source_summary, source_rows


async def run_review_source_postgres_smoke(
    args: argparse.Namespace,
    *,
    source_rows_path: Path,
) -> tuple[int, dict[str, Any]]:
    default_fields = parse_default_fields_or_exit(args.default_field)
    channels = _csv_list(args.channels)
    min_drafts = (
        int(args.min_drafts)
        if args.min_drafts is not None
        else int(args.limit) * len(channels)
    )
    pool = await _create_pool(str(args.database_url))
    source_summary: dict[str, Any] = {}
    source_rows: list[dict[str, Any]] = []
    ingestion_report: dict[str, Any] | None = None
    import_result: dict[str, Any] | None = None
    drafts_result: dict[str, Any] | None = None
    saved_drafts: list[dict[str, Any]] = []
    errors: list[str] = []
    imported_target_ids: list[str] = []
    try:
        try:
            source_summary, source_rows = await _fetch_review_inputs(pool, args)
            quote_grade_ready = int(source_summary.get("quote_grade_rows") or 0) >= int(
                args.min_quote_grade_rows
            )
            if not quote_grade_ready:
                errors.append(
                    "source has fewer quote-grade rows than required: "
                    f"{source_summary.get('quote_grade_rows', 0)}"
                )
            if quote_grade_ready and len(source_rows) < int(args.limit):
                errors.append(
                    f"expected {int(args.limit)} exported source row(s), got {len(source_rows)}"
                )

            if not errors:
                _write_jsonl(source_rows, source_rows_path)
                report = inspect_ingestion_file(
                    source_rows_path,
                    source_rows=True,
                    source_format="jsonl",
                    target_mode=str(args.target_mode),
                    default_fields=default_fields,
                )
                ingestion_report = report.as_dict()
                if not report.ok:
                    errors.append("ingestion inspection failed")
                if report.warnings and not bool(args.allow_ingestion_warnings):
                    errors.append(
                        "ingestion inspection produced warnings; provide --default-field "
                        "bindings or pass --allow-ingestion-warnings"
                    )

            if not errors:
                errors.extend(
                    await schema_readiness_errors(
                        pool,
                        opportunity_table=str(args.opportunity_table),
                    )
                )

            if not errors:
                loaded = load_source_campaign_opportunities_from_file(
                    source_rows_path,
                    file_format="jsonl",
                    target_mode=str(args.target_mode),
                    default_fields=default_fields,
                )
                imported = await import_campaign_opportunities(
                    pool,
                    loaded.opportunities,
                    scope=TenantScope(account_id=str(args.account_id), user_id=args.user_id),
                    target_mode=str(args.target_mode),
                    opportunity_table=str(args.opportunity_table),
                    replace_existing=not bool(args.keep_existing_opportunities),
                    normalize=False,
                    warnings=loaded.warnings,
                    source=loaded.source,
                )
                import_result = imported.as_dict()
                imported_target_ids = list(imported.target_ids)
                if imported.skipped:
                    errors.append(f"import skipped {imported.skipped} row(s)")
                if imported.inserted < int(args.limit):
                    errors.append(
                        f"expected {int(args.limit)} imported opportunity row(s), "
                        f"got {imported.inserted}"
                    )

            if not errors:
                drafts_result = await generate_imported_target_drafts(
                    pool=pool,
                    account_id=str(args.account_id),
                    user_id=args.user_id,
                    target_mode=str(args.target_mode),
                    channels=channels,
                    target_ids=imported_target_ids,
                    opportunity_table=str(args.opportunity_table),
                )
                saved_drafts = await fetch_saved_drafts(pool, drafts_result.get("saved_ids") or [])
                errors.extend(generation_errors(drafts_result))
                errors.extend(
                    draft_errors(
                        {"drafts": saved_drafts},
                        min_drafts=min_drafts,
                        forbidden_phrases=args.forbidden_phrase,
                    )
                )
                if int(drafts_result.get("generated") or 0) < min_drafts:
                    errors.append(
                        f"expected at least {min_drafts} persisted draft(s), "
                        f"got {drafts_result.get('generated', 0)}"
                    )
                errors.extend(saved_draft_target_errors(saved_drafts, imported_target_ids))
        except Exception as exc:  # pragma: no cover - exercised by live host DBs
            errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable

    payload = {
        "ok": not errors,
        "source": str(args.source),
        "vendor": args.vendor,
        "source_summary": source_summary,
        "source_rows": len(source_rows),
        "source_rows_path": str(source_rows_path),
        "ingestion": ingestion_report,
        "import": import_result,
        "drafts": drafts_result,
        "saved_drafts": saved_drafts,
        "errors": errors,
    }
    if args.output_result:
        args.output_result.parent.mkdir(parents=True, exist_ok=True)
        args.output_result.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return (0 if not errors else 1), payload

def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    if payload.get("ok"):
        imported = payload.get("import") if isinstance(payload.get("import"), Mapping) else {}
        drafts = payload.get("drafts") if isinstance(payload.get("drafts"), Mapping) else {}
        print(
            "Content Ops review-source Postgres smoke passed: "
            f"source={payload.get('source')} "
            f"source_rows={payload.get('source_rows')} "
            f"imported={imported.get('inserted', 0)} "
            f"generated={drafts.get('generated', 0)}"
        )
        return
    print("Content Ops review-source Postgres smoke failed:", file=sys.stderr)
    for error in payload.get("errors") or []:
        print(f"- {error}", file=sys.stderr)


async def _main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = _parse_args(argv)
    _validate_args(args)
    if args.output_source_rows:
        code, payload = await run_review_source_postgres_smoke(
            args,
            source_rows_path=args.output_source_rows,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="content-ops-review-source-db-") as tmpdir:
            code, payload = await run_review_source_postgres_smoke(
                args,
                source_rows_path=Path(tmpdir) / "review_sources.jsonl",
            )
    _print_payload(payload, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
