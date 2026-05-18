#!/usr/bin/env python3
"""Smoke-test review-source export through offline Content Ops generation."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_example import (  # noqa: E402
    generate_campaign_drafts_from_payload,
)
from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ingestion_diagnostics import (  # noqa: E402
    inspect_ingestion_file,
)


def _load_review_exporter_module():
    path = ROOT / "scripts/export_content_ops_review_sources.py"
    spec = importlib.util.spec_from_file_location("content_ops_review_source_exporter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load review source exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_review_exporter = _load_review_exporter_module()
DEFAULT_PHRASE_FIELDS = _review_exporter.DEFAULT_PHRASE_FIELDS
DEFAULT_POLARITIES = _review_exporter.DEFAULT_POLARITIES
DEFAULT_SUMMARY_SOURCES = _review_exporter.DEFAULT_SUMMARY_SOURCES
_create_pool = _review_exporter._create_pool
_default_database_url = _review_exporter._default_database_url
_load_dotenv_files = _review_exporter._load_dotenv_files
fetch_review_source_rows = _review_exporter.fetch_review_source_rows
fetch_review_source_summary = _review_exporter.fetch_review_source_summary
render_jsonl = _review_exporter.render_jsonl


DEFAULT_CHANNELS = ("email_cold", "email_followup")
DEFAULT_FORBIDDEN_PHRASES = ("appears to be weighing",)


def _csv_list(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = value.split(",")
    else:
        raw = value
    return tuple(str(item or "").strip() for item in raw if str(item or "").strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test Atlas review-source rows through Content Ops source "
            "export, ingestion inspection, and offline draft generation."
        )
    )
    parser.add_argument("--source", default="g2")
    parser.add_argument("--vendor", default=None, help="Optional vendor_name filter.")
    parser.add_argument("--limit", type=int, default=2)
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
        "--forbidden-phrase",
        action="append",
        default=list(DEFAULT_FORBIDDEN_PHRASES),
        help="Fail if generated draft bodies contain this phrase. Repeatable.",
    )
    parser.add_argument("--output-source-rows", type=Path, default=None)
    parser.add_argument("--output-drafts", type=Path, default=None)
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
    if not args.database_url:
        raise SystemExit("Missing --database-url, EXTRACTED_DATABASE_URL, or DATABASE_URL")


def _row_for_source(summary_rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Any]:
    normalized = str(source or "").strip().lower()
    for row in summary_rows:
        if str(row.get("source") or "").strip().lower() == normalized:
            return dict(row)
    return {
        "source": normalized,
        "total_rows": 0,
        "canonical_rows": 0,
        "enriched_rows": 0,
        "export_candidate_rows": 0,
        "quote_grade_rows": 0,
    }


def _write_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render_jsonl(rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _draft_errors(
    result: Mapping[str, Any],
    *,
    min_drafts: int,
    forbidden_phrases: Sequence[str],
) -> list[str]:
    drafts = result.get("drafts")
    if not isinstance(drafts, list):
        return ["result.drafts is missing or not a list"]
    if len(drafts) < min_drafts:
        return [f"expected at least {min_drafts} draft(s), got {len(drafts)}"]
    errors: list[str] = []
    forbidden = [phrase.lower() for phrase in forbidden_phrases if phrase]
    for index, draft in enumerate(drafts[:min_drafts], start=1):
        if not isinstance(draft, Mapping):
            errors.append(f"draft {index} is not an object")
            continue
        for field in ("subject", "body", "target_id", "channel"):
            if not str(draft.get(field) or "").strip():
                errors.append(f"draft {index} missing {field}")
        body = str(draft.get("body") or "").lower()
        for phrase in forbidden:
            if phrase in body:
                errors.append(f"draft {index} contains forbidden phrase: {phrase}")
    return errors


async def _fetch_review_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = await _create_pool(str(args.database_url))
    try:
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
    finally:
        close = getattr(pool, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if hasattr(maybe_awaitable, "__await__"):
                await maybe_awaitable


async def run_review_source_generation_smoke(
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
    source_summary, source_rows = await _fetch_review_inputs(args)
    errors: list[str] = []
    quote_grade_ready = int(source_summary.get("quote_grade_rows") or 0) >= int(
        args.min_quote_grade_rows
    )
    if not quote_grade_ready:
        errors.append(
            "source has fewer quote-grade rows than required: "
            f"{source_summary.get('quote_grade_rows', 0)}"
        )
    if quote_grade_ready and len(source_rows) < int(args.limit):
        errors.append(f"expected {int(args.limit)} exported source row(s), got {len(source_rows)}")

    drafts_result: dict[str, Any] | None = None
    ingestion_report: dict[str, Any] | None = None
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
            payload = load_source_campaign_opportunities_from_file(
                source_rows_path,
                file_format="jsonl",
                default_fields=default_fields,
            ).as_payload()
            payload["target_mode"] = str(args.target_mode)
            payload["limit"] = int(args.limit)
            payload["channels"] = list(channels)
            drafts_result = await generate_campaign_drafts_from_payload(payload)
            errors.extend(
                _draft_errors(
                    drafts_result,
                    min_drafts=min_drafts,
                    forbidden_phrases=args.forbidden_phrase,
                )
            )

    payload = {
        "ok": not errors,
        "source": str(args.source),
        "vendor": args.vendor,
        "source_summary": source_summary,
        "source_rows": len(source_rows),
        "source_rows_path": str(source_rows_path),
        "ingestion": ingestion_report,
        "drafts": drafts_result,
        "errors": errors,
    }
    if args.output_drafts and drafts_result is not None:
        args.output_drafts.parent.mkdir(parents=True, exist_ok=True)
        args.output_drafts.write_text(
            json.dumps(drafts_result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        payload["drafts_path"] = str(args.output_drafts)
    return (0 if not errors else 1), payload


def _print_payload(payload: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(dict(payload), indent=2, sort_keys=True, default=str))
        return
    if payload.get("ok"):
        drafts = payload.get("drafts") if isinstance(payload.get("drafts"), Mapping) else {}
        result = drafts.get("result") if isinstance(drafts.get("result"), Mapping) else {}
        print(
            "Content Ops review-source smoke passed: "
            f"source={payload.get('source')} "
            f"source_rows={payload.get('source_rows')} "
            f"generated={result.get('generated', 0)}"
        )
        return
    print("Content Ops review-source smoke failed:", file=sys.stderr)
    for error in payload.get("errors") or []:
        print(f"- {error}", file=sys.stderr)


async def _main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = _parse_args(argv)
    _validate_args(args)
    if args.output_source_rows:
        code, payload = await run_review_source_generation_smoke(
            args,
            source_rows_path=args.output_source_rows,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="content-ops-review-source-") as tmpdir:
            code, payload = await run_review_source_generation_smoke(
                args,
                source_rows_path=Path(tmpdir) / "review_sources.jsonl",
            )
    _print_payload(payload, as_json=bool(args.json))
    return code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
