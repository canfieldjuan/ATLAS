#!/usr/bin/env python3
"""Build a grounded FAQ Markdown file from ticket-like source rows."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
    parse_default_fields_or_exit,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    DEFAULT_TITLE,
    build_ticket_faq_markdown,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ticket-like source rows into a grounded Markdown FAQ."
        )
    )
    parser.add_argument("path", type=Path, help="Source JSON, JSONL, or CSV file.")
    parser.add_argument(
        "--source-format",
        choices=("auto", "json", "jsonl", "csv"),
        default="auto",
        help="Source file format. Defaults to suffix-based detection.",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Markdown H1 title.",
    )
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--max-evidence-per-item", type=int, default=3)
    parser.add_argument(
        "--window-days",
        type=int,
        help="Keep only rows dated within this many days of --as-of-date or today.",
    )
    parser.add_argument(
        "--as-of-date",
        help="YYYY-MM-DD date used with --window-days for reproducible windows.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Maximum source text characters copied into each evidence row.",
    )
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help="Fallback metadata applied to every source row. Repeat as key=value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write FAQ Markdown to this path instead of stdout.",
    )
    parser.add_argument(
        "--result-output",
        type=Path,
        help="Write compact JSON run diagnostics to this path.",
    )
    parser.add_argument(
        "--support-contact",
        help="Phone, email, or URL shown when the FAQ tells users to contact support.",
    )
    parser.add_argument(
        "--documentation-term",
        action="append",
        default=[],
        help=(
            "Existing documentation term or heading used for vocabulary-gap "
            "suggestions. Repeat to provide multiple terms."
        ),
    )
    parser.add_argument(
        "--require-output-checks",
        action="store_true",
        help="Fail when generated FAQ output checks are not all true.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_items < 1:
        raise SystemExit("--max-items must be positive")
    if args.max_evidence_per_item < 1:
        raise SystemExit("--max-evidence-per-item must be positive")
    if args.max_text_chars < 1:
        raise SystemExit("--max-text-chars must be positive")
    if args.window_days is not None and args.window_days < 1:
        raise SystemExit("--window-days must be positive")
    if args.as_of_date and args.window_days is None:
        raise SystemExit("--as-of-date requires --window-days")
    if args.as_of_date:
        try:
            date.fromisoformat(args.as_of_date)
        except ValueError:
            raise SystemExit("--as-of-date must use YYYY-MM-DD format") from None

    loaded = load_source_campaign_opportunities_from_file(
        args.path,
        file_format=args.source_format,
        max_text_chars=args.max_text_chars,
        default_fields=parse_default_fields_or_exit(args.default_field),
    )
    result = build_ticket_faq_markdown(
        loaded.opportunities,
        title=args.title,
        max_items=args.max_items,
        max_evidence_per_item=args.max_evidence_per_item,
        window_days=args.window_days,
        as_of_date=args.as_of_date,
        support_contact=args.support_contact,
        documentation_terms=args.documentation_term,
    )
    failed_checks = _failed_output_checks(result.output_checks)
    if args.result_output:
        args.result_output.write_text(
            json.dumps(
                _result_payload(args, result, loaded.warning_dicts(), failed_checks),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.require_output_checks and failed_checks:
        raise SystemExit(f"FAQ output checks failed: {', '.join(failed_checks)}")
    if args.output:
        args.output.write_text(result.markdown, encoding="utf-8")
    else:
        print(result.markdown, end="")
    return 0


def _failed_output_checks(output_checks: dict[str, bool]) -> list[str]:
    return [name for name, passed in sorted(output_checks.items()) if passed is not True]


def _result_payload(
    args: argparse.Namespace,
    result: Any,
    load_warnings: list[dict[str, Any]],
    failed_checks: list[str],
) -> dict[str, Any]:
    items = [dict(item) for item in result.items]
    ticket_counts = [int(item.get("ticket_count") or 0) for item in items]
    question_source_counts = _count_by(items, "question_source")
    rendered_ticket_source_count = _rendered_ticket_source_count(items)
    warnings = load_warnings + [dict(warning) for warning in result.warnings]
    return {
        "status": "failed_output_checks" if failed_checks else "ok",
        "input": {
            "path": str(args.path),
            "source_format": args.source_format,
        },
        "output": {
            "markdown_path": str(args.output) if args.output else None,
            "result_path": str(args.result_output) if args.result_output else None,
        },
        "config": {
            "title": args.title,
            "max_items": args.max_items,
            "max_evidence_per_item": args.max_evidence_per_item,
            "max_text_chars": args.max_text_chars,
            "window_days": args.window_days,
            "as_of_date": args.as_of_date,
            "require_output_checks": bool(args.require_output_checks),
            "support_contact": args.support_contact,
            "documentation_terms": list(args.documentation_term),
        },
        "source_count": result.source_count,
        "ticket_source_count": result.ticket_source_count,
        "generated": len(items),
        "output_checks": dict(result.output_checks),
        "failed_output_checks": list(failed_checks),
        "diagnostics": {
            "output_check_details": _output_check_details(result, failed_checks, rendered_ticket_source_count),
            "question_source_counts": question_source_counts,
            "ticket_counts": ticket_counts,
            "rendered_ticket_source_count": rendered_ticket_source_count,
            "unrepresented_ticket_sources": max(result.ticket_source_count - rendered_ticket_source_count, 0),
            "term_mapping_count": _term_mapping_count(items),
            "term_mappings": _term_mapping_summaries(items),
            "warning_count": len(warnings),
            "warnings": warnings[:50],
            "warnings_truncated": len(warnings) > 50,
            "items": [_item_summary(index, item) for index, item in enumerate(items, start=1)],
        },
    }


def _output_check_details(
    result: Any,
    failed_checks: list[str],
    rendered_ticket_source_count: int,
) -> list[dict[str, Any]]:
    details = []
    for name, passed in sorted(result.output_checks.items()):
        detail: dict[str, Any] = {"check": name, "passed": bool(passed)}
        if name in failed_checks:
            detail["why"] = _output_check_hint(name, result, rendered_ticket_source_count)
        details.append(detail)
    return details


def _output_check_hint(name: str, result: Any, rendered_ticket_source_count: int) -> str:
    if name == "condensed":
        if rendered_ticket_source_count != result.ticket_source_count:
            return (
                "Some ticket sources were not represented in generated FAQ items. "
                f"ticket_source_count={result.ticket_source_count}, "
                f"rendered_ticket_source_count={rendered_ticket_source_count}."
            )
        return (
            "The FAQ produced one item per ticket source, so the output was not condensed. "
            f"ticket_source_count={result.ticket_source_count}, generated={len(result.items)}."
        )
    if name == "uses_user_vocabulary":
        return "One or more FAQ questions did not come from customer wording or source policy."
    if name == "has_action_items":
        return "One or more FAQ items did not include actionable next steps."
    return "Output check failed."


def _item_summary(index: int, item: dict[str, Any]) -> dict[str, Any]:
    source_ids = item.get("source_ids") or ()
    steps = item.get("steps") or ()
    term_mappings = item.get("term_mappings") or ()
    return {
        "rank": index,
        "topic": item.get("topic"),
        "question": item.get("question"),
        "question_source": item.get("question_source"),
        "frequency": item.get("frequency"),
        "failure_risk_score": item.get("failure_risk_score"),
        "failure_risk_signals": list(item.get("failure_risk_signals") or ()),
        "opportunity_score": item.get("opportunity_score"),
        "ticket_count": item.get("ticket_count"),
        "evidence_count": item.get("evidence_count"),
        "source_id_count": len(source_ids),
        "first_source_id": source_ids[0] if source_ids else None,
        "step_count": len(steps),
        "term_mapping_count": len(term_mappings),
    }


def _term_mapping_count(items: list[dict[str, Any]]) -> int:
    return sum(len(item.get("term_mappings") or ()) for item in items)


def _term_mapping_summaries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        for mapping in item.get("term_mappings") or ():
            if not isinstance(mapping, dict):
                continue
            out.append({
                "rank": index,
                "topic": item.get("topic"),
                "customer_term": mapping.get("customer_term"),
                "documentation_term": mapping.get("documentation_term"),
                "source_id_count": mapping.get("source_id_count"),
                "first_source_id": mapping.get("first_source_id"),
            })
    return out


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _rendered_ticket_source_count(items: list[dict[str, Any]]) -> int:
    source_ids: set[str] = set()
    for item in items:
        values = item.get("source_ids") or ()
        source_ids.update(str(value) for value in values if str(value))
    return len(source_ids)


if __name__ == "__main__":
    raise SystemExit(main())
