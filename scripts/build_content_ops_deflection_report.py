#!/usr/bin/env python3
"""Build a customer-facing support-ticket deflection report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
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
from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    build_deflection_report_artifact,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
    DEFAULT_INTENT_RULES,
    DEFAULT_TITLE,
    build_ticket_faq_markdown,
)


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

    artifact = build_report(args)
    failed_checks = _failed_output_checks(artifact.faq_result.output_checks)
    if args.result_output:
        args.result_output.write_text(
            json.dumps(
                _result_payload(args, artifact, failed_checks),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if args.require_output_checks and failed_checks:
        raise SystemExit(
            f"Deflection report output checks failed: {', '.join(failed_checks)}"
        )
    if args.output:
        args.output.write_text(artifact.markdown, encoding="utf-8")
    else:
        print(artifact.markdown, end="")
    if args.summary_output:
        args.summary_output.write_text(
            json.dumps(artifact.summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(artifact.summary, sort_keys=True))
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Source JSON, JSONL, or CSV file.")
    parser.add_argument(
        "--source-format",
        choices=("auto", "json", "jsonl", "csv"),
        default="auto",
        help="Source file format. Defaults to suffix-based detection.",
    )
    parser.add_argument("--title", default="Support Ticket Deflection Report")
    parser.add_argument("--faq-title", default=DEFAULT_TITLE)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--max-evidence-per-item", type=int, default=3)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--window-days", type=int)
    parser.add_argument("--as-of-date")
    parser.add_argument("--support-contact")
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help="Fallback metadata applied to every source row. Repeat as key=value.",
    )
    parser.add_argument(
        "--documentation-term",
        action="append",
        default=[],
        help="Existing documentation term or heading. Repeat to provide multiple terms.",
    )
    parser.add_argument(
        "--vocabulary-gap-rule",
        action="append",
        default=[],
        help="Comma-separated customer/documentation aliases. Repeat for multiple rules.",
    )
    parser.add_argument(
        "--intent-rule",
        action="append",
        default=[],
        help="Custom intent mapping shaped as topic=keyword,keyword. Repeat for multiple rules.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--result-output", type=Path)
    parser.add_argument(
        "--require-output-checks",
        action="store_true",
        help="Exit non-zero when generated FAQ output checks fail.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace):
    custom_intent_rules = _parse_intent_rules(args.intent_rule)
    args.custom_intent_rules = custom_intent_rules
    intent_rules = (*custom_intent_rules, *DEFAULT_INTENT_RULES)
    loaded = load_source_campaign_opportunities_from_file(
        args.path,
        file_format=args.source_format,
        max_text_chars=args.max_text_chars,
        default_fields=parse_default_fields_or_exit(args.default_field),
    )
    faq_result = build_ticket_faq_markdown(
        loaded.opportunities,
        title=args.faq_title,
        max_items=args.max_items,
        max_evidence_per_item=args.max_evidence_per_item,
        window_days=args.window_days,
        as_of_date=args.as_of_date,
        support_contact=args.support_contact,
        intent_rules=intent_rules,
        documentation_terms=tuple(args.documentation_term or ()),
        vocabulary_gap_rules=_parse_vocabulary_gap_rules(args.vocabulary_gap_rule),
    )
    return build_deflection_report_artifact(
        faq_result,
        title=args.title,
        source_label=str(args.path),
    )


def _parse_vocabulary_gap_rules(values: list[str]) -> tuple[tuple[str, ...], ...]:
    rules: list[tuple[str, ...]] = []
    for raw in values or ():
        parts = tuple(part.strip() for part in str(raw).split(",") if part.strip())
        if len(parts) < 2:
            raise SystemExit("--vocabulary-gap-rule must contain at least two comma-separated terms")
        rules.append(parts)
    return tuple(rules)


def _parse_intent_rules(values: list[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rules: list[tuple[str, tuple[str, ...]]] = []
    for value in values or ():
        topic, separator, raw_keywords = str(value).partition("=")
        topic = topic.strip()
        keywords = _parse_intent_rule_keywords(raw_keywords)
        if not separator or not topic or not keywords:
            raise SystemExit(
                "--intent-rule must use topic=keyword,keyword with at least one keyword"
            )
        rules.append((topic, keywords))
    return tuple(rules)


def _parse_intent_rule_keywords(value: str) -> tuple[str, ...]:
    keywords: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        keyword = part.strip()
        key = keyword.lower()
        if not keyword or key in seen:
            continue
        seen.add(key)
        keywords.append(keyword)
    return tuple(keywords)


def _failed_output_checks(output_checks: Mapping[str, Any]) -> list[str]:
    return [name for name, passed in sorted(output_checks.items()) if passed is not True]


def _result_payload(
    args: argparse.Namespace,
    artifact: Any,
    failed_checks: list[str],
) -> dict[str, Any]:
    faq_result = artifact.faq_result
    items = [dict(item) for item in faq_result.items]
    return {
        "status": "failed_output_checks" if failed_checks else "ok",
        "input": {
            "path": str(args.path),
            "source_format": args.source_format,
        },
        "output": {
            "markdown_path": str(args.output) if args.output else None,
            "summary_path": str(args.summary_output) if args.summary_output else None,
            "result_path": str(args.result_output) if args.result_output else None,
        },
        "config": {
            "title": args.title,
            "faq_title": args.faq_title,
            "max_items": args.max_items,
            "max_evidence_per_item": args.max_evidence_per_item,
            "max_text_chars": args.max_text_chars,
            "window_days": args.window_days,
            "as_of_date": args.as_of_date,
            "require_output_checks": bool(args.require_output_checks),
            "support_contact": args.support_contact,
            "documentation_terms": list(args.documentation_term or ()),
            "vocabulary_gap_rules": [
                list(rule) for rule in _parse_vocabulary_gap_rules(args.vocabulary_gap_rule)
            ],
            "custom_intent_rules": [
                {"topic": topic, "keywords": list(keywords)}
                for topic, keywords in getattr(args, "custom_intent_rules", ())
            ],
        },
        "summary": dict(artifact.summary),
        "source_count": faq_result.source_count,
        "ticket_source_count": faq_result.ticket_source_count,
        "generated": len(items),
        "output_checks": dict(faq_result.output_checks),
        "failed_output_checks": list(failed_checks),
        "diagnostics": {
            "item_count": len(items),
            "items": [
                _item_summary(index, item)
                for index, item in enumerate(items, start=1)
            ],
        },
    }


def _item_summary(rank: int, item: Mapping[str, Any]) -> dict[str, Any]:
    source_ids = _texts(item.get("source_ids"))
    return {
        "rank": rank,
        "topic": _text(item.get("topic")),
        "question": _text(item.get("question")),
        "answer_evidence_status": _text(item.get("answer_evidence_status")),
        "ticket_count": _int(item.get("ticket_count")),
        "opportunity_score": _int(item.get("opportunity_score")),
        "source_id_count": len(source_ids),
        "first_source_id": source_ids[0] if source_ids else "",
        "step_count": len(_texts(item.get("steps"))),
        "evidence_count": len(_texts(item.get("evidence_quotes"))),
        "term_mapping_count": len(_mappings(item.get("term_mappings"))),
    }


def _mappings(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _texts(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, list | tuple):
        values = value
    else:
        values = (value,)
    return [text for raw in values if (text := _text(raw))]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
