#!/usr/bin/env python3
"""Build a customer-facing support-ticket deflection report."""

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
from extracted_content_pipeline.faq_deflection_report import (  # noqa: E402
    build_deflection_report_artifact,
)
from extracted_content_pipeline.ticket_faq_markdown import (  # noqa: E402
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout.")
    return parser.parse_args(argv)


def build_report(args: argparse.Namespace):
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


if __name__ == "__main__":
    raise SystemExit(main())
