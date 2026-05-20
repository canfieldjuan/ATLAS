#!/usr/bin/env python3
"""Build a grounded FAQ Markdown file from ticket-like source rows."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


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
            "Convert support-ticket, case, conversation, or complaint source "
            "rows into a grounded Markdown FAQ."
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
    )
    if args.output:
        args.output.write_text(result.markdown, encoding="utf-8")
    else:
        print(result.markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
