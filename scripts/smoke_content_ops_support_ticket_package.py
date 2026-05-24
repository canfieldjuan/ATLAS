#!/usr/bin/env python3
"""Smoke-test support-ticket input packaging before DB or LLM work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extracted_content_pipeline.campaign_source_adapters import (
    SourceDataFormat,
    load_source_rows_from_file,
)
from extracted_content_pipeline.support_ticket_input_package import (
    DEFAULT_SUPPORT_TICKET_OUTPUTS,
    build_support_ticket_input_package,
)


def build_support_ticket_package_smoke_summary(
    path: str | Path,
    *,
    file_format: SourceDataFormat = "auto",
    outputs: tuple[str, ...] = DEFAULT_SUPPORT_TICKET_OUTPUTS,
    window_days: int = 90,
    max_rows: int = 1000,
    require_included_rows: bool = False,
) -> dict[str, Any]:
    """Return a compact summary of the support-ticket package built from a file."""

    rows = load_source_rows_from_file(path, file_format=file_format)
    package = build_support_ticket_input_package(
        rows,
        outputs=outputs,
        window_days=window_days,
        max_rows=max_rows,
    )
    inputs = package.inputs
    included_row_count = int(inputs.get("included_ticket_row_count") or 0)
    if require_included_rows and included_row_count < 1:
        raise ValueError("No usable support-ticket rows survived packaging.")
    customer_wording_examples = list(inputs.get("customer_wording_examples") or [])
    faq_questions = list(inputs.get("faq_questions") or [])
    return {
        "path": str(Path(path)),
        "provider": package.provider,
        "outputs": list(package.outputs),
        "source_row_count": int(inputs.get("source_row_count") or 0),
        "included_ticket_row_count": included_row_count,
        "skipped_ticket_row_count": int(inputs.get("skipped_ticket_row_count") or 0),
        "truncated_ticket_row_count": int(inputs.get("truncated_ticket_row_count") or 0),
        "source_period": inputs.get("source_period"),
        "has_window_filter": "faq_window_days" in inputs,
        "faq_window_days": inputs.get("faq_window_days"),
        "faq_question_count": len(faq_questions),
        "faq_questions": faq_questions,
        "question_like_ticket_count": int(inputs.get("question_like_ticket_count") or 0),
        "top_ticket_clusters": list(inputs.get("top_ticket_clusters") or []),
        "customer_wording_example_count": len(customer_wording_examples),
        "customer_wording_examples": customer_wording_examples,
        "warning_count": len(package.warnings),
        "warnings": list(package.warnings),
        "metadata": dict(package.metadata),
    }


def _parse_outputs(value: str) -> tuple[str, ...]:
    outputs = tuple(item.strip() for item in value.split(",") if item.strip())
    if not outputs:
        raise argparse.ArgumentTypeError("--outputs must include at least one output")
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="CSV, JSON, or JSONL support-ticket export")
    parser.add_argument(
        "--format",
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
        help="Source file format. Defaults to extension-based auto detection.",
    )
    parser.add_argument(
        "--outputs",
        type=_parse_outputs,
        default=DEFAULT_SUPPORT_TICKET_OUTPUTS,
        help="Comma-separated Content Ops outputs to package for.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Date-window label to use when every included row has a parseable date.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=1000,
        help="Maximum source rows to include before reporting truncation.",
    )
    parser.add_argument(
        "--require-included-rows",
        action="store_true",
        help="Exit non-zero if no usable ticket wording survives packaging.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = build_support_ticket_package_smoke_summary(
            args.path,
            file_format=args.format,
            outputs=args.outputs,
            window_days=args.window_days,
            max_rows=args.max_rows,
            require_included_rows=args.require_included_rows,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"support-ticket package smoke failed: {exc}", file=sys.stderr)
        return 1
    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
