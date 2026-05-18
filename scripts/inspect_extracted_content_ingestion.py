#!/usr/bin/env python3
"""Inspect AI Content Ops ingestion readiness without writing data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.ingestion_diagnostics import inspect_ingestion_file  # noqa: E402
from extracted_content_pipeline.campaign_source_adapters import parse_default_fields_or_exit  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect campaign opportunity or source-row ingestion readiness."
    )
    parser.add_argument("path", type=Path, help="Opportunity or source-row JSON/JSONL/CSV file.")
    parser.add_argument(
        "--format",
        choices=("auto", "json", "csv"),
        default="auto",
        help="Opportunity file format when --source-rows is not selected.",
    )
    parser.add_argument(
        "--source-rows",
        action="store_true",
        help="Treat input as source rows and convert them to opportunities first.",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "json", "jsonl", "csv"),
        default="auto",
        help="Source-row file format when --source-rows is selected.",
    )
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument(
        "--max-source-text-chars",
        type=int,
        default=1200,
        help="Maximum source text characters copied into each evidence row.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=3,
        help="Maximum normalized opportunity samples included in the report.",
    )
    parser.add_argument(
        "--default-field",
        action="append",
        default=[],
        help=(
            "Fallback metadata applied to every source row when --source-rows "
            "is selected. Repeat as key=value."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit full JSON report.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.sample_limit < 0:
        raise SystemExit("--sample-limit must be non-negative")
    if args.max_source_text_chars < 1:
        raise SystemExit("--max-source-text-chars must be positive")

    report = inspect_ingestion_file(
        args.path,
        source_rows=bool(args.source_rows),
        file_format=args.format,
        source_format=args.source_format,
        target_mode=args.target_mode,
        max_source_text_chars=args.max_source_text_chars,
        sample_limit=args.sample_limit,
        default_fields=parse_default_fields_or_exit(args.default_field),
    )
    payload = report.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "ok={ok} mode={mode} opportunities={opportunity_count} "
            "warnings={warning_count} missing={missing}".format(
                ok=str(payload["ok"]).lower(),
                mode=payload["mode"],
                opportunity_count=payload["opportunity_count"],
                warning_count=payload["warning_count"],
                missing=payload["missing_field_counts"],
            )
        )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
