#!/usr/bin/env python3
"""Build campaign opportunities from richer source JSON/JSONL/CSV rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_source_adapters import (  # noqa: E402
    load_source_campaign_opportunities_from_file,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert review, transcript, complaint, or document source rows "
            "into AI Content Ops campaign opportunities."
        )
    )
    parser.add_argument("path", type=Path, help="Source JSON, JSONL, or CSV file.")
    parser.add_argument(
        "--format",
        choices=("auto", "json", "jsonl", "csv"),
        default="auto",
        help="Source file format. Defaults to suffix-based detection.",
    )
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--channel", default="email")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Maximum source text characters copied into each evidence row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write opportunity payload JSON to this path instead of stdout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.max_text_chars < 1:
        raise SystemExit("--max-text-chars must be positive")
    loaded = load_source_campaign_opportunities_from_file(
        args.path,
        file_format=args.format,
        target_mode=args.target_mode,
        max_text_chars=args.max_text_chars,
    )
    payload = loaded.as_payload(
        target_mode=args.target_mode,
        channel=args.channel,
        limit=args.limit,
    )
    output = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
