#!/usr/bin/env python3
"""Run a host-facing offline smoke test for the extracted content pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.campaign_customer_data import (  # noqa: E402
    load_campaign_opportunities_from_file,
)
from extracted_content_pipeline.campaign_example import (  # noqa: E402
    generate_campaign_drafts_from_payload,
)


DEFAULT_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)


def _load_payload(path: Path, *, file_format: str) -> dict[str, Any]:
    if file_format == "csv" or (file_format == "auto" and path.suffix.lower() == ".csv"):
        loaded = load_campaign_opportunities_from_file(path, file_format="csv")
        return loaded.as_payload()

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    loaded = load_campaign_opportunities_from_file(path, file_format="json")
    return loaded.as_payload()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test a standalone AI Content Ops install by generating "
            "offline campaign drafts from customer opportunity data."
        )
    )
    parser.add_argument(
        "payload",
        nargs="?",
        type=Path,
        default=DEFAULT_PAYLOAD,
        help="Customer opportunity JSON or CSV file. Defaults to the packaged example.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "csv"),
        default="auto",
        help="Input format. Defaults to suffix-based auto detection.",
    )
    parser.add_argument(
        "--channels",
        default="email_cold,email_followup",
        help="Comma-separated channels to generate for each opportunity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum opportunities to smoke-test.",
    )
    parser.add_argument(
        "--min-drafts",
        type=int,
        help=(
            "Minimum generated drafts required for a passing smoke test. "
            "Defaults to limit multiplied by selected channel count."
        ),
    )
    return parser.parse_args(argv)


def _draft_errors(result: dict[str, Any], *, min_drafts: int) -> list[str]:
    drafts = result.get("drafts")
    if not isinstance(drafts, list):
        return ["result.drafts is missing or not a list"]
    if len(drafts) < min_drafts:
        return [f"expected at least {min_drafts} draft(s), got {len(drafts)}"]

    errors: list[str] = []
    for index, draft in enumerate(drafts[:min_drafts], start=1):
        if not isinstance(draft, dict):
            errors.append(f"draft {index} is not an object")
            continue
        for field in ("subject", "body", "target_id", "channel"):
            if not str(draft.get(field) or "").strip():
                errors.append(f"draft {index} missing {field}")
    return errors


async def _main() -> int:
    args = _parse_args()
    payload = _load_payload(args.payload, file_format=args.format)
    payload["limit"] = int(args.limit)
    payload["channels"] = [
        item.strip()
        for item in str(args.channels or "").split(",")
        if item.strip()
    ]
    min_drafts = (
        int(args.min_drafts)
        if args.min_drafts is not None
        else int(args.limit) * len(payload["channels"])
    )

    result = await generate_campaign_drafts_from_payload(payload)
    errors = _draft_errors(result, min_drafts=min_drafts)
    if errors:
        print("AI Content Ops host smoke failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    summary = result.get("result") if isinstance(result.get("result"), dict) else {}
    drafts = result.get("drafts") if isinstance(result.get("drafts"), list) else []
    first = drafts[0] if drafts and isinstance(drafts[0], dict) else {}
    print(
        "AI Content Ops host smoke passed: "
        f"generated={summary.get('generated', len(drafts))} "
        f"model={result.get('llm_model', 'unknown')} "
        f"first_subject={first.get('subject', '')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
