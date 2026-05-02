#!/usr/bin/env python3
"""Run the extracted campaign generation product over a JSON payload."""

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

from extracted_content_pipeline.campaign_example import (  # noqa: E402
    generate_campaign_drafts_from_payload,
)


DEFAULT_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)


def _load_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("campaign generation payload must be a JSON object")
    return data


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline campaign drafts from customer opportunity JSON "
            "using the extracted content pipeline ports."
        )
    )
    parser.add_argument(
        "payload",
        nargs="?",
        type=Path,
        default=DEFAULT_PAYLOAD,
        help="Path to a campaign generation payload JSON file.",
    )
    parser.add_argument(
        "--target-mode",
        help="Override the payload target_mode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Override the payload limit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write generated draft JSON to this file instead of stdout.",
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    payload = _load_payload(args.payload)
    if args.target_mode:
        payload["target_mode"] = args.target_mode
    if args.limit is not None:
        payload["limit"] = args.limit

    result = await generate_campaign_drafts_from_payload(payload)
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
