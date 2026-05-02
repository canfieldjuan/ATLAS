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
from extracted_content_pipeline.campaign_customer_data import (  # noqa: E402
    load_campaign_opportunities_from_file,
)


DEFAULT_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)


def _load_payload(path: Path, *, file_format: str = "auto") -> dict[str, Any]:
    if file_format == "csv" or (file_format == "auto" and path.suffix.lower() == ".csv"):
        loaded = load_campaign_opportunities_from_file(path, file_format="csv")
        return loaded.as_payload()

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        loaded = load_campaign_opportunities_from_file(path, file_format="json")
        return loaded.as_payload()
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
        "--format",
        choices=("auto", "json", "csv"),
        default="auto",
        help="Customer data input format. Defaults to suffix-based auto detection.",
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
    parser.add_argument(
        "--llm",
        choices=("offline", "pipeline"),
        default="offline",
        help=(
            "Use the deterministic offline LLM or the product PipelineLLMClient "
            "configured through EXTRACTED_CAMPAIGN_LLM_* environment variables."
        ),
    )
    return parser.parse_args()


def _dependency_overrides(args: argparse.Namespace) -> dict[str, Any]:
    if args.llm == "offline":
        return {}

    from extracted_content_pipeline.campaign_llm_client import (  # noqa: PLC0415
        create_pipeline_llm_client,
    )
    from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: PLC0415

    return {
        "llm": create_pipeline_llm_client(),
        "skills": get_skill_registry(),
    }


async def _main() -> int:
    args = _parse_args()
    payload = _load_payload(args.payload, file_format=args.format)
    if args.target_mode:
        payload["target_mode"] = args.target_mode
    if args.limit is not None:
        payload["limit"] = args.limit

    result = await generate_campaign_drafts_from_payload(
        payload,
        **_dependency_overrides(args),
    )
    output = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
