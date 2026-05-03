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
from extracted_content_pipeline.campaign_reasoning_data import (  # noqa: E402
    load_campaign_reasoning_context_provider,
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--channels",
        help=(
            "Comma-separated draft channels to generate per opportunity, "
            "for example email_cold,email_followup."
        ),
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
        "--reasoning-context",
        type=Path,
        help=(
            "Optional JSON file containing host-provided reasoning context "
            "keyed by target id, company, email, or vendor."
        ),
    )
    parser.add_argument(
        "--skills-root",
        type=Path,
        help=(
            "Optional directory of host-provided markdown skill prompts. "
            "Custom prompts override packaged prompts with the same name."
        ),
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
    return parser.parse_args(argv)


def _dependency_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.reasoning_context:
        overrides["reasoning_context"] = load_campaign_reasoning_context_provider(
            args.reasoning_context
        )
    if args.skills_root:
        from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: PLC0415

        overrides["skills"] = get_skill_registry(root=args.skills_root)
    if args.llm == "offline":
        return overrides

    from extracted_content_pipeline.campaign_llm_client import (  # noqa: PLC0415
        create_pipeline_llm_client,
    )
    from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: PLC0415

    overrides["llm"] = create_pipeline_llm_client()
    overrides.setdefault("skills", get_skill_registry())
    return overrides


async def _main() -> int:
    args = _parse_args()
    payload = _load_payload(args.payload, file_format=args.format)
    if args.target_mode:
        payload["target_mode"] = args.target_mode
    if args.channels:
        payload["channels"] = [
            item.strip()
            for item in args.channels.split(",")
            if item.strip()
        ]
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
