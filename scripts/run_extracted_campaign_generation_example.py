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
    load_reasoning_provider_port,
)
from extracted_content_pipeline.services.single_pass_reasoning_provider import (  # noqa: E402
    SinglePassCampaignReasoningProvider,
    SinglePassReasoningConfig,
)


DEFAULT_PAYLOAD = (
    ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.json"
)
DEFAULT_REASONING_CONFIG = SinglePassReasoningConfig()


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
        "--single-pass-reasoning",
        action="store_true",
        help=(
            "Generate campaign reasoning context with the packaged single-pass "
            "reasoning prompt. Requires --llm pipeline."
        ),
    )
    parser.add_argument(
        "--reasoning-skill-name",
        default=DEFAULT_REASONING_CONFIG.skill_name,
        help="Skill name for --single-pass-reasoning.",
    )
    parser.add_argument(
        "--reasoning-max-tokens",
        type=int,
        default=DEFAULT_REASONING_CONFIG.max_tokens,
        help="Maximum LLM output tokens for --single-pass-reasoning.",
    )
    parser.add_argument(
        "--reasoning-temperature",
        type=float,
        default=DEFAULT_REASONING_CONFIG.temperature,
        help="LLM temperature for --single-pass-reasoning.",
    )
    parser.add_argument(
        "--no-reasoning-source-opportunity",
        dest="reasoning_include_source_opportunity",
        action="store_false",
        default=DEFAULT_REASONING_CONFIG.include_source_opportunity,
        help="Do not include the full source opportunity in the reasoning prompt.",
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


def _validate_reasoning_args(args: argparse.Namespace) -> None:
    if args.reasoning_context and args.single_pass_reasoning:
        raise SystemExit(
            "--reasoning-context and --single-pass-reasoning cannot be combined"
        )
    if args.single_pass_reasoning and args.llm != "pipeline":
        raise SystemExit("--single-pass-reasoning requires --llm pipeline")


def _single_pass_config_from_args(args: argparse.Namespace) -> SinglePassReasoningConfig:
    return SinglePassReasoningConfig(
        skill_name=str(args.reasoning_skill_name or ""),
        max_tokens=int(args.reasoning_max_tokens),
        temperature=float(args.reasoning_temperature),
        include_source_opportunity=bool(args.reasoning_include_source_opportunity),
    )


def _dependency_overrides(args: argparse.Namespace) -> dict[str, Any]:
    _validate_reasoning_args(args)
    overrides: dict[str, Any] = {}
    if args.reasoning_context:
        overrides["reasoning_context"] = load_reasoning_provider_port(
            args.reasoning_context
        )
    skills = None
    if args.skills_root:
        from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: PLC0415

        skills = get_skill_registry(root=args.skills_root)
        overrides["skills"] = skills
    if args.llm == "offline":
        return overrides

    from extracted_content_pipeline.campaign_llm_client import (  # noqa: PLC0415
        create_pipeline_llm_client,
    )
    from extracted_content_pipeline.skills.registry import get_skill_registry  # noqa: PLC0415

    llm = create_pipeline_llm_client()
    skills = skills or get_skill_registry()
    overrides["llm"] = llm
    overrides.setdefault("skills", skills)
    if args.single_pass_reasoning:
        overrides["reasoning_context"] = SinglePassCampaignReasoningProvider(
            llm=llm,
            skills=skills,
            config=_single_pass_config_from_args(args),
        )
    return overrides


async def _main() -> int:
    args = _parse_args()
    _validate_reasoning_args(args)
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
