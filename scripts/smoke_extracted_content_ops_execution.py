#!/usr/bin/env python3
"""Run an offline smoke test for the AI Content Ops execution seam."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extracted_content_pipeline.content_ops_execution import (  # noqa: E402
    ContentOpsExecutionServices,
    execute_content_ops_from_mapping,
)
from extracted_content_pipeline.signal_extraction import SignalExtractionService  # noqa: E402


# PR-OptionA-1/2/3 graduated several plan-step fields to load-bearing kwargs
# (channels, default_report_type, default_brief_type, temperature,
# max_tokens, parse_retry_attempts, parse_retry_response_excerpt_chars,
# quality_revalidation_enabled, quality_prompt_proof_term_limit,
# quality_gates_enabled). The smoke fakes accept (and ignore) those via
# ``**extras`` -- this CLI exercises the seam end-to-end, not the kwargs
# contract. Strict per-kwarg assertions live in the dispatcher tests in
# ``tests/test_extracted_content_ops_execution.py``.
class _OpportunityAssetService:
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(
        self,
        *,
        scope: Any,
        target_mode: str,
        limit: int | None = None,
        filters: Mapping[str, Any] | None = None,
        **extras: Any,
    ) -> dict[str, Any]:
        del scope, extras
        return {
            "generated": int(limit or 1),
            "asset": self.name,
            "target_mode": target_mode,
            "filters": dict(filters or {}),
            "saved_ids": [f"{self.name}-draft-1"],
        }


class _LandingPageAssetService:
    async def generate(
        self,
        *,
        scope: Any,
        campaign: Any,
        **extras: Any,
    ) -> dict[str, Any]:
        del scope, extras
        return {
            "generated": 1,
            "asset": "landing_page",
            "campaign_name": getattr(campaign, "name", ""),
            "saved_ids": ["landing-page-draft-1"],
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the AI Content Ops execution seam with host-injected "
            "offline services. No database, network, sender, or LLM is used."
        )
    )
    parser.add_argument("--preset", default="full_campaign")
    parser.add_argument(
        "--outputs",
        help="Comma-separated output ids. Overrides --preset when supplied.",
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--target-mode", default="vendor_retention")
    parser.add_argument("--target-account", default="Acme")
    parser.add_argument("--offer", default="Churn intelligence audit")
    parser.add_argument("--topic", default="Churn pressure")
    parser.add_argument("--opportunity-id", default="opp_smoke")
    parser.add_argument("--audience", default="B2B SaaS founders")
    parser.add_argument("--campaign-name", default="Content Ops smoke")
    parser.add_argument(
        "--source-material",
        default="Pricing became hard to justify after renewal.",
    )
    parser.add_argument("--source-id", default="source-smoke-1")
    parser.add_argument("--source-vendor", default="HubSpot")
    parser.add_argument("--source-contact-email", default="buyer@example.com")
    parser.add_argument(
        "--source-max-text-chars",
        type=int,
        help=(
            "Cap evidence text at this many characters before signal extraction. "
            "Omit to use the service default (1200)."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "preset": str(args.preset or "").strip() or "full_campaign",
        "target_mode": str(args.target_mode or "").strip() or "vendor_retention",
        "limit": max(1, int(args.limit or 1)),
        "inputs": {
            "target_account": args.target_account,
            "offer": args.offer,
            "topic": args.topic,
            "opportunity_id": args.opportunity_id,
            "audience": args.audience,
            "campaign_name": args.campaign_name,
            "source_material": [
                {
                    "id": args.source_id,
                    "company": args.target_account,
                    "vendor": args.source_vendor,
                    "text": args.source_material,
                    "contact_email": args.source_contact_email,
                }
            ],
        },
    }
    if args.source_max_text_chars is not None:
        payload["inputs"]["source_max_text_chars"] = args.source_max_text_chars
    if args.outputs:
        payload["outputs"] = [
            item.strip()
            for item in str(args.outputs).split(",")
            if item.strip()
        ]
    return payload


def _services() -> ContentOpsExecutionServices:
    return ContentOpsExecutionServices(
        campaign=_OpportunityAssetService("email_campaign"),
        blog_post=_OpportunityAssetService("blog_post"),
        report=_OpportunityAssetService("report"),
        landing_page=_LandingPageAssetService(),
        sales_brief=_OpportunityAssetService("sales_brief"),
        signal_extraction=SignalExtractionService(),
    )


def _execution_errors(result: Mapping[str, Any]) -> list[str]:
    if result.get("status") != "completed":
        return [f"expected completed status, got {result.get('status')!r}"]
    steps = result.get("steps")
    if not isinstance(steps, list) or not steps:
        return ["result.steps is missing or empty"]
    errors: list[str] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, Mapping):
            errors.append(f"step {index} is not an object")
            continue
        if step.get("status") != "completed":
            errors.append(f"step {index} did not complete: {step.get('error')}")
        result_payload = step.get("result")
        if not _step_has_output_payload(step, result_payload):
            errors.append(f"step {index} missing output payload")
    return errors


def _step_has_output_payload(
    step: Mapping[str, Any],
    result_payload: Any,
) -> bool:
    if not isinstance(result_payload, Mapping):
        return False
    if step.get("output") == "signal_extraction":
        return bool(result_payload.get("opportunities"))
    return bool(result_payload.get("saved_ids"))


async def _main() -> int:
    args = _parse_args()
    result = await execute_content_ops_from_mapping(
        _payload(args),
        services=_services(),
    )
    errors = _execution_errors(result)
    if args.json:
        if errors:
            result = dict(result)
            result["smoke_errors"] = errors
        print(json.dumps(result, sort_keys=True))
    if errors:
        print("AI Content Ops execution smoke failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if not args.json:
        outputs = [
            step.get("output", "")
            for step in result.get("steps", [])
            if isinstance(step, Mapping)
        ]
        print(
            "AI Content Ops execution smoke passed: "
            f"status={result.get('status')} outputs={','.join(outputs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
