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
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownService  # noqa: E402


_POSTGRES_FIXTURE_PAYLOAD: dict[str, Any] = {
    "reasoning_context": {
        "summary": "Postgres fixture reasoning context",
        "top_theses": [
            {
                "claim": "Renewal pricing",
                "summary": "Acme is reviewing pricing pressure before renewal.",
            }
        ],
        "proof_points": [{"label": "source_material", "value": "pricing"}],
    }
}


def _sample_consumed_reasoning_contexts(count: int) -> list[dict[str, Any]]:
    return [
        {
            "summary": "Offline smoke reasoning context",
            "proof_points": [{"label": "source_material", "value": "pricing"}],
        }
        for _ in range(max(0, count))
    ]


class _OfflinePostgresReasoningPool:
    """Tiny asyncpg-shaped pool for the host smoke's DB-provider mode."""

    async def fetchrow(
        self,
        _query: str,
        _account_id: str,
        selectors: list[str],
        target_mode: str = "",
    ) -> dict[str, Any] | None:
        if str(target_mode or "") not in {
            "",
            "vendor_retention",
            "marketing_campaign",
        }:
            return None
        wanted = {"opp_smoke", "content ops smoke", "acme"}
        if wanted.intersection({str(item).strip().lower() for item in selectors}):
            return {"payload": _POSTGRES_FIXTURE_PAYLOAD}
        return None


def _postgres_fixture_reasoning_provider() -> Any:
    from extracted_content_pipeline.campaign_reasoning_postgres import (
        PostgresCampaignReasoningContextRepository,
    )

    return PostgresCampaignReasoningContextRepository(
        pool=_OfflinePostgresReasoningPool()
    )


async def _consumed_contexts_from_provider(
    provider: Any | None,
    *,
    count: int,
    scope: Any,
    target_id: str,
    target_mode: str,
    opportunity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if provider is None or count <= 0:
        return []
    reader = getattr(provider, "read_campaign_reasoning_context", None)
    if not callable(reader):
        return _sample_consumed_reasoning_contexts(count)
    context = await reader(
        scope=scope,
        target_id=target_id,
        target_mode=target_mode,
        opportunity=opportunity,
    )
    if context is None:
        return []
    payload = context.as_dict() if hasattr(context, "as_dict") else dict(context)
    return [dict(payload) for _ in range(count)]


# PR-OptionA-1/2/3 graduated several plan-step fields to load-bearing kwargs
# (channels, default_report_type, default_brief_type, temperature,
# max_tokens, parse_retry_attempts, parse_retry_response_excerpt_chars,
# quality_revalidation_enabled, quality_prompt_proof_term_limit,
# quality_gates_enabled). The smoke fakes accept those via
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
        del scope
        result = {
            "generated": int(limit or 1),
            "asset": self.name,
            "target_mode": target_mode,
            "filters": dict(filters or {}),
            "saved_ids": [f"{self.name}-draft-1"],
        }
        if "quality_gates_enabled" in extras:
            result["quality_gates_enabled"] = extras["quality_gates_enabled"]
        if "quality_revalidation_enabled" in extras:
            result["quality_revalidation_enabled"] = extras[
                "quality_revalidation_enabled"
            ]
        return result


class _ReasoningAwareOpportunityAssetService(_OpportunityAssetService):
    def __init__(self, name: str, provider: Any | None = None) -> None:
        super().__init__(name)
        self.provider = provider

    def with_reasoning_context(
        self,
        provider: Any | None,
    ) -> "_ReasoningAwareOpportunityAssetService":
        return _ReasoningAwareOpportunityAssetService(self.name, provider=provider)

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        result = await super().generate(**kwargs)
        if self.provider is not None:
            count = int(result.get("generated") or 0)
            result["reasoning_contexts_used"] = count
            result["consumed_reasoning_contexts"] = await _consumed_contexts_from_provider(
                self.provider,
                count=count,
                scope=kwargs.get("scope"),
                target_id="opp_smoke",
                target_mode=str(kwargs.get("target_mode") or "vendor_retention"),
                opportunity={
                    "target_id": "opp_smoke",
                    "company_name": "Acme",
                    "vendor_name": "HubSpot",
                },
            )
        return result


class _LandingPageAssetService:
    async def generate(
        self,
        *,
        scope: Any,
        campaign: Any,
        **extras: Any,
    ) -> dict[str, Any]:
        del scope
        result = {
            "generated": 1,
            "asset": "landing_page",
            "campaign_name": getattr(campaign, "name", ""),
            "saved_ids": ["landing-page-draft-1"],
        }
        if "quality_gates_enabled" in extras:
            result["quality_gates_enabled"] = extras["quality_gates_enabled"]
        return result


class _ReasoningAwareLandingPageAssetService(_LandingPageAssetService):
    def __init__(self, provider: Any | None = None) -> None:
        self.provider = provider

    def with_reasoning_context(
        self,
        provider: Any | None,
    ) -> "_ReasoningAwareLandingPageAssetService":
        return _ReasoningAwareLandingPageAssetService(provider=provider)

    async def generate(self, **kwargs: Any) -> dict[str, Any]:
        result = await super().generate(**kwargs)
        if self.provider is not None:
            count = int(result.get("generated") or 0)
            result["reasoning_contexts_used"] = count
            campaign = kwargs.get("campaign")
            campaign_name = str(getattr(campaign, "name", "") or "Content Ops smoke")
            result["consumed_reasoning_contexts"] = await _consumed_contexts_from_provider(
                self.provider,
                count=count,
                scope=kwargs.get("scope"),
                target_id=campaign_name,
                target_mode="marketing_campaign",
                opportunity={"company_name": "Acme", "target_id": campaign_name},
            )
        return result


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
    parser.add_argument(
        "--target-mode",
        default="vendor_retention",
        help="Target mode to pass through the execution request.",
    )
    parser.add_argument(
        "--no-quality-gates",
        action="store_true",
        help="Set require_quality_gates=false in the execution request.",
    )
    parser.add_argument("--target-account", default="Acme")
    parser.add_argument("--offer", default="Churn intelligence audit")
    parser.add_argument("--topic", default="Churn pressure")
    parser.add_argument("--opportunity-id", default="opp_smoke")
    parser.add_argument("--audience", default="B2B SaaS founders")
    parser.add_argument("--campaign-name", default="Content Ops smoke")
    parser.add_argument(
        "--source-material",
        default="I cannot reset my password from the login page.",
    )
    parser.add_argument("--source-id", default="source-smoke-1")
    parser.add_argument("--source-type", default="support_ticket")
    parser.add_argument("--source-title", default="pricing after renewal")
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
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help=(
            "Attach an offline host reasoning provider to reasoning-aware "
            "generated-asset services."
        ),
    )
    parser.add_argument(
        "--reasoning-provider",
        choices=("sample", "postgres-fixture"),
        default="sample",
        help=(
            "Offline provider fixture used with --with-reasoning. "
            "'postgres-fixture' exercises the real Postgres reasoning adapter "
            "against an in-memory asyncpg-shaped pool."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "preset": str(args.preset or "").strip() or "full_campaign",
        "target_mode": str(args.target_mode or "").strip() or "vendor_retention",
        "require_quality_gates": not bool(args.no_quality_gates),
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
                    "source_type": args.source_type,
                    "source_title": args.source_title,
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


def _services(
    *,
    reasoning: bool = False,
    reasoning_provider: str = "sample",
) -> ContentOpsExecutionServices:
    opportunity_service = (
        _ReasoningAwareOpportunityAssetService
        if reasoning
        else _OpportunityAssetService
    )
    landing_page: Any = (
        _ReasoningAwareLandingPageAssetService()
        if reasoning
        else _LandingPageAssetService()
    )
    services = ContentOpsExecutionServices(
        campaign=opportunity_service("email_campaign"),
        blog_post=opportunity_service("blog_post"),
        report=opportunity_service("report"),
        landing_page=landing_page,
        sales_brief=opportunity_service("sales_brief"),
        signal_extraction=SignalExtractionService(),
        faq_markdown=TicketFAQMarkdownService(),
    )
    if reasoning:
        provider = (
            _postgres_fixture_reasoning_provider()
            if reasoning_provider == "postgres-fixture"
            else object()
        )
        services = services.with_reasoning_context(provider)
    return services


def _execution_errors(
    result: Mapping[str, Any],
    *,
    require_reasoning_usage: bool = False,
) -> list[str]:
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
        if require_reasoning_usage and step.get("output") != "signal_extraction":
            errors.extend(_reasoning_usage_errors(index, step, result_payload))
    return errors


def _step_has_output_payload(
    step: Mapping[str, Any],
    result_payload: Any,
) -> bool:
    if not isinstance(result_payload, Mapping):
        return False
    if step.get("output") == "signal_extraction":
        return bool(result_payload.get("opportunities"))
    if step.get("output") == "faq_markdown":
        checks = result_payload.get("output_checks")
        return (
            bool(result_payload.get("markdown"))
            and bool(result_payload.get("items"))
            and isinstance(checks, Mapping)
            and bool(checks)
            and all(value is True for value in checks.values())
        )
    return bool(result_payload.get("saved_ids"))


def _reasoning_usage_errors(
    index: int,
    step: Mapping[str, Any],
    result_payload: Any,
) -> list[str]:
    errors: list[str] = []
    result_count = (
        result_payload.get("reasoning_contexts_used")
        if isinstance(result_payload, Mapping)
        else None
    )
    reasoning = step.get("reasoning")
    audit_count = (
        reasoning.get("contexts_used")
        if isinstance(reasoning, Mapping)
        else None
    )
    if not isinstance(result_count, int) or isinstance(result_count, bool):
        errors.append(f"step {index} missing result.reasoning_contexts_used")
    if not isinstance(audit_count, int) or isinstance(audit_count, bool):
        errors.append(f"step {index} missing reasoning.contexts_used")
    if isinstance(result_count, int) and isinstance(audit_count, int):
        if not isinstance(result_count, bool) and not isinstance(audit_count, bool):
            if result_count != audit_count:
                errors.append(
                    f"step {index} reasoning usage mismatch: "
                    f"result={result_count} audit={audit_count}"
                )
    if (
        isinstance(reasoning, Mapping)
        and isinstance(result_count, int)
        and not isinstance(result_count, bool)
        and result_count > 0
    ):
        consumed = reasoning.get("consumed_contexts")
        if not isinstance(consumed, list) or len(consumed) != result_count:
            errors.append(f"step {index} missing reasoning.consumed_contexts")
    return errors


async def _main() -> int:
    args = _parse_args()
    result = await execute_content_ops_from_mapping(
        _payload(args),
        services=_services(
            reasoning=bool(args.with_reasoning),
            reasoning_provider=str(args.reasoning_provider or "sample"),
        ),
    )
    errors = _execution_errors(
        result,
        require_reasoning_usage=bool(args.with_reasoning),
    )
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
