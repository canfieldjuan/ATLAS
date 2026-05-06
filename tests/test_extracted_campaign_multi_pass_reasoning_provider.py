from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from extracted_content_pipeline.campaign_ports import (
    CampaignReasoningContext,
    TenantScope,
)
from extracted_content_pipeline.services.multi_pass_reasoning_provider import (
    MultiPassCampaignReasoningProvider,
    MultiPassReasoningProviderConfig,
)
from extracted_reasoning_core.api import ConfigurationError
from extracted_reasoning_core.types import ReasoningPorts


class FakeLLMPort:
    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        self.calls.append({"max_tokens": max_tokens, "metadata": dict(metadata or {})})
        return self.responses.pop(0)


def _opportunity() -> dict[str, Any]:
    return {
        "vendor_id": "acme",
        "evidence": [
            {"source_type": "review", "source_id": "r1", "text": "Renewal pricing too high."},
            {"source_type": "ticket", "source_id": "t9", "text": "Onboarding pain."},
        ],
        "context": {"industry": "saas"},
    }


@pytest.mark.asyncio
async def test_multi_pass_provider_translates_run_reasoning_result_into_campaign_context() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Pricing pressure dominates the displacement signal.",
                "claims": [
                    {"claim": "Renewal pricing drives churn.", "confidence": "high", "source_ids": ["r1"]},
                    {"claim": "Onboarding friction is secondary.", "confidence": 0.55, "source_ids": ["t9"]},
                ],
                "confidence": 0.8,
            }),
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert isinstance(context, CampaignReasoningContext)
    # Top theses ordered by confidence (high → ~0.55).
    assert context.top_theses[0]["claim"] == "Renewal pricing drives churn."
    assert context.top_theses[0]["confidence"] == pytest.approx(0.9)
    assert context.top_theses[1]["claim"] == "Onboarding friction is secondary."
    # Reference ids aggregate unique source ids across claims.
    assert set(context.reference_ids["top_theses"]) == {"r1", "t9"}
    assert context.canonical_reasoning["summary"].startswith("Pricing pressure")
    assert context.canonical_reasoning["confidence"] == pytest.approx(0.8)
    assert context.canonical_reasoning["generation"] == 1
    assert context.scope_summary["entity_id"] == "acme"
    assert context.scope_summary["entity_type"] == "vendor"
    # LLM was called with the right reasoning_mode tag, proving the
    # reasoning-core surface is the one that actually ran.
    assert llm.calls[0]["metadata"]["reasoning_mode"] == "single_pass_synthesis"


@pytest.mark.asyncio
async def test_multi_pass_provider_returns_none_when_run_reasoning_fails_validation() -> None:
    # Two responses both missing claims → run_reasoning returns failed result.
    llm = FakeLLMPort([
        {"response": json.dumps({"summary": "no claims yet"}), "usage": {}},
        {"response": json.dumps({"summary": "still no claims"}), "usage": {}},
    ])
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts(llm=llm))

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is None


@pytest.mark.asyncio
async def test_multi_pass_provider_raises_configuration_error_without_llm_port() -> None:
    provider = MultiPassCampaignReasoningProvider(ports=ReasoningPorts())

    with pytest.raises(ConfigurationError, match="ReasoningPorts.llm"):
        await provider.read_campaign_reasoning_context(
            scope=TenantScope(),
            target_id="acme",
            target_mode="vendor",
            opportunity=_opportunity(),
        )


@pytest.mark.asyncio
async def test_multi_pass_provider_honors_top_thesis_limit() -> None:
    llm = FakeLLMPort([
        {
            "response": json.dumps({
                "summary": "Multiple drivers identified.",
                "claims": [
                    {"claim": f"Claim {i}", "confidence": 0.9 - 0.1 * i, "source_ids": [f"r{i}"]}
                    for i in range(5)
                ],
                "confidence": 0.7,
            }),
            "usage": {},
        }
    ])
    provider = MultiPassCampaignReasoningProvider(
        ports=ReasoningPorts(llm=llm),
        config=MultiPassReasoningProviderConfig(top_thesis_limit=2),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="acme",
        target_mode="vendor",
        opportunity=_opportunity(),
    )

    assert context is not None
    assert len(context.top_theses) == 2
    # Reference ids still aggregate from all claims (limit applies to top_theses only).
    assert len(context.reference_ids["top_theses"]) == 5
