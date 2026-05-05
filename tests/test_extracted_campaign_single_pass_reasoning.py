from __future__ import annotations

import json
from typing import Any

import pytest

from extracted_content_pipeline.campaign_ports import (
    LLMMessage,
    LLMResponse,
    TenantScope,
)
from extracted_content_pipeline.services.single_pass_reasoning_provider import (
    DEFAULT_REASONING_SKILL_NAME,
    SinglePassCampaignReasoningProvider,
    SinglePassReasoningConfig,
    parse_reasoning_context_response,
)
from extracted_content_pipeline.skills.registry import get_skill_registry


class _LLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages,
        *,
        max_tokens,
        temperature,
        metadata=None,
    ):
        self.calls.append({
            "messages": tuple(messages),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "metadata": metadata,
        })
        return LLMResponse(content=self.content, model="test-model")


class _Skills:
    def __init__(self, prompts: dict[str, str]) -> None:
        self.prompts = prompts

    def get_prompt(self, name: str) -> str | None:
        return self.prompts.get(name)


def _response_payload() -> str:
    return json.dumps({
        "reasoning_context": {
            "wedge": "price_squeeze",
            "confidence": "medium",
            "summary": "Acme is feeling renewal pressure.",
            "why_now": "The contract window is close.",
            "recommended_action": "Lead with cost control.",
            "key_signals": ["pricing_mentions"],
        },
        "campaign_reasoning_context": {
            "proof_points": [
                {
                    "label": "pricing_mentions",
                    "value": 12,
                    "interpretation": "Pricing repeats in the source data.",
                }
            ],
            "timing_windows": [
                {
                    "window_type": "renewal",
                    "anchor": "Q3",
                    "urgency": "medium",
                }
            ],
            "coverage_limits": ["thin_account_signals"],
        },
        "reasoning_scope_summary": {
            "selection_strategy": "single_pass_opportunity",
            "reviews_considered_total": 12,
            "reviews_in_scope": 4,
        },
    })


@pytest.mark.asyncio
async def test_single_pass_reasoning_provider_builds_normalized_context() -> None:
    llm = _LLM(_response_payload())
    skills = _Skills({
        DEFAULT_REASONING_SKILL_NAME: (
            "target={target_id};mode={target_mode};scope={scope};"
            "opportunity={opportunity_json}"
        )
    })
    provider = SinglePassCampaignReasoningProvider(
        llm=llm,
        skills=skills,
        config=SinglePassReasoningConfig(max_tokens=500, temperature=0.1),
    )

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(
            account_id="acct_1",
            user_id="user_1",
            allowed_vendors=("LegacyCRM",),
        ),
        target_id="opp_1",
        target_mode="vendor_retention",
        opportunity={"company_name": "Acme", "vendor_name": "LegacyCRM"},
    )

    assert context is not None
    assert context.as_dict()["wedge"] == "price_squeeze"
    assert context.proof_points[0]["label"] == "pricing_mentions"
    assert context.timing_windows[0]["anchor"] == "Q3"
    assert context.coverage_limits == ("thin_account_signals",)
    assert context.scope_summary["reviews_in_scope"] == 4

    call = llm.calls[0]
    assert call["max_tokens"] == 500
    assert call["temperature"] == 0.1
    assert call["metadata"] == {
        "target_mode": "vendor_retention",
        "target_id": "opp_1",
        "skill_name": DEFAULT_REASONING_SKILL_NAME,
        "reasoning_provider": "single_pass",
    }
    messages = call["messages"]
    assert all(isinstance(message, LLMMessage) for message in messages)
    assert "{target_id}" not in messages[0].content
    assert "LegacyCRM" in messages[0].content
    assert "acct_1" in messages[1].content


def test_parse_reasoning_context_response_accepts_fenced_json() -> None:
    context = parse_reasoning_context_response(
        "```json\n{\"reasoning_context\":{\"summary\":\"ready\"}}\n```"
    )

    assert context is not None
    assert context.as_dict()["summary"] == "ready"


def test_parse_reasoning_context_response_accepts_embedded_json() -> None:
    context = parse_reasoning_context_response(
        "Here is the answer: {\"reasoning_context\":{\"confidence\":\"low\"}}"
    )

    assert context is not None
    assert context.as_dict()["confidence"] == "low"


def test_parse_reasoning_context_response_returns_none_for_empty_context() -> None:
    assert parse_reasoning_context_response("{}") is None
    assert parse_reasoning_context_response("not json") is None


@pytest.mark.asyncio
async def test_single_pass_reasoning_provider_requires_skill() -> None:
    provider = SinglePassCampaignReasoningProvider(
        llm=_LLM(_response_payload()),
        skills=_Skills({}),
    )

    with pytest.raises(ValueError, match="Campaign reasoning skill not found"):
        await provider.read_campaign_reasoning_context(
            scope=TenantScope(),
            target_id="opp_1",
            target_mode="vendor_retention",
            opportunity={},
        )


@pytest.mark.asyncio
async def test_single_pass_reasoning_provider_can_omit_source_opportunity() -> None:
    llm = _LLM(_response_payload())
    provider = SinglePassCampaignReasoningProvider(
        llm=llm,
        skills=_Skills({DEFAULT_REASONING_SKILL_NAME: "opportunity={opportunity}"}),
        config=SinglePassReasoningConfig(include_source_opportunity=False),
    )

    await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="opp_1",
        target_mode="vendor_retention",
        opportunity={"company_name": "Acme"},
    )

    assert "opportunity={}" in llm.calls[0]["messages"][0].content


def test_single_pass_reasoning_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="skill_name is required"):
        SinglePassReasoningConfig(skill_name="")

    with pytest.raises(ValueError, match="max_tokens must be positive"):
        SinglePassReasoningConfig(max_tokens=0)


def test_packaged_single_pass_reasoning_skill_is_loadable() -> None:
    prompt = get_skill_registry().get_prompt(DEFAULT_REASONING_SKILL_NAME)

    assert prompt is not None
    assert "Return one JSON object only" in prompt
