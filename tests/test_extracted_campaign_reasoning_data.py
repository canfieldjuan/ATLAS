from __future__ import annotations

import json

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.campaign_reasoning_data import (
    FileCampaignReasoningContextProvider,
    load_campaign_reasoning_context_provider,
)


@pytest.mark.asyncio
async def test_file_reasoning_provider_matches_target_id() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "contexts": [
            {
                "target_id": "opp-1",
                "reasoning_context": {
                    "wedge": "renewal pressure",
                    "confidence": "high",
                },
                "campaign_reasoning_context": {
                    "proof_points": [{"label": "pricing_mentions", "value": 12}]
                },
            }
        ]
    })

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(account_id="acct-1"),
        target_id="opp-1",
        target_mode="vendor_retention",
        opportunity={"company_name": "Acme"},
    )

    assert context is not None
    assert context.as_dict()["wedge"] == "renewal pressure"
    assert context.proof_points[0]["label"] == "pricing_mentions"


@pytest.mark.asyncio
async def test_file_reasoning_provider_accepts_mapping_index() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "opp-1": {
            "reasoning_context": {
                "summary": "Acme is reviewing vendors before renewal."
            }
        }
    })

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="opp-1",
        target_mode="vendor_retention",
        opportunity={},
    )

    assert context is not None
    assert context.as_dict()["summary"] == "Acme is reviewing vendors before renewal."


@pytest.mark.asyncio
async def test_file_reasoning_provider_matches_company_fallback_case_insensitive() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "contexts": [
            {
                "company_name": "Acme Logistics",
                "context": {
                    "campaign_reasoning_context": {
                        "account_signals": [
                            {"company": "Acme Logistics", "primary_pain": "pricing"}
                        ]
                    }
                },
            }
        ]
    })

    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="missing-id",
        target_mode="vendor_retention",
        opportunity={"company_name": "acme logistics"},
    )

    assert context is not None
    assert context.account_signals[0]["primary_pain"] == "pricing"


@pytest.mark.asyncio
async def test_file_reasoning_provider_returns_none_for_missing_or_empty_context() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "contexts": [
            {"target_id": "empty", "reasoning_context": {}},
            {"target_id": "other", "reasoning_context": {"wedge": "pricing"}},
        ]
    })

    assert await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="empty",
        target_mode="vendor_retention",
        opportunity={},
    ) is None
    assert await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="missing",
        target_mode="vendor_retention",
        opportunity={},
    ) is None


@pytest.mark.asyncio
async def test_file_reasoning_provider_does_not_match_on_target_mode_only() -> None:
    provider = FileCampaignReasoningContextProvider.from_payload({
        "contexts": [
            {
                "target_mode": "vendor_retention",
                "reasoning_context": {"wedge": "pricing"},
            }
        ]
    })

    assert await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="opp-1",
        target_mode="vendor_retention",
        opportunity={},
    ) is None


@pytest.mark.asyncio
async def test_load_file_reasoning_provider(tmp_path) -> None:
    path = tmp_path / "reasoning.json"
    path.write_text(
        json.dumps({
            "contexts": [
                {
                    "target_id": "opp-1",
                    "reasoning_context": {"confidence": "medium"},
                }
            ]
        }),
        encoding="utf-8",
    )

    provider = load_campaign_reasoning_context_provider(path)
    context = await provider.read_campaign_reasoning_context(
        scope=TenantScope(),
        target_id="opp-1",
        target_mode="vendor_retention",
        opportunity={},
    )

    assert provider.source == str(path)
    assert context is not None
    assert context.as_dict()["confidence"] == "medium"


def test_load_reasoning_provider_port_is_protocol_compatible(tmp_path) -> None:
    path = tmp_path / "reasoning.json"
    path.write_text('[]', encoding="utf-8")

    provider = load_reasoning_provider_port(path)

    assert isinstance(provider, FileCampaignReasoningContextProvider)
