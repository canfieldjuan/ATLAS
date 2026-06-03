from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.quote_card_generation import (
    QuoteCardGenerationConfig,
    QuoteCardGenerationService,
)


@pytest.mark.asyncio
async def test_quote_card_service_generates_evidence_backed_cards() -> None:
    service = QuoteCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme Logistics",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
                "pain_category": "pricing pressure",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["target_mode"] == "vendor_retention"
    assert payload["warnings"] == []
    assert payload["cards"] == [
        {
            "id": "review-1",
            "theme": "customer_proof",
            "quote": "Pricing became hard to justify after renewal.",
            "attribution": "Acme Logistics",
            "headline": "Customer proof for HubSpot",
            "supporting_text": "Use this quote to frame pricing pressure.",
            "source_id": "review-1",
            "source_type": "review",
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "HubSpot",
            "pain_points": ["pricing pressure"],
        }
    ]


@pytest.mark.asyncio
async def test_quote_card_service_skips_unusable_rows_with_warnings() -> None:
    service = QuoteCardGenerationService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=["not a row", {"review_id": "missing-text"}],
    )

    payload = result.as_dict()
    assert payload["generated"] == 0
    assert [warning["code"] for warning in payload["warnings"]] == [
        "row_not_object",
        "missing_source_text",
        "missing_quote_card_evidence",
    ]


@pytest.mark.asyncio
async def test_quote_card_service_expands_supported_source_material_bundles() -> None:
    service = QuoteCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "vendor": "Zendesk",
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "requester_company": "Acme Logistics",
                    "message": "Macros still require too much manual triage.",
                    "pain_category": "manual support triage",
                }
            ],
        },
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["warnings"] == []
    assert payload["cards"][0]["source_id"] == "ticket-1"
    assert payload["cards"][0]["source_type"] == "support_ticket"
    assert payload["cards"][0]["headline"] == "Customer proof for Zendesk"
    assert payload["cards"][0]["attribution"] == "Acme Logistics"


@pytest.mark.asyncio
async def test_quote_card_service_keeps_fields_within_configured_bounds() -> None:
    config = QuoteCardGenerationConfig(
        max_quote_chars=40,
        max_headline_chars=18,
        max_supporting_text_chars=34,
    )
    service = QuoteCardGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "reviewer_company": "Acme Logistics",
                "vendor": "VeryLongVendorName",
                "review_text": " ".join(["renewal pressure"] * 80),
                "pain_category": "pricing pressure from enterprise renewals",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    card = payload["cards"][0]
    assert len(card["quote"]) <= config.max_quote_chars
    assert card["quote"].endswith("...")
    assert len(card["headline"]) <= config.max_headline_chars
    assert card["headline"].endswith("...")
    assert len(card["supporting_text"]) <= config.max_supporting_text_chars
    assert card["supporting_text"].endswith("...")


@pytest.mark.asyncio
async def test_quote_card_service_applies_limit_to_usable_rows() -> None:
    service = QuoteCardGenerationService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=2,
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify.",
                "pain_category": "pricing pressure",
            },
            {
                "review_id": "review-2",
                "vendor": "Zendesk",
                "review_text": "Support queues took too long to triage.",
                "pain_category": "slow support",
            },
            {
                "review_id": "review-3",
                "vendor": "Intercom",
                "review_text": "Reporting did not explain where leads came from.",
                "pain_category": "unclear reporting",
            },
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 2
    assert [card["source_id"] for card in payload["cards"]] == ["review-1", "review-2"]


@pytest.mark.asyncio
async def test_quote_card_service_rejects_invalid_limits() -> None:
    service = QuoteCardGenerationService()

    with pytest.raises(ValueError, match="limit must be at least 1"):
        await service.generate(
            scope=TenantScope(),
            target_mode="vendor_retention",
            source_material=[],
            limit=0,
        )

    with pytest.raises(ValueError, match="max_text_chars must be at least 1"):
        await service.generate(
            scope=TenantScope(),
            target_mode="vendor_retention",
            source_material=[],
            max_text_chars=0,
        )
