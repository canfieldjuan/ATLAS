from __future__ import annotations

import pytest

from extracted_content_pipeline.ad_copy_generation import (
    AdCopyGenerationConfig,
    AdCopyGenerationService,
)
from extracted_content_pipeline.campaign_ports import TenantScope


@pytest.mark.asyncio
async def test_ad_copy_service_generates_evidence_backed_ads() -> None:
    service = AdCopyGenerationService()

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
    assert payload["ads"] == [
        {
            "id": "review-1",
            "channel": "paid_social",
            "format": "single_image",
            "headline": "HubSpot proof: pricing pressure",
            "primary_text": (
                "When HubSpot buyers mention pricing pressure, use the proof. "
                '"Pricing became hard to justify after renewal." Turn review '
                "signal into the next campaign."
            ),
            "cta": "See the proof",
            "source_id": "review-1",
            "source_type": "review",
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "HubSpot",
            "pain_points": ["pricing pressure"],
        }
    ]


@pytest.mark.asyncio
async def test_ad_copy_service_skips_unusable_rows_with_warnings() -> None:
    service = AdCopyGenerationService()

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
        "missing_ad_copy_evidence",
    ]


@pytest.mark.asyncio
async def test_ad_copy_service_keeps_fields_within_configured_bounds() -> None:
    config = AdCopyGenerationConfig(
        max_headline_chars=28,
        max_primary_text_chars=110,
    )
    service = AdCopyGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "HubSpot",
                "review_text": " ".join(["renewal pressure"] * 80),
                "pain_category": "pricing pressure from enterprise renewals",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    ad = payload["ads"][0]
    assert len(ad["headline"]) <= config.max_headline_chars
    assert ad["headline"].endswith("...")
    assert len(ad["primary_text"]) <= config.max_primary_text_chars
    assert ad["primary_text"].endswith("...")


@pytest.mark.asyncio
async def test_ad_copy_service_applies_limit_to_usable_rows() -> None:
    service = AdCopyGenerationService()

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
    assert [ad["source_id"] for ad in payload["ads"]] == ["review-1", "review-2"]
