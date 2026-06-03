from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.social_post_generation import (
    SocialPostGenerationConfig,
    SocialPostGenerationService,
)


@pytest.mark.asyncio
async def test_social_post_service_generates_evidence_backed_posts() -> None:
    service = SocialPostGenerationService()

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
    assert payload["posts"] == [
        {
            "id": "review-1",
            "channel": "linkedin",
            "text": (
                "Customer evidence for HubSpot flags pricing pressure. "
                "Source note: \"Pricing became hard to justify after renewal.\" "
                "Use this proof point to sharpen the next landing page, blog "
                "post, or sales brief."
            ),
            "source_id": "review-1",
            "source_type": "review",
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "HubSpot",
            "pain_points": ["pricing pressure"],
        }
    ]


@pytest.mark.asyncio
async def test_social_post_service_skips_unusable_rows_with_warnings() -> None:
    service = SocialPostGenerationService()

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
        "missing_social_post_evidence",
    ]


@pytest.mark.asyncio
async def test_social_post_service_keeps_truncated_posts_within_configured_bound() -> None:
    config = SocialPostGenerationConfig(max_post_chars=120)
    service = SocialPostGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "HubSpot",
                "review_text": " ".join(["renewal pressure"] * 80),
                "pain_category": "pricing pressure",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert len(payload["posts"][0]["text"]) <= config.max_post_chars
    assert payload["posts"][0]["text"].endswith("...")


@pytest.mark.asyncio
async def test_social_post_service_applies_limit_to_usable_rows() -> None:
    service = SocialPostGenerationService()

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
    assert [post["source_id"] for post in payload["posts"]] == ["review-1", "review-2"]
