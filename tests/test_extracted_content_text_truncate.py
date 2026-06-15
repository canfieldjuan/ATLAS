from __future__ import annotations

import pytest

from extracted_content_pipeline.ad_copy_generation import (
    AdCopyGenerationConfig,
    AdCopyGenerationService,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.social_post_generation import (
    SocialPostGenerationConfig,
    SocialPostGenerationService,
)
from extracted_content_pipeline.stat_card_generation import (
    StatCardGenerationConfig,
    StatCardGenerationService,
)
from extracted_content_pipeline.text_truncate import truncate_with_ellipsis


def test_truncate_with_ellipsis_reserves_suffix_length() -> None:
    output = truncate_with_ellipsis("alpha beta gamma delta", 16)

    assert output == "alpha beta ga..."
    assert len(output) <= 16
    assert output.endswith("...")


def test_truncate_with_ellipsis_compacts_whitespace_by_default() -> None:
    output = truncate_with_ellipsis("alpha   beta\n\ngamma", 15)

    assert output == "alpha beta g..."
    assert len(output) <= 15


def test_truncate_with_ellipsis_supports_custom_suffix() -> None:
    output = truncate_with_ellipsis("abcdefghij", 7, suffix=" [x]")

    assert output == "abc [x]"
    assert len(output) <= 7
    assert output.endswith(" [x]")


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_truncate_with_ellipsis_keeps_tiny_limits_bounded(limit: int) -> None:
    output = truncate_with_ellipsis("abcdef", limit)

    assert output == "." * limit
    assert len(output) <= limit


def test_truncate_with_ellipsis_can_preserve_small_limit_text_contract() -> None:
    output = truncate_with_ellipsis(
        "abcdef", 2, compact_whitespace=False, small_limit="text"
    )

    assert output == "ab"
    assert len(output) <= 2


def test_truncate_with_ellipsis_treats_non_integer_limit_as_empty() -> None:
    assert truncate_with_ellipsis("abcdef", None) == ""
    assert truncate_with_ellipsis("abcdef", "not-a-limit") == ""
    assert truncate_with_ellipsis("abcdef", True) == ""


@pytest.mark.asyncio
async def test_social_post_generation_uses_bounded_shared_truncation() -> None:
    config = SocialPostGenerationConfig(max_post_chars=18)
    service = SocialPostGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "HubSpot",
                "review_text": " ".join(["renewal pressure"] * 20),
                "pain_category": "pricing pressure",
            }
        ],
    )

    post = result.as_dict()["posts"][0]
    assert post["text"] == "Customer eviden..."
    assert post["text"].endswith("...")
    assert len(post["text"]) <= config.max_post_chars


@pytest.mark.asyncio
async def test_ad_copy_generation_uses_bounded_shared_truncation() -> None:
    config = AdCopyGenerationConfig(max_headline_chars=16, max_primary_text_chars=22)
    service = AdCopyGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "HubSpot",
                "review_text": " ".join(["renewal pressure"] * 20),
                "pain_category": "pricing pressure",
            }
        ],
    )

    ad = result.as_dict()["ads"][0]
    assert ad["headline"] == "HubSpot proof..."
    assert len(ad["headline"]) <= config.max_headline_chars
    assert ad["primary_text"].endswith("...")
    assert len(ad["primary_text"]) <= config.max_primary_text_chars


@pytest.mark.asyncio
async def test_stat_card_generation_preserves_existing_small_limit_text_contract() -> None:
    config = StatCardGenerationConfig(
        max_claim_chars=2,
        max_headline_chars=2,
        max_supporting_text_chars=2,
        max_evidence_chars=2,
    )
    service = StatCardGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "VeryLongVendorName",
                "review_text": "NPS score dropped to 42 after renewal pressure.",
                "nps_score": 42,
                "pain_category": "pricing pressure",
            }
        ],
    )

    stat = result.as_dict()["stats"][0]
    assert stat["claim"] == "NP"
    assert stat["headline"] == "Cu"
    assert stat["supporting_text"] == "Us"
    assert stat["evidence"] == "42"


def test_truncate_no_compaction_retains_internal_whitespace() -> None:
    value = "NPS   score   dropped here"

    kept = truncate_with_ellipsis(
        value, 18, compact_whitespace=False, small_limit="text"
    )
    collapsed = truncate_with_ellipsis(
        value, 18, compact_whitespace=True, small_limit="text"
    )

    assert "  " in kept
    assert "  " not in collapsed
    assert kept != collapsed


def test_truncate_rstrips_boundary_space_before_suffix() -> None:
    assert truncate_with_ellipsis("alpha beta gamma", 14) == "alpha beta..."
