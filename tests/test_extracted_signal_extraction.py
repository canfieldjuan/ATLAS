from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.signal_extraction import SignalExtractionService


@pytest.mark.asyncio
async def test_signal_extraction_service_normalizes_source_rows() -> None:
    service = SignalExtractionService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme",
                "vendor": "HubSpot",
                "review_text": "Pricing became hard to justify after renewal.",
                "pain_category": "pricing pressure",
                "contact_email": "buyer@example.com",
            }
        ],
        limit=5,
    )

    assert result.generated == 1
    assert result.warnings == ()
    assert result.opportunities[0]["target_id"] == "Acme"
    assert result.opportunities[0]["source_id"] == "review-1"
    assert result.opportunities[0]["target_mode"] == "vendor_retention"
    assert result.opportunities[0]["company_name"] == "Acme"
    assert result.opportunities[0]["pain_points"] == ["pricing pressure"]


@pytest.mark.asyncio
async def test_signal_extraction_service_accepts_wrapped_rows_and_limits() -> None:
    service = SignalExtractionService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material={
            "reviews": [
                {"id": "review-1", "vendor": "HubSpot", "text": "First"},
                {"id": "review-2", "vendor": "HubSpot", "text": "Second"},
            ]
        },
        limit=1,
    )

    assert result.generated == 1
    assert result.opportunities[0]["target_id"] == "review-1"


@pytest.mark.asyncio
async def test_signal_extraction_service_reports_missing_source_text_warning() -> None:
    service = SignalExtractionService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[{"id": "empty-1", "vendor": "HubSpot"}],
    )

    assert result.generated == 0
    assert [warning.code for warning in result.warnings] == ["missing_source_text"]
    assert result.as_dict()["warnings"][0]["code"] == "missing_source_text"


@pytest.mark.asyncio
async def test_signal_extraction_service_rejects_invalid_limits() -> None:
    service = SignalExtractionService()

    with pytest.raises(ValueError, match="limit must be at least 1"):
        await service.generate(
            scope=TenantScope(),
            target_mode="vendor_retention",
            source_material="pricing pressure",
            limit=0,
        )


@pytest.mark.asyncio
async def test_signal_extraction_service_stops_after_requested_valid_rows() -> None:
    service = SignalExtractionService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "id": "review-1",
                "company": "Acme",
                "vendor": "HubSpot",
                "text": "Pricing is a problem.",
                "contact_email": "buyer@example.com",
            },
            {"id": "empty-1", "vendor": "HubSpot"},
        ],
        limit=1,
    )

    assert result.generated == 1
    assert result.opportunities[0]["target_id"] == "review-1"
    assert result.warnings == ()
