from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.stat_card_generation import (
    StatCardGenerationConfig,
    StatCardGenerationService,
)


@pytest.mark.asyncio
async def test_stat_card_service_generates_evidence_backed_stats() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "reviewer_company": "Acme Logistics",
                "vendor": "Zendesk",
                "review_text": "NPS score dropped to 42 after renewal.",
                "nps_score": 42,
                "pain_category": "renewal dissatisfaction",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["target_mode"] == "vendor_retention"
    assert payload["warnings"] == []
    assert payload["stats"] == [
        {
            "id": "review-1",
            "theme": "customer_metric",
            "metric_label": "NPS score",
            "metric_value": 42,
            "metric_display": "42",
            "claim": "NPS score: 42",
            "headline": "Customer metric for Zendesk",
            "supporting_text": "Use this stat to frame renewal dissatisfaction.",
            "evidence": "NPS score dropped to 42 after renewal.",
            "source_id": "review-1",
            "source_type": "review",
            "target_id": "review-1",
            "company_name": "Acme Logistics",
            "vendor_name": "Zendesk",
            "pain_points": ["renewal dissatisfaction"],
        }
    ]


@pytest.mark.asyncio
async def test_stat_card_service_requires_numeric_value_in_evidence() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "Zendesk",
                "review_text": "NPS moved from 40 to 41 after renewal.",
                "nps_score": 72,
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 0
    assert payload["warnings"] == [
        {
            "code": "unsupported_numeric_claim",
            "message": (
                "Skipped metric because the numeric value is not present in "
                "source evidence."
            ),
            "row_index": 1,
            "field": "nps_score",
        }
    ]


@pytest.mark.asyncio
async def test_stat_card_service_preserves_metric_value_in_bounded_evidence() -> None:
    service = StatCardGenerationService(
        StatCardGenerationConfig(max_evidence_chars=36)
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "Zendesk",
                "review_text": (
                    "Long setup without numbers. " * 12
                    + "NPS score finally settled at 42 after renewal."
                ),
                "nps_score": 42,
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    evidence = payload["stats"][0]["evidence"]
    assert len(evidence) <= 36
    assert evidence.startswith("...")
    assert "42" in evidence


@pytest.mark.asyncio
async def test_stat_card_service_rejects_non_numeric_metric_values() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-1",
                "vendor": "Zendesk",
                "review_text": "NPS score was high after onboarding.",
                "nps_score": "high",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 0
    assert payload["warnings"] == [
        {
            "code": "invalid_stat_card_metric",
            "message": "Skipped metric because its value is not numeric.",
            "row_index": 1,
            "field": "nps_score",
        }
    ]


@pytest.mark.asyncio
async def test_stat_card_service_skips_rows_without_supported_metrics() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            "not a row",
            {
                "review_id": "review-1",
                "vendor": "Zendesk",
                "review_text": "Pricing became hard to justify after renewal.",
            },
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 0
    assert [warning["code"] for warning in payload["warnings"]] == [
        "row_not_object",
        "missing_stat_card_metric",
    ]


@pytest.mark.asyncio
async def test_stat_card_service_expands_supported_source_material_bundles() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "vendor": "Zendesk",
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "requester_company": "Acme Logistics",
                    "message": "CSAT score reached 88 after the workflow fix.",
                    "csat_score": "88",
                    "pain_category": "workflow friction",
                }
            ],
        },
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["warnings"] == []
    stat = payload["stats"][0]
    assert stat["source_id"] == "ticket-1"
    assert stat["source_type"] == "support_ticket"
    assert stat["metric_label"] == "CSAT score"
    assert stat["metric_value"] == 88
    assert stat["headline"] == "Customer metric for Zendesk"
    assert stat["company_name"] == "Acme Logistics"


@pytest.mark.asyncio
async def test_stat_card_service_supports_percent_numbers_in_evidence() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "source_id": "win-rate-1",
                "source_type": "crm_note",
                "company_name": "Acme Logistics",
                "vendor_name": "Zendesk",
                "notes": "Win rate improved to 42% after switching scripts.",
                "win_rate": "42",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    assert payload["stats"][0]["metric_label"] == "Win Rate"
    assert payload["stats"][0]["metric_display"] == "42"


@pytest.mark.asyncio
async def test_stat_card_service_keeps_fields_within_configured_bounds() -> None:
    config = StatCardGenerationConfig(
        max_claim_chars=10,
        max_headline_chars=22,
        max_supporting_text_chars=34,
        max_evidence_chars=36,
    )
    service = StatCardGenerationService(config=config)

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        source_material=[
            {
                "review_id": "review-long",
                "vendor": "VeryLongVendorName",
                "review_text": "NPS score dropped to 42 after " + "renewal pressure " * 20,
                "nps_score": 42,
                "pain_category": "pricing pressure from enterprise renewals",
            }
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 1
    stat = payload["stats"][0]
    assert len(stat["claim"]) <= config.max_claim_chars
    assert stat["claim"].endswith("...")
    assert len(stat["headline"]) <= config.max_headline_chars
    assert stat["headline"].endswith("...")
    assert len(stat["supporting_text"]) <= config.max_supporting_text_chars
    assert stat["supporting_text"].endswith("...")
    assert len(stat["evidence"]) <= config.max_evidence_chars
    assert stat["evidence"].endswith("...")


@pytest.mark.asyncio
async def test_stat_card_service_applies_limit_to_usable_rows() -> None:
    service = StatCardGenerationService()

    result = await service.generate(
        scope=TenantScope(),
        target_mode="vendor_retention",
        limit=2,
        source_material=[
            {"review_id": "review-1", "review_text": "NPS score is 42.", "nps_score": 42},
            {"review_id": "review-2", "review_text": "CSAT score is 88.", "csat_score": 88},
            {"review_id": "review-3", "review_text": "Urgency score is 9.", "urgency_score": 9},
        ],
    )

    payload = result.as_dict()
    assert payload["generated"] == 2
    assert [stat["source_id"] for stat in payload["stats"]] == ["review-1", "review-2"]


@pytest.mark.asyncio
async def test_stat_card_service_rejects_invalid_limits() -> None:
    service = StatCardGenerationService()

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
