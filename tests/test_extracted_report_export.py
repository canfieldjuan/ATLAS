from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.report_export import (
    ReportDraftExportResult,
    export_report_drafts,
)
from extracted_content_pipeline.report_ports import ReportDraft, ReportSection


class _Repository:
    def __init__(self, drafts=None) -> None:
        self.drafts = tuple(drafts or ())
        self.list_calls: list[dict] = []

    async def save_drafts(self, drafts, *, scope):
        raise NotImplementedError

    async def list_drafts(
        self,
        *,
        scope,
        status=None,
        target_mode=None,
        report_type=None,
        limit=None,
    ):
        self.list_calls.append(
            {
                "scope": scope,
                "status": status,
                "target_mode": target_mode,
                "report_type": report_type,
                "limit": limit,
            }
        )
        return self.drafts

    async def update_status(self, report_id, status, *, scope):
        raise NotImplementedError


def _draft(**overrides) -> ReportDraft:
    draft = ReportDraft(
        target_id="acme",
        target_mode="vendor_retention",
        report_type="vendor_pressure",
        title="Acme pressure report",
        summary="Renewal pricing pressure dominates.",
        sections=(
            ReportSection(
                id="summary",
                title="Summary",
                body_markdown="Pricing is the main pressure signal.",
                claim_ids=("c1",),
                evidence_ids=("r1",),
            ),
        ),
        reference_ids=("r1", "r2"),
        metadata={
            "generation_usage": {
                "input_tokens": 12,
                "output_tokens": 6,
                "total_tokens": 18,
            },
            "generation_parse_attempts": 2,
            "reasoning_context": {
                "wedge": "price_squeeze",
                "confidence": "high",
            },
        },
    )
    return ReportDraft(
        target_id=overrides.get("target_id", draft.target_id),
        target_mode=overrides.get("target_mode", draft.target_mode),
        report_type=overrides.get("report_type", draft.report_type),
        title=overrides.get("title", draft.title),
        summary=overrides.get("summary", draft.summary),
        sections=overrides.get("sections", draft.sections),
        reference_ids=overrides.get("reference_ids", draft.reference_ids),
        metadata=overrides.get("metadata", draft.metadata),
    )


@pytest.mark.asyncio
async def test_export_report_drafts_passes_filters_to_repository() -> None:
    repo = _Repository(drafts=[_draft()])

    result = await export_report_drafts(
        repo,
        scope={"account_id": "acct_1", "user_id": "user_1"},
        status="approved",
        target_mode="vendor_retention",
        report_type="vendor_pressure",
        limit=7,
    )

    call = repo.list_calls[0]
    assert isinstance(call["scope"], TenantScope)
    assert call["scope"].account_id == "acct_1"
    assert call["scope"].user_id == "user_1"
    assert call["status"] == "approved"
    assert call["target_mode"] == "vendor_retention"
    assert call["report_type"] == "vendor_pressure"
    assert call["limit"] == 7
    assert result.limit == 7
    assert result.filters == {
        "status": "approved",
        "account_id": "acct_1",
        "target_mode": "vendor_retention",
        "report_type": "vendor_pressure",
    }


@pytest.mark.asyncio
async def test_export_report_drafts_derives_review_summary_fields() -> None:
    result = await export_report_drafts(
        _Repository(drafts=[_draft()]),
        scope=TenantScope(account_id="acct_1"),
    )

    row = result.rows[0]
    assert row["target_id"] == "acme"
    assert row["section_count"] == 1
    assert row["reference_count"] == 2
    assert row["generation_input_tokens"] == 12
    assert row["generation_output_tokens"] == 6
    assert row["generation_total_tokens"] == 18
    assert row["generation_parse_attempts"] == 2
    assert row["reasoning_context_used"] is True
    assert row["reasoning_wedge"] == "price_squeeze"
    assert row["reasoning_confidence"] == "high"


@pytest.mark.asyncio
async def test_export_report_drafts_defaults_summary_fields_without_metadata() -> None:
    result = await export_report_drafts(
        _Repository(drafts=[_draft(metadata={})]),
        limit=1,
    )

    row = result.rows[0]
    assert row["generation_input_tokens"] is None
    assert row["generation_output_tokens"] is None
    assert row["generation_total_tokens"] is None
    assert row["generation_parse_attempts"] is None
    assert row["reasoning_context_used"] is False
    assert row["reasoning_wedge"] is None
    assert row["reasoning_confidence"] is None


@pytest.mark.asyncio
async def test_export_report_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_report_drafts(_Repository(), limit=-1)


def test_report_draft_export_result_renders_dict_and_csv() -> None:
    result = ReportDraftExportResult(
        rows=(
            {
                "target_id": "acme",
                "target_mode": "vendor_retention",
                "report_type": "vendor_pressure",
                "title": "Acme report",
                "summary": "summary",
                "section_count": 1,
                "reference_count": 1,
                "generation_input_tokens": 12,
                "generation_output_tokens": 6,
                "generation_total_tokens": 18,
                "generation_parse_attempts": 1,
                "reasoning_context_used": True,
                "reasoning_wedge": "price_squeeze",
                "reasoning_confidence": "high",
                "sections": [{"id": "summary"}],
                "reference_ids": ["r1"],
                "metadata": {"scope": {"account_id": "acct_1"}},
            },
        ),
        limit=1,
        filters={"status": "draft"},
    )

    as_dict = result.as_dict()
    csv_text = result.as_csv()

    assert as_dict["count"] == 1
    assert as_dict["rows"][0]["reasoning_wedge"] == "price_squeeze"
    assert "target_id,target_mode,report_type" in csv_text
    assert "generation_input_tokens,generation_output_tokens" in csv_text
    assert "reasoning_context_used,reasoning_wedge,reasoning_confidence" in csv_text
    assert "price_squeeze" in csv_text
