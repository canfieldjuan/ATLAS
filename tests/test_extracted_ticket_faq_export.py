from __future__ import annotations

import pytest

from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_export import export_ticket_faq_drafts
from extracted_content_pipeline.ticket_faq_ports import TicketFAQDraft


class _Repository:
    def __init__(self, drafts=()) -> None:
        self.drafts = tuple(drafts)
        self.calls: list[dict[str, object]] = []

    async def list_drafts(self, *, scope, status=None, target_mode=None, limit=None):
        self.calls.append({
            "scope": scope,
            "status": status,
            "target_mode": target_mode,
            "limit": limit,
        })
        return self.drafts


def _draft() -> TicketFAQDraft:
    return TicketFAQDraft(
        id="faq-uuid-1",
        status="draft",
        target_id="acct_1",
        target_mode="support_account",
        title="Support FAQ",
        markdown="# Support FAQ\n\n## How do I reset login?",
        items=[{
            "question": "How do I reset login?",
            "answer": "Use the reset link.",
        }],
        source_count=3,
        ticket_source_count=2,
        output_checks={"uses_user_vocabulary": True, "has_action_items": True},
        warnings=[{"code": "thin_evidence"}],
        metadata={"scope": {"account_id": "acct_1"}},
    )


@pytest.mark.asyncio
async def test_export_ticket_faq_drafts_passes_filters_to_repository() -> None:
    repo = _Repository([_draft()])

    result = await export_ticket_faq_drafts(
        repo,
        scope=TenantScope(account_id="acct_1"),
        status="draft",
        target_mode="support_account",
        limit=5,
    )

    assert repo.calls == [{
        "scope": TenantScope(account_id="acct_1"),
        "status": "draft",
        "target_mode": "support_account",
        "limit": 5,
    }]
    assert result.as_dict()["filters"] == {
        "status": "draft",
        "account_id": "acct_1",
        "target_mode": "support_account",
    }
    assert result.as_dict()["rows"][0]["passed_output_checks"] == 2
    assert result.as_dict()["rows"][0]["warning_count"] == 1


@pytest.mark.asyncio
async def test_ticket_faq_draft_export_result_renders_csv() -> None:
    result = await export_ticket_faq_drafts(
        _Repository([_draft()]),
        scope={"account_id": "acct_1"},
    )

    csv_text = result.as_csv()

    assert result.as_dict()["count"] == 1
    assert csv_text.startswith("target_id,target_mode,title")
    assert "Support FAQ" in csv_text
    assert "uses_user_vocabulary" in csv_text


@pytest.mark.asyncio
async def test_export_ticket_faq_drafts_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="limit must be non-negative"):
        await export_ticket_faq_drafts(_Repository(), limit=-1)
