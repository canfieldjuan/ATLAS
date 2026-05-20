from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.ticket_faq_markdown import (
    TicketFAQMarkdownService,
    build_ticket_faq_markdown,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/build_extracted_ticket_faq_markdown.py"
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
SUPPORT_TICKET_BUNDLE = ROOT / "extracted_content_pipeline/examples/support_ticket_bundle.json"


class _FAQRepository:
    def __init__(self) -> None:
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": drafts, "scope": scope})
        return ("faq-uuid-1",)


def test_build_ticket_faq_markdown_groups_grounded_ticket_evidence() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities)

    assert result.source_count == 2
    assert result.ticket_source_count == 2
    assert [item["topic"] for item in result.items] == ["manual follow-up", "reporting friction"]
    assert "campaign attribution data" in result.markdown
    assert "`ticket-acme-1` - Reporting export blocked before renewal" in result.markdown
    assert "**What to do next:**" in result.markdown
    assert "Check whether your plan and role include the needed export" in result.markdown
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }


@pytest.mark.asyncio
async def test_ticket_faq_service_generates_from_inline_source_material() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "subject": "Login email change",
                    "source_type": "ticket",
                    "message": "How do I change my email address?",
                    "pain_category": "login",
                }
            ]
        },
        max_items=2,
    )

    assert result.as_dict()["generated"] == 1
    assert "How do I change my email address?" in result.markdown
    assert "`ticket-1` - Login email change" in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_preserves_explicit_empty_source_type_filter() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[{
            "source_id": "review-1",
            "source_type": "review",
            "text": "Export settings are hard to find.",
            "pain_category": "exports",
        }],
        source_types=(),
    )

    assert result.as_dict()["generated"] == 1
    assert "Export settings are hard to find." in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_exposes_source_normalization_warnings() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "ticket_id": "ticket-1",
                "source_type": "ticket",
                "subject": "Export issue",
                "message": "The export keeps timing out.",
            },
            {
                "ticket_id": "ticket-2",
                "source_type": "ticket",
                "subject": "Missing body",
            },
        ],
    )

    assert result.as_dict()["generated"] == 1
    assert result.as_dict()["warnings"][0]["code"] == "missing_source_text"


@pytest.mark.asyncio
async def test_ticket_faq_service_skips_empty_source_material_containers() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [],
            "rows": [{
                "ticket_id": "ticket-1",
                "source_type": "ticket",
                "message": "The dashboard export is not working.",
                "pain_category": "exports",
            }],
        },
    )

    assert result.as_dict()["generated"] == 1
    assert "The dashboard export is not working." in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_saves_generated_markdown_when_repository_configured() -> None:
    repository = _FAQRepository()
    service = TicketFAQMarkdownService(ticket_faqs=repository)

    result = await service.generate(
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
        target_mode="vendor_retention",
        source_material=[{
            "ticket_id": "ticket-1",
            "source_type": "ticket",
            "message": "The attribution export is missing renewals.",
            "pain_category": "exports",
        }],
        title="Renewal FAQ",
    )

    assert result.as_dict()["saved_ids"] == ["faq-uuid-1"]
    assert len(repository.saved) == 1
    draft = repository.saved[0]["drafts"][0]
    assert draft.target_id == "ticket-1"
    assert draft.target_mode == "vendor_retention"
    assert draft.title == "Renewal FAQ"
    assert "The attribution export is missing renewals." in draft.markdown
    assert draft.metadata["source_types"] == [
        "ticket",
        "support_ticket",
        "case",
        "conversation",
        "complaint",
    ]


@pytest.mark.asyncio
async def test_ticket_faq_service_does_not_save_empty_results() -> None:
    repository = _FAQRepository()
    service = TicketFAQMarkdownService(ticket_faqs=repository)

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=[{
            "source_id": "review-1",
            "source_type": "review",
            "text": "Pricing is high.",
        }],
    )

    assert result.as_dict()["saved_ids"] == []
    assert repository.saved == []


def test_build_ticket_faq_markdown_uses_nested_ticket_thread_text() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_BUNDLE, file_format="json")

    result = build_ticket_faq_markdown(loaded.opportunities)

    assert "Every demo follow-up still has to be rebuilt by hand." in result.markdown
    assert "The workflow automation feature is not available on the current plan." in result.markdown
    assert "`support-riverbend-2` - Manual sequence cleanup after demos" in result.markdown


def test_build_ticket_faq_markdown_skips_non_ticket_sources_and_validates_limits() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "review",
        "pain_points": ["pricing"],
        "evidence": [{"text": "Pricing is too high.", "source_id": "review-1", "source_type": "review"}],
    }])

    assert result.items == ()
    assert "No ticket FAQ items were generated." in result.markdown
    with pytest.raises(ValueError, match="max_items must be positive"):
        build_ticket_faq_markdown([], max_items=0)
    with pytest.raises(ValueError, match="max_evidence_per_item must be positive"):
        build_ticket_faq_markdown([], max_evidence_per_item=0)


def test_build_ticket_faq_markdown_accepts_ticket_source_type_alias() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "ticket",
        "pain_points": ["billing"],
        "evidence": [{"text": "I need help with billing.", "source_id": "ticket-1", "source_type": "ticket"}],
    }])

    assert result.ticket_source_count == 1
    assert result.items[0]["source_ids"] == ("ticket-1",)
    assert "I need help with billing." in result.markdown


def test_build_ticket_faq_markdown_normalizes_source_type_and_keeps_unidentified_rows() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support ticket",
            "pain_points": ["login"],
            "evidence": [{"text": "I cannot log in.", "source_type": "support-ticket"}],
        },
        {
            "source_type": "support ticket",
            "pain_points": ["login"],
            "evidence": [{"text": "I cannot log in.", "source_type": "support ticket"}],
        },
    ])

    assert len(result.items) == 1
    assert result.ticket_source_count == 2
    assert result.items[0]["evidence_count"] == 2
    assert result.items[0]["source_ids"] == ("row:1:evidence:1", "row:2:evidence:1")
    assert "Evidence comes from 2 ticket source(s)." in result.markdown


def test_build_ticket_faq_markdown_counts_distinct_source_ids_per_item() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "support_ticket",
        "pain_points": ["exports"],
        "evidence": [
            {"text": "Export failed on Monday.", "source_id": "ticket-1", "source_type": "support_ticket"},
            {"text": "Export failed again Tuesday.", "source_id": "ticket-1", "source_type": "support_ticket"},
        ],
    }])

    assert result.items[0]["evidence_count"] == 2
    assert result.items[0]["source_ids"] == ("ticket-1",)
    assert "Evidence comes from 1 ticket source(s)." in result.markdown


def test_ticket_faq_cli_writes_markdown_file(tmp_path: Path) -> None:
    output = tmp_path / "ticket_faq.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(SUPPORT_TICKET_CSV),
            "--source-format",
            "csv",
            "--title",
            "Support FAQ",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Support FAQ")
    assert "Ticket evidence rows used: 2" in markdown
    assert "ticket-acme-1" in markdown


def test_ticket_faq_cli_stdout_limits_and_result_serializes() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(SUPPORT_TICKET_BUNDLE),
            "--source-format",
            "json",
            "--max-items",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.startswith("# Customer Ticket FAQ")
    assert completed.stdout.count("## ") == 1
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")
    encoded = json.dumps(build_ticket_faq_markdown(loaded.opportunities).as_dict(), sort_keys=True)
    assert "action_items" in encoded
    assert "output_checks" in encoded
