from __future__ import annotations

from html.parser import HTMLParser
import json
import subprocess
import sys
from pathlib import Path

import markdown
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


class _RenderedFAQHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1: list[str] = []
        self.h2: list[str] = []
        self.paragraphs: list[str] = []
        self.list_items: list[str] = []
        self.strong: list[str] = []
        self.ul_count = 0
        self._stack: list[str] = []
        self._buffers: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._stack.append(tag)
        if tag == "ul":
            self.ul_count += 1
        if tag in {"h1", "h2", "p", "li", "strong"}:
            self._buffers[tag] = []

    def handle_data(self, data: str) -> None:
        for tag in ("h1", "h2", "p", "li", "strong"):
            if tag in self._stack and tag in self._buffers:
                self._buffers[tag].append(data)

    def handle_endtag(self, tag: str) -> None:
        text = " ".join("".join(self._buffers.pop(tag, [])).split())
        if text:
            if tag == "h1":
                self.h1.append(text)
            elif tag == "h2":
                self.h2.append(text)
            elif tag == "p":
                self.paragraphs.append(text)
            elif tag == "li":
                self.list_items.append(text)
            elif tag == "strong":
                self.strong.append(text)
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        elif tag in self._stack:
            self._stack.remove(tag)


class _FAQRepository:
    def __init__(self) -> None:
        self.saved = []

    async def save_drafts(self, drafts, *, scope):
        self.saved.append({"drafts": drafts, "scope": scope})
        return ("faq-uuid-1",)


def _write_ticket_csv(tmp_path: Path, *rows: str) -> Path:
    source = tmp_path / "tickets.csv"
    source.write_text(
        "\n".join((
            "Ticket ID,Created At,Subject,Description,Pain Category",
            *rows,
            "",
        )),
        encoding="utf-8",
    )
    return source


def _run_ticket_faq_cli(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--source-format",
            "csv",
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


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


def test_ticket_faq_markdown_renders_action_and_source_lists_from_packaged_rows() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities)
    rendered = _RenderedFAQHTML()
    rendered.feed(markdown.markdown(result.markdown))

    assert rendered.h1 == ["Customer Ticket FAQ"]
    assert rendered.h2 == [
        "1. What are customers asking about manual follow-up?",
        "2. What are customers asking about reporting friction?",
    ]
    assert rendered.strong.count("What to do next:") == 2
    assert rendered.strong.count("Sources:") == 2
    assert rendered.ul_count == 4
    assert any(
        "Support notes show campaign handoffs are still being reconciled manually"
        in paragraph
        for paragraph in rendered.paragraphs
    )
    assert any(
        "Check the workflow or automation rule that should handle this step."
        in item
        for item in rendered.list_items
    )
    assert any(
        "ticket-acme-1 - Reporting export blocked before renewal" in item
        for item in rendered.list_items
    )
    assert len(result.items) == 2
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }


def test_build_ticket_faq_markdown_filters_to_requested_date_window() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "createdAt": "2026-05-01T12:00:00Z",
                "pain_points": ["login"],
                "evidence": [{
                    "text": "How do I update my login email?",
                    "source_id": "ticket-new",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "created_at": "2026-01-01",
                "pain_points": ["billing"],
                "evidence": [{
                    "text": "Billing export is confusing.",
                    "source_id": "ticket-old",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "pain_points": ["exports"],
                "evidence": [{
                    "text": "Export settings are missing.",
                    "source_id": "ticket-undated",
                    "source_type": "support_ticket",
                }],
            },
        ],
        window_days=90,
        as_of_date="2026-05-20",
    )

    assert result.ticket_source_count == 1
    assert "ticket-new" in result.markdown
    assert "ticket-old" not in result.markdown
    assert "ticket-undated" not in result.markdown
    with pytest.raises(ValueError, match="window_days must be positive"):
        build_ticket_faq_markdown([], window_days=0)
    with pytest.raises(ValueError, match="as_of_date must be a valid ISO date"):
        build_ticket_faq_markdown([], window_days=90, as_of_date="not-a-date")
    with pytest.raises(ValueError, match="as_of_date must be a valid ISO date"):
        build_ticket_faq_markdown([], window_days=90, as_of_date="2026-99-99")
    with pytest.raises(ValueError, match="as_of_date must be a valid ISO date"):
        build_ticket_faq_markdown([], window_days=90, as_of_date="2026-05-20abc")
    with pytest.raises(ValueError, match="as_of_date requires window_days"):
        build_ticket_faq_markdown([], as_of_date="2026-05-20")


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
            "created_at": "2026-05-01",
            "message": "The attribution export is missing renewals.",
            "pain_category": "exports",
        }],
        title="Renewal FAQ",
        window_days=90,
        as_of_date="2026-05-20",
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
    assert draft.metadata["window_days"] == 90
    assert draft.metadata["as_of_date"] == "2026-05-20"


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


def test_ticket_faq_cli_filters_csv_to_date_window(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
            "ticket-old,2026-01-01,Billing export,Billing export is confusing.,billing",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--window-days",
        "90",
        "--as-of-date",
        "2026-05-20",
    )

    assert completed.returncode == 0
    assert "ticket-new" in completed.stdout
    assert "ticket-old" not in completed.stdout
    assert "Ticket evidence rows used: 1" in completed.stdout


@pytest.mark.parametrize("value", ("2026-99-99", "2026-05-20abc", "2026/05/20"))
def test_ticket_faq_cli_rejects_invalid_as_of_date(tmp_path: Path, value: str) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--window-days",
        "90",
        "--as-of-date",
        value,
    )

    assert completed.returncode != 0
    assert "--as-of-date must use YYYY-MM-DD format" in completed.stderr


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


def test_ticket_faq_cli_rejects_as_of_date_without_window(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
    )
    completed = _run_ticket_faq_cli(source, "--as-of-date", "2026-05-20")

    assert completed.returncode == 1
    assert "--as-of-date requires --window-days" in completed.stderr
