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
    TicketFAQMarkdownConfig,
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
        "uses_user_vocabulary": False,
        "condensed": False,
        "has_action_items": True,
    }


def test_build_ticket_faq_markdown_clusters_repeated_user_intent() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Profile change question",
                "evidence": [{
                    "text": "How do I change my login email?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Account access issue",
                "evidence": [{
                    "text": "I need to update the email on my account.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert [item["topic"] for item in result.items] == ["email and profile updates"]
    assert result.items[0]["question"] == "How do I change my login email?"
    assert result.items[0]["question_source"] == "customer_wording"
    assert result.items[0]["evidence_count"] == 2
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2")
    assert "How do I change my login email?" in result.markdown
    assert "I need to update the email on my account." in result.markdown
    assert result.output_checks["uses_user_vocabulary"] is True
    assert result.output_checks["condensed"] is True


def test_build_ticket_faq_markdown_falls_back_to_topic_question() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Export issue",
                "evidence": [{
                    "text": "The dashboard export button disappears for analysts.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["question"] == "What are customers asking about reporting friction?"
    assert result.items[0]["question_source"] == "topic_fallback"
    assert result.output_checks["uses_user_vocabulary"] is False


def test_build_ticket_faq_markdown_falls_back_from_long_customer_question() -> None:
    long_question = "How do I " + ("change every nested account setting " * 8).strip() + "?"
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Profile update",
                "evidence": [{
                    "text": long_question,
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert len(long_question) > 140
    assert result.items[0]["topic"] == "email and profile updates"
    assert result.items[0]["question"] == "What are customers asking about email and profile updates?"
    assert result.items[0]["question_source"] == "topic_fallback"


def test_build_ticket_faq_markdown_extracts_question_sentence_from_ticket_text() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "For context, I tried updating profile settings all morning. How do I reset my password?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_normalizes_missing_question_mark() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my password",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_turns_first_person_issue_into_customer_question() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "I cannot reset my password from the login page.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password from the login page?"
    assert result.items[0]["question_source"] == "customer_wording"
    assert result.output_checks["uses_user_vocabulary"] is True


def test_build_ticket_faq_markdown_strips_customer_speaker_label() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "Customer: How do I reset my password",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_ignores_agent_questions() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "Customer: Login is broken. Agent: Can you share a screenshot?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "What are customers asking about login reset?"
    assert result.items[0]["question_source"] == "topic_fallback"


def test_build_ticket_faq_markdown_uses_unlabeled_customer_text_before_agent_label() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my password?\nAgent: Can you share a screenshot?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_keeps_inline_support_colon_as_customer_text() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "I copied this from support: How do I reset my password?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_ignores_url_query_markers() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Help article",
                "evidence": [{
                    "text": "I opened https://example.com/help?article=123 and the page is blank.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert result.items[0]["question"] == "What are customers asking about help article?"
    assert result.items[0]["question_source"] == "topic_fallback"


def test_build_ticket_faq_markdown_accepts_host_intent_rules() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Data sync is behind",
                "evidence": [{
                    "text": "The warehouse sync is delayed every morning.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Connector lag",
                "evidence": [{
                    "text": "Our CRM connector does not finish before standup.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        intent_rules=(("data freshness", ("warehouse sync", "connector lag")),),
    )

    assert [item["topic"] for item in result.items] == ["data freshness"]
    assert result.items[0]["evidence_count"] == 2


def test_build_ticket_faq_markdown_keeps_total_volume_when_display_is_capped() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": f"Sync delay {index}",
                "evidence": [{
                    "text": f"The warehouse sync is delayed for team {index}.",
                    "source_id": f"ticket-{index}",
                    "source_type": "support_ticket",
                }],
            }
            for index in range(1, 5)
        ],
        max_evidence_per_item=2,
        intent_rules=(("data freshness", ("warehouse sync",)),),
    )

    item = result.items[0]
    assert item["topic"] == "data freshness"
    assert item["ticket_count"] == 4
    assert item["evidence_count"] == 2
    assert item["displayed_evidence_count"] == 2
    assert item["source_ids"] == ("ticket-1", "ticket-2", "ticket-3", "ticket-4")
    assert len(item["source_labels"]) == 2
    assert "Evidence comes from 4 ticket source(s)." in item["answer"]
    assert "ticket-3" not in result.markdown


def test_build_ticket_faq_markdown_normalizes_intent_whitespace() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Profile update",
                "evidence": [{
                    "text": "How do I change\nmy\temail before renewal?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }
        ]
    )

    assert [item["topic"] for item in result.items] == ["email and profile updates"]


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
        "uses_user_vocabulary": False,
        "condensed": False,
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
async def test_ticket_faq_service_uses_configured_intent_rules() -> None:
    service = TicketFAQMarkdownService(
        TicketFAQMarkdownConfig(intent_rules=(("access setup", ("invite link", "new user")),))
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "source_type": "support_ticket",
                    "subject": "Invite link does not work",
                    "message": "The invite link expired before the new user joined.",
                },
                {
                    "ticket_id": "ticket-2",
                    "source_type": "support_ticket",
                    "subject": "New user cannot get in",
                    "message": "A new user needs another invite link.",
                },
            ]
        },
    )

    assert [item["topic"] for item in result.items] == ["access setup"]
    assert result.items[0]["evidence_count"] == 2


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
    assert result.output_checks == {
        "uses_user_vocabulary": False,
        "condensed": False,
        "has_action_items": False,
    }
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
    assert result.items[0]["source_ids"] == ("row:1", "row:2")
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
    assert result.ticket_source_count == 1
    assert "Evidence comes from 1 ticket source(s)." in result.markdown


def test_build_ticket_faq_markdown_counts_distinct_ticket_sources_for_output_checks() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "support_ticket",
        "source_title": "Login reset",
        "evidence": [
            {
                "text": "How do I reset my password?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            },
            {
                "text": "How do I update the account email?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "source_title": "Profile change question",
            },
        ],
    }])

    assert result.ticket_source_count == 1
    assert result.items[0]["source_ids"] == ("ticket-1",)
    assert result.output_checks["condensed"] is True


def test_build_ticket_faq_markdown_counts_unidentified_source_rows_once() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "support_ticket",
        "source_title": "Login reset",
        "evidence": [
            {"text": "How do I reset my password?", "source_type": "support_ticket"},
            {"text": "How do I update the account email?", "source_type": "support_ticket"},
        ],
    }])

    assert result.ticket_source_count == 1
    assert result.items[0]["source_ids"] == ("row:1",)
    assert result.output_checks["condensed"] is True


def test_build_ticket_faq_markdown_does_not_treat_max_items_truncation_as_condensed() -> None:
    opportunities = [
        {
            "source_type": "support_ticket",
            "source_title": f"Unique issue {index}",
            "evidence": [{
                "text": f"How do I handle unique issue {index}?",
                "source_id": f"ticket-{index}",
                "source_type": "support_ticket",
            }],
        }
        for index in range(1, 10)
    ]

    result = build_ticket_faq_markdown(opportunities, max_items=8)

    assert len(result.items) == 8
    assert result.ticket_source_count == 9
    assert result.output_checks["uses_user_vocabulary"] is True
    assert result.output_checks["condensed"] is False


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
    assert "Ticket sources used: 2" in markdown
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
    assert "Ticket sources used: 1" in completed.stdout


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
