from __future__ import annotations

import csv
from html.parser import HTMLParser
import json
import math
import subprocess
import sys
from pathlib import Path

import markdown
import pytest

from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
)
from extracted_content_pipeline.campaign_ports import TenantScope
from extracted_content_pipeline.embedding_port import cosine_similarity
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.ticket_faq_markdown import (
    TicketFAQMarkdownConfig,
    TicketFAQMarkdownService,
    _output_checks,
    _resolution_advisory_signals,
    _resolution_signal_tokens,
    _resolution_text_is_publishable,
    _safe_vocabulary_representative_label,
    _question_subclusters,
    _source_date_span,
    build_ticket_faq_markdown,
    normalize_intent_rules,
    weighted_source_volume_by_group,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/build_extracted_ticket_faq_markdown.py"
SUPPORT_TICKET_CSV = ROOT / "extracted_content_pipeline/examples/support_ticket_sources.csv"
SUPPORT_TICKET_BUNDLE = ROOT / "extracted_content_pipeline/examples/support_ticket_bundle.json"
FAQ_CUSTOM_RULES = ROOT / "extracted_content_pipeline/examples/faq_custom_rules.json"
FAQ_DOCUMENTATION_TERMS = (
    ROOT / "extracted_content_pipeline/examples/faq_documentation_terms.txt"
)


class _StubEmbeddingPort:
    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self.vectors = vectors

    def embed_texts(self, texts):
        return [self.vectors.get(text, ()) for text in texts]


class _RenderedFAQHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.h1: list[str] = []
        self.h2: list[str] = []
        self.paragraphs: list[str] = []
        self.list_items: list[str] = []
        self.strong: list[str] = []
        self.ul_count = 0
        self.ol_count = 0
        self._stack: list[str] = []
        self._buffers: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        self._stack.append(tag)
        if tag == "ul":
            self.ul_count += 1
        if tag == "ol":
            self.ol_count += 1
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


def _write_source_csv(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    source = tmp_path / name
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
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


def test_source_date_span_accepts_us_export_dates() -> None:
    span = _source_date_span([
        {"source_id": "ticket-1", "date": "05/01/2026"},
        {"source_id": "ticket-2", "created_at": "5-3-26"},
    ])

    assert span == {
        "start": "2026-05-01",
        "end": "2026-05-03",
        "window_days": 3,
        "dated_source_count": 2,
        "missing_source_count": 0,
    }


def test_single_ticket_with_multiple_evidence_rows_is_not_a_repeat() -> None:
    # Codex review finding on PR #1486: density must be measured in DISTINCT
    # source tickets, not evidence rows. One ticket contributing two rows of
    # the same question is still one ticket asking once.
    opportunities = [
        {
            "source_id": "ticket-solo-1",
            "source_type": "support_ticket",
            "evidence": [
                {
                    "text": "How do I rotate the workspace signing key safely?",
                    "source_id": "ticket-solo-1",
                    "source_type": "support_ticket",
                },
                {
                    "text": (
                        "Follow-up: how do I rotate the workspace signing "
                        "key safely?"
                    ),
                    "source_id": "ticket-solo-1",
                    "source_type": "support_ticket",
                },
            ],
        },
    ]

    result = build_ticket_faq_markdown(opportunities)

    assert result.items == ()
    assert result.non_repeat_ticket_count == 1
    assert result.non_repeat_question_count >= 1
    warning_codes = [warning["code"] for warning in result.warnings]
    assert "non_repeat_tickets_excluded" in warning_codes


def test_build_ticket_faq_markdown_groups_grounded_ticket_evidence() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="1-800-555-0100")

    assert result.source_count == 4
    assert result.ticket_source_count == 4
    assert [item["topic"] for item in result.items] == [
        "reporting friction",
        "email and profile updates",
    ]
    assert [item["ticket_count"] for item in result.items] == [2, 2]
    assert result.items[0]["summary"].startswith(
        "Customers are asking about reporting friction across 2 ticket sources."
    )
    assert result.items[0]["steps"] == (
        "Review the cited ticket evidence and confirm the policy-approved answer before publishing.",
        "Draft the customer-facing steps from a verified help article, runbook, macro, or resolved ticket.",
        "If it still does not work, contact support at 1-800-555-0100 and include the cited ticket details.",
    )
    assert result.items[0]["answer_evidence_status"] == "draft_needs_review"
    assert result.items[0]["resolution_source_count"] == 0
    assert result.items[0]["failure_risk_score"] == 1
    assert result.items[0]["failure_risk_signals"] == ("blocked_access",)
    assert result.items[0]["opportunity_score"] == 4
    assert "the export is missing" in result.items[0]["when_to_contact_support"]
    assert result.items[0]["evidence_quotes"] == (
        '`ticket-northstar-1` - Export campaign reports: "How do we export the campaign reporting dashboard before renewal?"',
        '`ticket-northstar-2` - Reporting dashboard export: "We cannot export the campaign reporting dashboard before renewal."',
    )
    assert "How do I change my login email?" in result.markdown
    assert "How do we export the campaign reporting dashboard before renewal?" in result.markdown
    assert "`ticket-acme-1` - Change login email" in result.markdown
    assert "**What to do next:**" in result.markdown
    assert "self-service" not in result.markdown.lower()
    assert "contact support at 1-800-555-0100" in result.markdown
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }


def test_build_ticket_faq_markdown_does_not_invent_support_contact() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities)

    assert "contact support at" not in result.markdown.lower()
    assert "If it still does not work, contact support and include the cited ticket details." in result.markdown


def test_build_ticket_faq_markdown_uses_resolution_evidence_for_steps() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export issue",
            "evidence": [{
                "text": "How do I export the attribution dashboard before renewal?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Open Analytics, choose the attribution dashboard, and select "
                    "Export CSV. Ask an admin to enable Report Downloads if the "
                    "button is hidden."
                ),
            }],
        }],
        support_contact="support@example.com",
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "Open Analytics, choose the attribution dashboard, and select Export CSV."
    )
    assert item["steps"][1] == (
        "Ask an admin to enable Report Downloads if the button is hidden."
    )
    assert item["steps"][2] == (
        "If it still does not work, contact support at support@example.com and "
        "include the cited ticket details."
    )
    assert "Customers mention:" not in item["answer"]
    assert item["answer"] == (
        "To resolve this, open Analytics, choose the attribution dashboard, and "
        "select Export CSV. Then ask an admin to enable Report Downloads if the "
        "button is hidden."
    )
    assert "Verified resolution evidence" not in item["answer"]
    assert "Draft the customer-facing steps" not in result.markdown
    assert "support@example.com" in result.markdown


def test_build_ticket_faq_markdown_rejects_closure_boilerplate_as_resolution() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export issue",
            "evidence": [
                {
                    "text": "How do I export billing reports?",
                    "source_id": "ticket-closure-1",
                    "source_type": "support_ticket",
                    "resolution_text": "Customer did not respond, closing this out.",
                },
                {
                    "text": "How do I export billing reports?",
                    "source_id": "ticket-closure-2",
                    "source_type": "support_ticket",
                    "resolution_text": "Customer did not respond, closing this out.",
                },
            ],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert item["steps"][0].startswith("Review the cited ticket evidence")
    assert "closing this out" not in result.markdown


def test_build_ticket_faq_markdown_rejects_internal_notes_as_resolution() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Refund issue",
            "evidence": [
                {
                    "text": "How do I get a refund for a duplicate charge?",
                    "source_id": "ticket-internal-1",
                    "source_type": "support_ticket",
                    "resolution_text": "Refunded per policy 4.2. Escalated to T2 for review.",
                },
                {
                    "text": "How do I get a refund for a duplicate charge?",
                    "source_id": "ticket-internal-2",
                    "source_type": "support_ticket",
                    "resolution_text": "Refunded per policy 4.2. Escalated to T2 for review.",
                },
            ],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert "policy 4.2" not in result.markdown
    assert "Escalated to T2" not in result.markdown


def test_build_ticket_faq_markdown_keeps_legitimate_policy_resolution() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Refund issue",
            "evidence": [{
                "text": "How do I request a refund under the billing policy?",
                "source_id": "ticket-policy-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Open Billing, choose Refunds, then review the refund policy "
                    "before submitting the request."
                ),
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "Open Billing, choose Refunds, then review the refund policy before "
        "submitting the request."
    )


def test_build_ticket_faq_markdown_uses_row_context_for_resolution_topic_match() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Receipt export",
            "pain_category": "billing export",
            "tags": "billing,invoices,resolved",
            "evidence": [{
                "text": "Can finance export subscription receipts without admin access?",
                "source_id": "ticket-context-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Open Billing then Invoices, filter by the quarter, and "
                    "download the PDF from the invoice row."
                ),
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "Open Billing then Invoices, filter by the quarter, and download the "
        "PDF from the invoice row."
    )


@pytest.mark.parametrize(
    ("question_text", "resolution_text", "expected_step"),
    (
        (
            "I cannot log in to my account.",
            "Reset the password and send a temporary code.",
            "Reset the password and send a temporary code.",
        ),
        (
            "How do I sign in after lockout?",
            "Reset the password and send a backup code.",
            "Reset the password and send a backup code.",
        ),
        (
            "Why does authentication fail on mobile?",
            "Clear saved credentials and reset the password.",
            "Clear saved credentials and reset the password.",
        ),
        (
            "How do I get receipts for finance?",
            "Open Billing, choose Invoices, and download the PDF.",
            "Open Billing, choose Invoices, and download the PDF.",
        ),
        (
            "Can I connect Salesforce?",
            "Open Integrations and refresh the sync.",
            "Open Integrations and refresh the sync.",
        ),
        (
            "How do I stop renewal?",
            "Open Billing and cancel the subscription.",
            "Open Billing and cancel the subscription.",
        ),
    ),
)
def test_build_ticket_faq_markdown_keeps_synonymous_resolution_topics(
    question_text: str,
    resolution_text: str,
    expected_step: str,
) -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Synonymous support wording",
            "evidence": [{
                "text": question_text,
                "source_id": "ticket-synonym-1",
                "source_type": "support_ticket",
                "resolution_text": resolution_text,
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == expected_step


@pytest.mark.parametrize(
    ("question_text", "resolution_text"),
    (
        (
            "I cannot log in to my account.",
            "Open Billing and download the invoice PDF.",
        ),
        (
            "How do I get receipts for finance?",
            "Reset the password and send a temporary code.",
        ),
    ),
)
def test_build_ticket_faq_markdown_publishes_off_topic_instruction_overlap_demoted(
    question_text: str,
    resolution_text: str,
) -> None:
    # #1466 Option 1: question-topic overlap is demoted from a hard reject to an
    # advisory signal. An off-topic-but-genuine instruction now PUBLISHES (it
    # passes the reject filters + instruction-shape gate); the advisory
    # `topic_aligned` signal still reports the mismatch for a future confidence
    # surface, but it no longer blocks publication.
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Off-topic support wording",
            "evidence": [
                {
                    "text": question_text,
                    "source_id": "ticket-off-topic-synonym-1",
                    "source_type": "support_ticket",
                    "resolution_text": resolution_text,
                },
                {
                    "text": question_text,
                    "source_id": "ticket-off-topic-synonym-2",
                    "source_type": "support_ticket",
                    "resolution_text": resolution_text,
                },
            ],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    advisory = _resolution_advisory_signals(
        _resolution_signal_tokens(resolution_text), question_text
    )
    assert advisory["topic_aligned"] is False


def test_resolution_advisory_topic_aligned_distinguishes_on_and_off_topic() -> None:
    # The demoted overlap signal still computes correctly: on-topic resolutions
    # align, off-topic ones do not (kept for a future confidence surface).
    on_topic = _resolution_advisory_signals(
        _resolution_signal_tokens("Reset the password and confirm the new login."),
        "I cannot log in to my account.",
    )
    off_topic = _resolution_advisory_signals(
        _resolution_signal_tokens("Open Billing and download the invoice PDF."),
        "I cannot log in to my account.",
    )
    assert on_topic["topic_aligned"] is True
    assert off_topic["topic_aligned"] is False


def test_build_ticket_faq_markdown_keeps_past_tense_action_resolution() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "MFA settings",
            "evidence": [{
                "text": "How do I update MFA settings?",
                "source_id": "ticket-past-tense-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "I enabled MFA and configured the authenticator, then "
                    "updated the settings."
                ),
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "I enabled MFA and configured the authenticator, then updated the settings."
    )


@pytest.mark.parametrize(
    "resolution_text",
    (
        "Reviewed the billing account and replied to the customer.",
        "Checked the account and sent the customer an update.",
        "Reviewed the billing account and sent an update to the customer.",
        "Checked the account and provided an update to the requester.",
        "Started reviewing the billing account and sent an update to the customer.",
    ),
)
def test_build_ticket_faq_markdown_rejects_disposition_only_agent_updates(
    resolution_text: str,
) -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Billing account",
            "evidence": [
                {
                    "text": "How do I update the billing account?",
                    "source_id": "ticket-disposition-1",
                    "source_type": "support_ticket",
                    "resolution_text": resolution_text,
                },
                {
                    "text": "How do I update the billing account?",
                    "source_id": "ticket-disposition-2",
                    "source_type": "support_ticket",
                    "resolution_text": resolution_text,
                },
            ],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert item["steps"][0].startswith("Review the cited ticket evidence")
    assert resolution_text not in result.markdown


def test_build_ticket_faq_markdown_keeps_concrete_step_after_account_review() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Billing account",
            "evidence": [{
                "text": "How do I update the billing account?",
                "source_id": "ticket-concrete-review-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Checked the billing account, opened Invoices, and updated "
                    "the payment method."
                ),
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "Checked the billing account, opened Invoices, and updated the payment method."
    )


def test_build_ticket_faq_markdown_keeps_concrete_started_return_step() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Product return",
            "evidence": [{
                "text": "How do I return a recent purchase from your store?",
                "source_id": "ticket-return-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Items should be returned within 30 days in their original "
                    "condition. Start the return in the online returns portal."
                ),
            }],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == (
        "Items should be returned within 30 days in their original condition."
    )
    assert item["steps"][1] == "Start the return in the online returns portal."


def test_build_ticket_faq_markdown_rejects_generic_response_metadata_as_resolution() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Response metadata is not resolution evidence",
            "support_ticket_cluster": "metadata response fields",
            "evidence": [
                {
                    "text": "Where is the export button?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                    "first_response": "Thanks for contacting support; we received your ticket.",
                },
                {
                    "text": "I cannot find the export button.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                    "last_response": "First response SLA met in 22 minutes.",
                },
                {
                    "text": "We cannot find the export button.",
                    "source_id": "ticket-3",
                    "source_type": "support_ticket",
                    "reply_text": "Auto-ack sent by routing workflow.",
                },
            ],
        }]
    )

    assert result.items
    assert {
        item["answer_evidence_status"] for item in result.items
    } == {"draft_needs_review"}
    assert {item["resolution_source_count"] for item in result.items} == {0}
    assert "Thanks for contacting support" not in result.markdown
    assert "SLA met" not in result.markdown
    assert "Auto-ack" not in result.markdown


def test_build_ticket_faq_markdown_fails_closed_when_resolution_has_no_question_scope() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export issue",
            "evidence": [{
                "text": "The export button disappears for analysts.",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "resolution_text": "Enable Report Downloads for the analyst role.",
            }],
        }]
    )

    item = result.items[0]
    assert item["question_source"] == "source_policy"
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "missing_question_scope"
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": False,
    }


def test_build_ticket_faq_markdown_truncates_resolution_steps_at_word_boundary() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "SSO setup",
            "evidence": [{
                "text": "How do I enable SSO after my domain is verified?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "resolution_text": (
                    "Open Settings > Security > Single sign-on, confirm the "
                    "verified domain badge is green, paste the IdP metadata URL, "
                    "map email to NameID, click Test SSO, then Save and enforce "
                    "for selected users before rollout."
                ),
            }],
        }]
    )

    step = result.items[0]["steps"][0]
    assert step == (
        "Open Settings > Security > Single sign-on, confirm the verified domain "
        "badge is green, paste the IdP metadata URL, map email to NameID, click "
        "Test SSO, then Save and enforce for..."
    )
    assert "se..." not in step


def test_build_ticket_faq_markdown_ignores_disposition_resolution_aliases() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export issue",
            "resolution": "Closed",
            "answer": "Yes",
            "solution": "Escalated",
            "evidence": [
                {
                    "text": "The reporting dashboard export is missing for my analyst role.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                },
                {
                    "text": "The reporting dashboard export is missing for our analyst role.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                },
            ],
        }]
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert item["steps"][0].startswith("Review the cited ticket evidence")
    assert "Use the uploaded resolution evidence: Closed" not in result.markdown


def test_build_ticket_faq_markdown_fails_closed_for_unscoped_resolution_beyond_display_rows() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Export issue",
                "evidence": [{
                    "text": "How do I export the attribution dashboard?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Export issue",
                "evidence": [{
                    "text": "The dashboard export is missing for analysts.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                    "resolution_text": "Enable Report Downloads for the analyst role.",
                }],
            },
        ],
        max_evidence_per_item=1,
    )

    item = result.items[0]
    assert item["question_source"] == "source_policy"
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "missing_question_scope"
    assert item["resolution_source_count"] == 1
    assert item["steps"][0] == "Enable Report Downloads for the analyst role."
    assert result.output_checks["resolution_evidence_scoped"] is False


def test_build_ticket_faq_markdown_counts_resolution_sources_not_unique_texts() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Login reset",
                "evidence": [{
                    "text": "I cannot reset my password.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                    "resolution_text": "Send the reset email from Account Settings.",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Login reset",
                "evidence": [{
                    "text": "How do I reset my password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                    "resolution_text": "Send the reset email from Account Settings.",
                }],
            },
        ],
        documentation_terms=("Export issue",),
    )

    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
    assert item["resolution_source_count"] == 2
    assert item["steps"][0] == "Send the reset email from Account Settings."
    assert item["answer"] == (
        "To resolve this, send the reset email from Account Settings."
    )
    assert all(not step.startswith("Use the uploaded") for step in item["steps"])


def test_build_ticket_faq_markdown_keeps_distinct_questions_from_sharing_resolutions() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "SCIM setup",
                "evidence": [{
                    "text": "Can we sync users before enforcing SSO?",
                    "source_id": "ticket-scim-1",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Enable SSO first, confirm every existing user email matches "
                        "the identity provider email, then enable SCIM in Preview mode."
                    ),
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Warehouse sync",
                "evidence": [{
                    "text": "Why is my dashboard data not refreshing after the warehouse sync?",
                    "source_id": "ticket-warehouse-1",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Check Data > Sync history for the warehouse job, then open "
                        "Analytics > Model refresh."
                    ),
                }],
            },
        ],
        documentation_terms=("Export issue",),
    )

    by_question = {item["question"]: item for item in result.items}
    scim = by_question["Can we sync users before enforcing SSO?"]
    warehouse = by_question[
        "Why is my dashboard data not refreshing after the warehouse sync?"
    ]

    assert scim["answer_evidence_status"] == "resolution_evidence"
    assert warehouse["answer_evidence_status"] == "resolution_evidence"
    assert scim["resolution_evidence_scope"] == "scoped"
    assert warehouse["resolution_evidence_scope"] == "scoped"
    assert scim["source_ids"] == ("ticket-scim-1",)
    assert warehouse["source_ids"] == ("ticket-warehouse-1",)
    assert "SCIM in Preview mode" in " ".join(scim["steps"])
    assert "Analytics > Model refresh" not in " ".join(scim["steps"])
    assert "Analytics > Model refresh" in " ".join(warehouse["steps"])
    assert "SCIM in Preview mode" not in " ".join(warehouse["steps"])


def test_build_ticket_faq_markdown_scopes_overflow_resolution_to_item_question() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "SCIM setup",
                "evidence": [{
                    "text": "Can we sync users before enforcing SSO?",
                    "source_id": "ticket-scim-1",
                    "source_type": "support_ticket",
                    "source_weight": 10,
                    "resolution_text": (
                        "Enable SSO first, confirm existing user emails, then enable "
                        "SCIM in Preview mode."
                    ),
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Warehouse sync",
                "evidence": [{
                    "text": "Why is my dashboard data not refreshing after the warehouse sync?",
                    "source_id": "ticket-warehouse-1",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Check Data > Sync history for the warehouse job, then open "
                        "Analytics > Model refresh."
                    ),
                }],
            },
        ],
        max_items=1,
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item["question"] == "Which remaining support questions need manual review?"
    assert item["source_ids"] == ("ticket-scim-1", "ticket-warehouse-1")
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert "SCIM in Preview mode" not in " ".join(item["steps"])
    assert "Analytics > Model refresh" not in " ".join(item["steps"])


def test_build_ticket_faq_markdown_fails_closed_for_resolved_and_unresolved_overflow() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "CSV export",
                "evidence": [{
                    "text": "How do I export reports?",
                    "source_id": "ticket-export-1",
                    "source_type": "support_ticket",
                    "source_weight": 10,
                    "resolution_text": "Open Reports, choose Export, then select CSV.",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "SSO setup",
                "evidence": [{
                    "text": "Can I turn on SSO for all users?",
                    "source_id": "ticket-sso-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "SSO setup",
                "evidence": [{
                    "text": "Can I turn on SSO for all of our users?",
                    "source_id": "ticket-sso-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        max_items=1,
    )

    assert len(result.items) == 1
    item = result.items[0]
    assert item["source_ids"] == ("ticket-export-1", "ticket-sso-1", "ticket-sso-2")
    assert item["answer_evidence_status"] == "draft_needs_review"
    assert item["resolution_evidence_scope"] == "not_applicable"
    assert item["resolution_source_count"] == 0
    assert "Open Reports" not in " ".join(item["steps"])


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


def test_ticket_faq_output_check_fails_unscoped_resolution_evidence() -> None:
    output_checks = _output_checks(
        items=({
            "question_source": "customer_wording",
            "action_items": ("Open Reports.",),
            "answer_evidence_status": "resolution_evidence",
            "resolution_evidence_scope": "scope_mismatch",
            "source_ids": ("ticket-1",),
        },),
        ticket_source_count=1,
        rendered_ticket_source_count=1,
    )

    assert output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": False,
    }


def test_build_ticket_faq_markdown_ranks_by_frequency_times_failure_risk() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "source_title": "Email update",
            "evidence": [{
                "text": "How do I change my email?",
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email settings",
            "evidence": [{
                "text": "I need to update the email on my account.",
                "source_id": "ticket-email-2",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email profile",
            "evidence": [{
                "text": "Where can I update the email?",
                "source_id": "ticket-email-3",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Billing failure",
            "evidence": [{
                "text": (
                    "The payment failed, the balance is wrong, and I cannot "
                    "pay because the billing page is locked."
                ),
                "source_id": "ticket-billing-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Billing locked",
            "evidence": [{
                "text": (
                    "I cannot pay because the payment failed, the balance is "
                    "wrong, and the billing page is locked."
                ),
                "source_id": "ticket-billing-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert [item["topic"] for item in result.items] == [
        "billing and payments",
        "email and profile updates",
    ]
    assert result.items[0]["frequency"] == 2
    assert result.items[0]["failure_risk_signals"] == (
        "blocked_access",
        "failed_workflow",
        "incorrect_record",
        "money_or_account_risk",
    )
    assert result.items[0]["failure_risk_score"] == 4
    assert result.items[0]["opportunity_score"] == 10
    assert result.items[1]["frequency"] == 3
    assert result.items[1]["failure_risk_score"] == 0
    assert result.items[1]["opportunity_score"] == 3


def test_build_ticket_faq_markdown_uses_frequency_tiebreak_after_opportunity_score() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "source_title": "Email blocked",
            "evidence": [{
                "text": "I cannot update the email address.",
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email settings",
            "evidence": [{
                "text": "How do I change the email address?",
                "source_id": "ticket-email-2",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email profile",
            "evidence": [{
                "text": "Where do I update the email address?",
                "source_id": "ticket-email-3",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Webhook failure",
            "evidence": [{
                "text": "The integration API webhook sync failed with the wrong status.",
                "source_id": "ticket-integration-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Webhook error",
            "evidence": [{
                "text": "The integration API webhook sync failed with the wrong status again.",
                "source_id": "ticket-integration-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert [(item["topic"], item["frequency"], item["opportunity_score"]) for item in result.items] == [
        ("email and profile updates", 3, 6),
        ("integration setup", 2, 6),
    ]


def test_build_ticket_faq_markdown_weights_aggregated_search_frequency() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "search_log",
            "query_id": "search-export-1",
            "search_query": "export attribution report",
            "search_count": "20",
            "evidence": [
                {
                    "text": "export attribution report",
                    "source_id": "search-export-1",
                    "source_type": "search_log",
                },
            ],
        },
        {
            "source_type": "search_log",
            "query_id": "search-export-2",
            "search_query": "export attribution reports",
            "search_count": "5",
            "evidence": [
                {
                    "text": "export attribution reports",
                    "source_id": "search-export-2",
                    "source_type": "search_log",
                },
            ],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email update",
            "evidence": [{
                "text": "How do I change my email?",
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email settings",
            "evidence": [{
                "text": "I need to update the email on my account.",
                "source_id": "ticket-email-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert [item["topic"] for item in result.items] == [
        "reporting friction",
        "email and profile updates",
    ]
    assert result.items[0]["frequency"] == 25
    assert result.items[0]["weighted_frequency"] == 25
    assert result.items[0]["ticket_count"] == 2
    assert result.items[0]["source_ids"] == ("search-export-1", "search-export-2")
    assert result.items[0]["opportunity_score"] == 25
    assert result.items[1]["frequency"] == 2
    assert result.items[1]["weighted_frequency"] == 2


def test_build_ticket_faq_markdown_prefers_explicit_aggregate_weight_fields() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "search_log",
            "query_id": "search-export-1",
            "search_query": "export attribution report",
            "frequency": "1",
            "search_count": "25",
            "evidence": [
                {
                    "text": "export attribution report",
                    "source_id": "search-export-1",
                    "source_type": "search_log",
                },
            ],
        },
        {
            "source_type": "search_log",
            "query_id": "search-export-2",
            "search_query": "export attribution reports",
            "frequency": "1",
            "search_count": "5",
            "evidence": [
                {
                    "text": "export attribution reports",
                    "source_id": "search-export-2",
                    "source_type": "search_log",
                },
            ],
        },
    ])

    assert result.items[0]["frequency"] == 30
    assert result.items[0]["weighted_frequency"] == 30
    assert result.items[0]["ticket_count"] == 2


def test_weighted_source_volume_by_group_accepts_unnormalized_weight_fields() -> None:
    result = weighted_source_volume_by_group(
        [{
            "source_type": "search_log",
            "query_id": "search-export-1",
            "search_count": "25",
        }],
        group_key=lambda row: str(row.get("source_type") or "unknown"),
        source_key=lambda row, _index: str(row.get("query_id") or "unknown"),
    )

    assert result == {"search_log": 25}


def test_build_ticket_faq_markdown_uses_max_weight_per_distinct_source() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "search_log",
        "query_id": "search-export-1",
        "search_query": "export attribution report",
        "evidence": [
            {
                "text": "export attribution report",
                "source_id": "search-export-1",
                "source_type": "search_log",
                "source_weight": "100",
            },
            {
                "text": "export attribution reports",
                "source_id": "search-export-1",
                "source_type": "search_log",
                "source_weight": "200",
            },
            {
                "text": "exported attribution report",
                "source_id": "search-export-2",
                "source_type": "search_log",
                "source_weight": "300",
            },
        ],
    }])

    assert result.items[0]["frequency"] == 500
    assert result.items[0]["weighted_frequency"] == 500
    assert result.items[0]["ticket_count"] == 2
    assert result.items[0]["source_ids"] == ("search-export-1", "search-export-2")


def test_build_ticket_faq_markdown_ranks_zero_result_searches_as_failure_risk() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "search_log",
            "query_id": "search-export-1",
            "search_query": "How do I export attribution report?",
            "results_count": 0,
            "zero_results": True,
            "evidence": [{
                "text": "How do I export attribution report?",
                "source_id": "search-export-1",
                "source_type": "search_log",
            }],
        },
        {
            "source_type": "search_log",
            "query_id": "search-export-2",
            "search_query": "export attribution reports",
            "result_count": "0",
            "evidence": [{
                "text": "export attribution reports",
                "source_id": "search-export-2",
                "source_type": "search_log",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email update",
            "evidence": [{
                "text": "How do I change my email?",
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email settings",
            "evidence": [{
                "text": "I need to update the email on my account.",
                "source_id": "ticket-email-2",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Email profile",
            "evidence": [{
                "text": "Where can I update the email?",
                "source_id": "ticket-email-3",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert [item["topic"] for item in result.items] == [
        "reporting friction",
        "email and profile updates",
    ]
    assert result.items[0]["frequency"] == 2
    assert result.items[0]["failure_risk_signals"] == ("zero_result_search",)
    assert result.items[0]["failure_risk_score"] == 1
    assert result.items[0]["opportunity_score"] == 4
    assert result.items[1]["frequency"] == 3
    assert result.items[1]["failure_risk_score"] == 0
    assert result.items[1]["opportunity_score"] == 3


def test_build_ticket_faq_markdown_adds_vocabulary_gap_from_documentation_terms() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Attribution dashboard",
                "evidence": [{
                    "text": "How do I export the attribution dashboard?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Attribution dashboard",
                "evidence": [{
                    "text": "How do we export the attribution dashboard?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=("Download report", "Analytics"),
    )

    assert result.items[0]["term_mappings"] == (
        {
            "customer_term": "export",
            "documentation_term": "Download report",
            "suggestion": (
                'Add "export" as alternate phrasing for "Download report" '
                "in FAQ headings and answer text."
            ),
            "source_id_count": 2,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 2,
            "first_source_id": "ticket-1",
        },
        {
            "customer_term": "dashboard",
            "documentation_term": "Analytics",
            "suggestion": (
                'Add "dashboard" as alternate phrasing for "Analytics" '
                "in FAQ headings and answer text."
            ),
            "source_id_count": 2,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 2,
            "first_source_id": "ticket-1",
        },
    )
    assert "**Vocabulary gaps:**" in result.markdown
    assert 'Customers say "export"; documentation says "Download report".' in result.markdown
    assert "(Seen in 2 source(s); mapping score 2.)" in result.markdown


def test_build_ticket_faq_markdown_accepts_custom_vocabulary_gap_rules() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "SSO access",
                "evidence": [{
                    "text": "How do I configure SSO for my team?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "SSO access",
                "evidence": [{
                    "text": "How do we configure SSO for our team?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=("Single sign-on setup",),
        vocabulary_gap_rules=(("SSO", "single sign-on"),),
    )

    assert result.items[0]["term_mappings"] == (
        {
            "customer_term": "SSO",
            "documentation_term": "Single sign-on setup",
            "suggestion": (
                'Add "SSO" as alternate phrasing for "Single sign-on setup" '
                "in FAQ headings and answer text."
            ),
            "source_id_count": 2,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 2,
            "first_source_id": "ticket-1",
        },
    )
    assert 'Customers say "SSO"; documentation says "Single sign-on setup".' in result.markdown


def test_build_ticket_faq_markdown_prioritizes_custom_vocabulary_gap_rules() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Export dashboard bill SSO",
                "evidence": [{
                    "text": "How do I export dashboard bill data after SSO setup?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Export dashboard bill SSO",
                "evidence": [{
                    "text": "How do we export dashboard bill data after SSO setup?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=(
            "Download report",
            "Analytics",
            "Invoice settings",
            "Single sign-on setup",
        ),
        vocabulary_gap_rules=(("SSO", "single sign-on"),),
    )

    mappings = result.items[0]["term_mappings"]
    assert len(mappings) == 3
    assert mappings[0]["customer_term"] == "SSO"
    assert mappings[0]["documentation_term"] == "Single sign-on setup"


@pytest.mark.parametrize(
    "rules",
    [
        ("SSO",),
        (("SSO",),),
        (("Export", "export"),),
    ],
)
def test_build_ticket_faq_markdown_rejects_invalid_custom_vocabulary_gap_rules(
    rules: object,
) -> None:
    with pytest.raises(ValueError, match="vocabulary_gap_rules entries"):
        build_ticket_faq_markdown(
            [{
                "source_type": "support_ticket",
                "source_title": "SSO access",
                "evidence": [{
                    "text": "How do I configure SSO for my team?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            }],
            documentation_terms=("Single sign-on setup",),
            vocabulary_gap_rules=rules,  # type: ignore[arg-type]
        )


def test_build_ticket_faq_markdown_uses_document_rows_for_vocabulary_gap_only() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "source_title": "Billing confusion",
            "evidence": [{
                "text": "Where can I find my bill?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Billing confusion",
            "evidence": [{
                "text": "Where do I find my bill?",
                "source_id": "ticket-2",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "document",
            "source_title": "Invoice settings",
            "evidence": [{
                "text": "Open invoice settings to download your statement.",
                "source_id": "doc-1",
                "source_type": "document",
                "source_title": "Invoice settings",
            }],
        },
    ])

    assert result.source_count == 3
    assert result.ticket_source_count == 2
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2")
    assert result.items[0]["term_mappings"][0]["customer_term"] == "bill"
    assert result.items[0]["term_mappings"][0]["documentation_term"] == "Invoice settings"
    assert "`doc-1`" not in result.markdown


def test_build_ticket_faq_markdown_skips_vocabulary_gap_when_docs_match_customer_term() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Export report",
                "evidence": [{
                    "text": "How do I export the attribution dashboard?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Export report",
                "evidence": [{
                    "text": "How do we export the attribution dashboard?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=("Export reports", "Dashboard analytics"),
    )

    assert result.items[0]["term_mappings"] == ()
    assert "**Vocabulary gaps:**" not in result.markdown


def test_build_ticket_faq_markdown_scores_vocabulary_gap_zero_result_searches() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "search_log",
                "query_id": "search-1",
                "search_query": "How do I export attribution report?",
                "results_count": 0,
                "evidence": [{
                    "text": "How do I export attribution report?",
                    "source_id": "search-1",
                    "source_type": "search_log",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Export attribution",
                "evidence": [{
                    "text": "I cannot export attribution data.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=("Download report",),
    )

    mapping = result.items[0]["term_mappings"][0]
    assert mapping["customer_term"] == "export"
    assert mapping["source_id_count"] == 2
    assert mapping["zero_result_source_count"] == 1
    assert mapping["failure_risk_score"] == 2
    assert mapping["failure_risk_signals"] == ("blocked_access", "zero_result_search")
    assert mapping["opportunity_score"] == 6
    assert "(Seen in 2 source(s); 1 zero-result search source(s); mapping score 6.)" in result.markdown


def test_build_ticket_faq_markdown_derives_question_from_complaint_narrative() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Checking account - Fees",
                "pain_points": ["Fees"],
                "evidence": [{
                    "text": "I was charged overdraft fees after I closed the account.",
                    "source_id": "cfpb:1",
                    "source_type": "support_ticket",
                    "source_title": "Checking account - Fees",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Checking account - Fees",
                "pain_points": ["Fees"],
                "evidence": [{
                    "text": "I was charged overdraft fees again after I closed the account.",
                    "source_id": "cfpb:2",
                    "source_type": "support_ticket",
                    "source_title": "Checking account - Fees",
                }],
            },
        ],
        support_contact="https://example.com/support",
    )

    assert result.items[0]["question"] == (
        "What should I do if I was charged overdraft fees after I closed the account?"
    )
    assert result.items[0]["question_source"] == "customer_wording"
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert "Review the cited ticket evidence and confirm the policy-approved answer" in result.markdown
    assert result.items[0]["answer_evidence_status"] == "draft_needs_review"
    assert "contact support at https://example.com/support" in result.markdown


def test_build_ticket_faq_markdown_uses_representative_label_when_customer_wording_is_missing() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "reporting friction",
                "source_title": "Export issue",
                "evidence": [{
                    "text": "The dashboard export button disappears for analysts.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "reporting friction",
                "source_title": "Export issue",
                "evidence": [{
                    "text": "The dashboard export button disappears for our analysts.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        documentation_terms=("Dashboard export",),
    )

    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["question"] == "What should I do about dashboard export?"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks["uses_user_vocabulary"] is True


def test_build_ticket_faq_markdown_uses_safe_terms_instead_of_pii_source_title() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Reset password for Sarah Chen",
                "text": "Reset password for Sarah Chen blocks support.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Reset password for Sarah Chen",
                "text": "Reset password for Sarah Chen blocks support.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
        documentation_terms=("Reset password",),
    )

    assert result.items[0]["question"] == "What should I do about reset password?"
    assert "sarah" not in result.items[0]["question"].lower()
    assert "chen" not in result.items[0]["question"].lower()


@pytest.mark.parametrize(
    ("unsafe_question", "blocked_fragment"),
    (
        ("How do I reset the password for jane.doe@acme.com?", "jane"),
        ("How do I reset the password for account 4829103?", "4829103"),
        ("How do I reset the password for 555-123-4567?", "555"),
        ("How do I reset the password for XXXXX?", "xxxxx"),
    ),
)
def test_build_ticket_faq_markdown_does_not_publish_pii_customer_wording_headings(
    unsafe_question: str,
    blocked_fragment: str,
) -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Password reset",
                "text": unsafe_question,
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Password reset",
                "text": unsafe_question,
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "What should I do about technical support?"
    assert result.items[0]["question_source"] == "source_policy"
    assert blocked_fragment not in result.items[0]["question"].lower()
    assert f"## 1. {unsafe_question}" not in result.markdown
    assert "## 1. What should I do about technical support?" in result.markdown


def test_build_ticket_faq_markdown_does_not_publish_pii_fallback_topic_or_body_text() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Case for jane.doe@acme.com",
                "text": "How can jane.doe@acme.com get help?",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "source_title": "Case for jane.doe@acme.com",
                "text": "How can jane.doe@acme.com get help?",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "What should I do about customer support issues?"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.items[0]["evidence_quotes"] == (
        "`ticket-1`: Customer-provided details omitted for privacy.",
        "`ticket-2`: Customer-provided details omitted for privacy.",
    )
    assert "jane" not in result.markdown.lower()
    assert "acme" not in result.markdown.lower()
    assert "Case for" not in result.markdown


def test_build_ticket_faq_markdown_keeps_safe_customer_wording_with_account_terms() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Account email",
                "text": "How do I update the account email?",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "source_title": "Account email",
                "text": "How do I update the account email?",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "How do I update the account email?"
    assert result.items[0]["question_source"] == "customer_wording"


@pytest.mark.parametrize(
    "safe_question",
    (
        "How do I download my 1099?",
        "How do I update the 2024 pricing?",
    ),
)
def test_build_ticket_faq_markdown_keeps_non_identifier_numbers_in_customer_wording(
    safe_question: str,
) -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Safe numeric question",
                "text": safe_question,
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "source_title": "Safe numeric question",
                "text": safe_question,
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == safe_question
    assert result.items[0]["question_source"] == "customer_wording"


def test_build_ticket_faq_markdown_skips_unsafe_question_then_uses_safe_customer_wording() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "text": "How do I reset the password for account 4829103?",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "text": "How do I reset the password?",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "How do I reset the password?"
    assert result.items[0]["question_source"] == "customer_wording"
    assert "4829103" not in result.items[0]["question"]


def test_build_ticket_faq_markdown_ignores_clean_source_title_without_safe_vocabulary() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Billing portal outage",
                "text": "The support case is still blocked for our team.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Billing portal outage",
                "text": "The support case is still blocked for our admins.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "What should I do about technical support?"
    assert "billing" not in result.items[0]["question"].lower()
    assert "portal" not in result.items[0]["question"].lower()


def test_build_ticket_faq_markdown_ignores_unlisted_structured_issue_vocabulary_without_documentation_terms() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject A",
                "product": "Support operations",
                "issue": "Device reboot loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject B",
                "product": "Support operations",
                "issue": "Device reboot loop",
                "text": "Device reboot loop still blocks admins.",
                "source_id": "ticket-2",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject C",
                "product": "Analytics",
                "issue": "Export timeout",
                "text": "The analytics export timeout blocks analysts.",
                "source_id": "ticket-3",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject D",
                "product": "Analytics",
                "issue": "Export timeout",
                "text": "Analytics export timeout still blocks managers.",
                "source_id": "ticket-4",
            },
        ],
        max_items=0,
    )

    assert [item["question"] for item in result.items] == [
        "What should I do about technical support?",
        "What should I do about technical support?",
    ]
    assert all(item["question_source"] == "source_policy" for item in result.items)
    assert "customer subject" not in " ".join(item["question"].lower() for item in result.items)
    assert "device reboot" not in " ".join(item["question"].lower() for item in result.items)


def test_build_ticket_faq_markdown_uses_injected_taxonomy_without_documentation_terms() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject A",
                "product": "Support operations",
                "issue": "Device reboot loop",
                "text": "Support operations device reboot loop blocks agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject B",
                "product": "Support operations",
                "issue": "Device reboot loop",
                "text": "The device reboot loop in support operations still blocks admins.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
        representative_taxonomy_terms=("Support operations", "Device reboot loop"),
    )

    assert result.items[0]["question"] == "What should I do about operations device reboot loop?"
    assert result.items[0]["question_source"] == "source_policy"
    assert "customer subject" not in result.items[0]["question"].lower()


def test_build_ticket_faq_markdown_does_not_use_structured_taxonomy_without_injection() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject A",
                "product": "Debt collection",
                "issue": "Communication tactics",
                "text": "Debt collection communication tactics keep blocking customers.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Customer subject B",
                "product": "Debt collection",
                "issue": "Communication tactics",
                "text": "The communication tactics in debt collection still block customers.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
    )

    assert result.items[0]["question"] == "What should I do about technical support?"
    assert "debt collection" not in result.items[0]["question"].lower()
    assert "communication tactics" not in result.items[0]["question"].lower()


def test_build_ticket_faq_markdown_uses_safe_terms_instead_of_pii_gist_tokens() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "Duplicate fee for jane.doe@acme.com account 4829103 remains open.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "Duplicate fee for jane.doe@acme.com account 4829103 remains open.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
        documentation_terms=("Duplicate fee",),
    )

    assert result.items[0]["question"] == "What should I do about duplicate fee?"
    assert "4829103" not in result.items[0]["question"]
    assert "acme" not in result.items[0]["question"].lower()


def test_build_ticket_faq_markdown_uses_repeated_gist_tokens_for_representative_label() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "The export timeout blocks agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "The export timeout blocks admins.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
        documentation_terms=("Export timeout",),
    )

    assert result.items[0]["question"] == "What should I do about export timeout?"


def test_safe_vocabulary_representative_label_rejects_single_occurrence_tokens() -> None:
    assert _safe_vocabulary_representative_label(
        "technical support",
        [
            {
                "text": "Asdf baz blarg blocks agents.",
            },
            {
                "text": "Frobnicate quux zorb blocks admins.",
            },
        ],
        documentation_terms=("Asdf baz", "Frobnicate quux"),
    ) == ""


@pytest.mark.parametrize(
    "unsafe_documentation_term",
    (
        "Reset password for jane.doe@acme.com",
        "Reset password for account 4829103",
    ),
)
def test_build_ticket_faq_markdown_ignores_unsafe_documentation_terms(
    unsafe_documentation_term: str,
) -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "Reset password blocks agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "technical support",
                "text": "Reset password blocks admins.",
                "source_id": "ticket-2",
            },
        ],
        max_items=0,
        documentation_terms=(unsafe_documentation_term,),
    )

    assert result.items[0]["question"] == "What should I do about technical support?"
    assert "reset" not in result.items[0]["question"].lower()
    assert "4829103" not in result.items[0]["question"]
    assert "@" not in result.items[0]["question"]


def test_build_ticket_faq_markdown_labels_topic_degraded_subclusters_without_merging() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device reboot loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device restart loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-2",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Export timeout",
                "text": "The analytics export timeout blocks support agents.",
                "source_id": "ticket-3",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Analytics export timeout",
                "text": "The analytics export timeout blocks support agents.",
                "source_id": "ticket-4",
            },
        ],
        max_items=0,
        documentation_terms=("Device reboot loop", "Analytics export timeout"),
    )

    assert result.as_dict()["generated"] == 2
    assert [
        (item["question"], item["source_ids"], item["ticket_count"])
        for item in result.items
    ] == [
        ("What should I do about device reboot loop?", ("ticket-1", "ticket-2"), 2),
        ("What should I do about analytics export timeout?", ("ticket-3", "ticket-4"), 2),
    ]
    assert {item["question_source"] for item in result.items} == {"source_policy"}
    assert {item["answer_evidence_status"] for item in result.items} == {"draft_needs_review"}


def test_build_ticket_faq_markdown_keeps_identical_topic_degraded_content_together() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device reboot loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-1",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device restart loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-2",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device reboot loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-3",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device restart loop",
                "text": "The device reboot loop blocks support agents.",
                "source_id": "ticket-4",
            },
        ],
        max_items=0,
        documentation_terms=("Device reboot loop",),
    )

    assert result.as_dict()["generated"] == 1
    assert result.items[0]["question"] == "What should I do about device reboot loop?"
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2", "ticket-3", "ticket-4")
    assert result.items[0]["ticket_count"] == 4


def test_question_subclusters_exact_join_merges_threshold_pairs_without_lsh_nomination() -> None:
    shared = tuple(f"shared{index}" for index in range(6))
    cases = [
        shared + tuple(f"lefta{index}" for index in range(6)),
        shared + tuple(f"righta{index}" for index in range(6)),
        shared + tuple(f"leftb{index}" for index in range(5)),
        shared + tuple(f"rightb{index}" for index in range(7)),
        tuple(f"lowshared{index}" for index in range(5))
        + tuple(f"leftc{index}" for index in range(7)),
        tuple(f"lowshared{index}" for index in range(5))
        + tuple(f"rightc{index}" for index in range(7)),
    ]
    rows = [
        {
            "text": " ".join(tokens),
            "source_key": f"ticket-{index}",
        }
        for index, tokens in enumerate(cases)
    ]

    clusters = _question_subclusters(rows)

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("ticket-0", "ticket-1", "ticket-2", "ticket-3"),
        ("ticket-4",),
        ("ticket-5",),
    ]


def test_question_subclusters_exact_join_keeps_empty_gists_separate() -> None:
    clusters = _question_subclusters([
        {"text": "", "source_key": "ticket-empty-1"},
        {"text": "", "source_key": "ticket-empty-2"},
    ])

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("ticket-empty-1",),
        ("ticket-empty-2",),
    ]


def test_embedding_booster_merges_no_overlap_reworded_repeat_in_builder() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "source_title": "Refund",
            "text": "How do I get my money back?",
            "source_id": "refund-a",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "source_title": "Refund",
            "text": "What is the process for a refund?",
            "source_id": "refund-b",
        },
    ]

    without_port = build_ticket_faq_markdown(rows, max_items=0)
    explicit_no_port = build_ticket_faq_markdown(rows, max_items=0, embedding_port=None)
    with_port = build_ticket_faq_markdown(
        rows,
        max_items=0,
        embedding_port=_StubEmbeddingPort({
            "How do I get my money back?": (1.0, 0.0),
            "What is the process for a refund?": (0.99, 0.01),
        }),
    )

    assert explicit_no_port.as_dict() == without_port.as_dict()
    assert without_port.items == ()
    assert without_port.non_repeat_ticket_count == 2
    assert len(with_port.items) == 1
    assert with_port.items[0]["ticket_count"] == 2
    assert with_port.items[0]["source_ids"] == ("refund-a", "refund-b")


def test_embedding_booster_records_only_accepted_semantic_pairs() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "text": "How do I get my money back?",
            "source_id": "refund-a",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "text": "What is the process for a refund?",
            "source_id": "refund-b",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "text": "Where can I update my billing address?",
            "source_id": "billing-address",
        },
    ]
    semantic_merges = []

    result = build_ticket_faq_markdown(
        rows,
        max_items=0,
        embedding_port=_StubEmbeddingPort({
            "How do I get my money back?": (1.0, 0.0),
            "What is the process for a refund?": (0.99, 0.01),
            "Where can I update my billing address?": (0.0, 1.0),
        }),
        embedding_merge_recorder=semantic_merges.append,
    )

    assert len(result.items) == 1
    assert result.items[0]["source_ids"] == ("refund-a", "refund-b")
    assert len(semantic_merges) == 1
    merge = semantic_merges[0]
    assert merge["left_source_id"] == "refund-a"
    assert merge["right_source_id"] == "refund-b"
    assert merge["left_text"] == "How do I get my money back?"
    assert merge["right_text"] == "What is the process for a refund?"
    assert merge["cosine"] > 0.99
    assert merge["left_margin"] > 0.9
    assert merge["right_margin"] > 0.9
    assert merge["token_jaccard"] < (1 / 3)


def test_embedding_booster_records_source_key_when_source_id_is_unknown() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "text": "How do I get my money back?",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "billing help",
            "text": "What is the process for a refund?",
        },
    ]
    semantic_merges = []

    result = build_ticket_faq_markdown(
        rows,
        max_items=0,
        embedding_port=_StubEmbeddingPort({
            "How do I get my money back?": (1.0, 0.0),
            "What is the process for a refund?": (0.99, 0.01),
        }),
        embedding_merge_recorder=semantic_merges.append,
    )

    assert len(result.items) == 1
    assert len(semantic_merges) == 1
    assert semantic_merges[0]["left_source_id"] == "row:1"
    assert semantic_merges[0]["right_source_id"] == "row:2"


def test_embedding_booster_skips_malformed_vectors_without_merging() -> None:
    rows = [
        {
            "text": "How do I get my money back?",
            "source_key": "refund-a",
        },
        {
            "text": "What is the process for a refund?",
            "source_key": "refund-b",
        },
    ]

    clusters = _question_subclusters(
        rows,
        embedding_port=_StubEmbeddingPort({
            "How do I get my money back?": (1.0, 0.0),
            "What is the process for a refund?": (),
        }),
    )

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("refund-a",),
        ("refund-b",),
    ]


@pytest.mark.parametrize(
    "bad_vector",
    [
        (math.nan, 1.0),
        (math.inf, 1.0),
        (-math.inf, 1.0),
    ],
)
def test_embedding_booster_skips_non_finite_vectors_without_merging(
    bad_vector: tuple[float, float],
) -> None:
    rows = [
        {"text": "How can I reset my password?", "source_key": "password"},
        {"text": "Where do I update my billing address?", "source_key": "billing"},
    ]

    clusters = _question_subclusters(
        rows,
        embedding_port=_StubEmbeddingPort({
            "How can I reset my password?": bad_vector,
            "Where do I update my billing address?": (1.0, 1.0),
        }),
    )

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("password",),
        ("billing",),
    ]


def test_embedding_booster_requires_both_mutual_neighbor_margins() -> None:
    rows = [
        {"text": "How do I get my money back?", "source_key": "refund"},
        {"text": "Where do I update my billing address?", "source_key": "billing"},
        {"text": "How do I export account reports?", "source_key": "exports"},
    ]

    clusters = _question_subclusters(
        rows,
        embedding_port=_StubEmbeddingPort({
            "How do I get my money back?": (1.0, 0.0),
            "Where do I update my billing address?": (0.9, 0.4358898943540673),
            "How do I export account reports?": (0.5680837372370346, 0.8229707573703964),
        }),
    )

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("refund",),
        ("billing",),
        ("exports",),
    ]


def test_embedding_booster_ignores_non_singleton_lexical_components() -> None:
    rows = [
        {
            "text": "password reset email link expired login access one",
            "source_key": "hub-a",
        },
        {
            "text": "password reset email link expired login access two",
            "source_key": "hub-b",
        },
        {
            "text": "password reset email link expired login access three",
            "source_key": "hub-c",
        },
        {
            "text": "billing address update payment invoice",
            "source_key": "spoke-billing",
        },
        {
            "text": "dashboard chart loading analytics view",
            "source_key": "spoke-dashboard",
        },
        {
            "text": "export csv report download file",
            "source_key": "spoke-export",
        },
    ]

    clusters = _question_subclusters(
        rows,
        embedding_port=_StubEmbeddingPort({
            rows[0]["text"]: (1.0, 0.0, 0.0),
            rows[1]["text"]: (0.0, 1.0, 0.0),
            rows[2]["text"]: (0.0, 0.0, 1.0),
            rows[3]["text"]: (0.99, 0.01, 0.0),
            rows[4]["text"]: (0.0, 0.99, 0.01),
            rows[5]["text"]: (0.01, 0.0, 0.99),
        }),
    )

    assert [tuple(row["source_key"] for row in cluster) for cluster in clusters] == [
        ("hub-a", "hub-b", "hub-c"),
        ("spoke-billing",),
        ("spoke-dashboard",),
        ("spoke-export",),
    ]


def test_cosine_similarity_rejects_malformed_vectors() -> None:
    assert cosine_similarity((1.0, 0.0), (0.5, 0.5)) == pytest.approx(0.70710678)
    assert cosine_similarity((1.0,), (1.0, 0.0)) is None
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) is None
    assert cosine_similarity("bad", (1.0, 0.0)) is None
    assert cosine_similarity((object(),), (1.0,)) is None
    assert cosine_similarity((math.nan, 1.0), (1.0, 1.0)) is None
    assert cosine_similarity((math.inf, 1.0), (1.0, 1.0)) is None
    assert cosine_similarity((-math.inf, 1.0), (1.0, 1.0)) is None


def test_build_ticket_faq_markdown_subclusters_are_order_insensitive() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "technical support",
            "source_title": "Password reset",
            "text": "password reset email link expired account login access issue",
            "source_id": "ticket-a",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "technical support",
            "source_title": "Password reset",
            "text": "password reset email link invalid account login problem",
            "source_id": "ticket-b",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "technical support",
            "source_title": "Invoice refund",
            "text": "invoice refund credit card charge duplicate billing issue",
            "source_id": "ticket-c",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "technical support",
            "source_title": "Invoice refund",
            "text": "invoice refund credit card charge duplicate payment problem",
            "source_id": "ticket-d",
        },
    ]

    first = build_ticket_faq_markdown(
        rows,
        max_items=0,
        documentation_terms=("Password reset", "Invoice refund"),
    )
    second = build_ticket_faq_markdown(
        tuple(reversed(rows)),
        max_items=0,
        documentation_terms=("Password reset", "Invoice refund"),
    )

    first_clusters = sorted(
        (tuple(sorted(item["source_ids"])), item["ticket_count"])
        for item in first.items
    )
    second_clusters = sorted(
        (tuple(sorted(item["source_ids"])), item["ticket_count"])
        for item in second.items
    )
    assert first_clusters == second_clusters == [
        (("ticket-a", "ticket-b"), 2),
        (("ticket-c", "ticket-d"), 2),
    ]


def test_build_ticket_faq_markdown_preserves_topic_degraded_creation_order_under_cap() -> None:
    labels = [
        "Alpha connector freeze",
        "Bravo invoice failure",
        "Charlie webhook delay",
        "Delta profile lock",
        "Echo export timeout",
        "Foxtrot seat limit",
        "Golf billing retry",
        "Hotel invite bounce",
        "India report blank",
        "Juliet sync pause",
        "Kilo audit stall",
    ]
    opportunities = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "technical support",
            "source_title": label,
            "text": f"{label} blocks the account team.",
            "source_id": f"ticket-{index}-{suffix}",
        }
        for index, label in enumerate(labels)
        for suffix in ("a", "b")
    ]

    result = build_ticket_faq_markdown(
        opportunities,
        max_items=5,
        documentation_terms=tuple(labels),
    )

    assert [item["question"] for item in result.items[:4]] == [
        "What should I do about alpha connector freeze?",
        "What should I do about bravo invoice failure?",
        "What should I do about charlie webhook delay?",
        "What should I do about delta profile lock?",
    ]
    assert result.items[-1]["topic"] == "other support issues"
    assert result.items[-1]["source_ids"] == tuple(
        f"ticket-{index}-{suffix}"
        for index in range(4, 11)
        for suffix in ("a", "b")
    )


def test_build_ticket_faq_markdown_does_not_merge_resolution_backed_same_label() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Device reboot loop",
                "evidence": [{
                    "text": "The device reboot loop blocks support agents.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Open Settings then Firmware, choose version 4.1, "
                        "and restart the device."
                    ),
                }],
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "technical support",
                "source_title": "Export timeout",
                "evidence": [{
                    "text": "The analytics export timeout blocks support agents.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Open Settings then Exports, increase the timeout, "
                        "and retry the job."
                    ),
                }],
            },
        ],
        max_items=0,
        documentation_terms=("Device reboot loop", "Export timeout"),
    )

    assert result.as_dict()["generated"] == 2
    assert [item["question"] for item in result.items] == [
        "What should I do about technical support?",
        "What should I do about technical support?",
    ]
    assert {
        item["answer_evidence_status"] for item in result.items
    } == {"resolution_evidence"}
    assert all(item["source_ids"] in {("ticket-1",), ("ticket-2",)} for item in result.items)


def test_build_ticket_faq_markdown_uses_source_policy_when_customer_question_is_too_long() -> None:
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Profile update",
                "evidence": [{
                    "text": long_question,
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert len(long_question) > 140
    assert result.items[0]["topic"] == "email and profile updates"
    assert result.items[0]["question"] == "What should I do about email and profile updates?"
    assert result.items[0]["question_source"] == "source_policy"


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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my account password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my password from the login page?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my account password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "Customer: Login is broken again. Agent: Can you share a screenshot?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert result.items[0]["question"] == "What should I do about login reset?"
    assert result.items[0]["question_source"] == "source_policy"


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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my account password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Password reset",
                "evidence": [{
                    "text": "How do I reset my account password?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Help article",
                "evidence": [{
                    "text": "I opened https://example.com/help?article=123 and the page is still blank.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert result.items[0]["question"] == "What should I do about help article?"
    assert result.items[0]["question_source"] == "source_policy"


@pytest.mark.parametrize(
    "text,source_title,expected_question",
    [
        (
            "I paid the XX/XX/2019 credit card installment of {$100 and the balance is still wrong.",
            "Credit card or prepaid card - Getting a credit card",
            "What should I do if my card application, offer, or activation does not look right?",
        ),
        (
            "I am not \" allowed '' to speak to a human.",
            "Customer support issues",
            "What should I do about customer support issues?",
        ),
        (
            "I received a letter dated XX/XX/XXXX, signed by Mr.",
            "Customer support issues",
            "What should I do about customer support issues?",
        ),
    ],
)
def test_build_ticket_faq_markdown_rejects_malformed_redacted_customer_questions(
    text: str,
    source_title: str,
    expected_question: str,
) -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Credit card complaint",
                "evidence": [{
                    "text": text,
                    "source_id": "cfpb:1",
                    "source_type": "support_ticket",
                    "source_title": source_title,
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Credit card complaint",
                "evidence": [{
                    "text": text,
                    "source_id": "cfpb:2",
                    "source_type": "support_ticket",
                    "source_title": source_title,
                }],
            },
        ]
    )

    assert result.items[0]["question"] == expected_question
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks["uses_user_vocabulary"] is True


def test_build_ticket_faq_markdown_accepts_host_intent_rules() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Data sync is behind",
                "evidence": [{
                    "text": "The CRM warehouse sync is delayed every morning by connector lag.",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Connector lag",
                "evidence": [{
                    "text": "The CRM connector lag keeps the warehouse sync delayed every morning.",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        intent_rules=(("data freshness", ("warehouse sync", "connector lag")),),
    )

    assert [item["topic"] for item in result.items] == ["data freshness"]
    assert result.items[0]["evidence_count"] == 2


def test_normalize_intent_rules_accepts_line_and_object_shapes() -> None:
    assert normalize_intent_rules(
        [
            "data freshness=warehouse sync,connector lag",
            {"topic": "access setup", "keywords": ["invite link", "new user"]},
        ],
        label="faq_intent_rules",
    ) == (
        ("data freshness", ("warehouse sync", "connector lag")),
        ("access setup", ("invite link", "new user")),
    )


def test_normalize_intent_rules_is_idempotent() -> None:
    rules = normalize_intent_rules(
        [
            "data freshness=warehouse sync,connector lag",
            {"topic": "access setup", "keywords": ["invite link", "new user"]},
        ],
        label="faq_intent_rules",
    )

    assert normalize_intent_rules(rules, label="intent_rules") == rules


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        ("bad", "faq_intent_rules must be an array"),
        (["bad"], "faq_intent_rules[1] must use topic=keyword,keyword"),
        ([{"topic": "data freshness", "keywords": []}], "faq_intent_rules[1]"),
        ([{"topic": "data freshness", "keywords": "sync"}], "faq_intent_rules[1]"),
        ([[None, ["sync"]]], "faq_intent_rules[1]"),
    ],
)
def test_normalize_intent_rules_rejects_invalid_shapes(
    rules: object,
    message: str,
) -> None:
    with pytest.raises(ValueError) as exc:
        normalize_intent_rules(rules, label="faq_intent_rules")  # type: ignore[arg-type]
    assert message in str(exc.value)


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
    assert item["answer"] == (
        "No verified resolution evidence was found in 4 ticket sources; keep "
        "this FAQ in review before answering: What should I do about data freshness?"
    )
    assert "ticket-3" not in result.markdown


def test_build_ticket_faq_markdown_condenses_tail_groups_when_item_cap_is_lower_than_topics() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "source_title": "Credit report dispute",
            "evidence": [{
                "text": "My credit report has the wrong balance on my account.",
                "source_id": f"credit-{index}",
                "source_type": "support_ticket",
            }],
        }
        for index in range(1, 4)
    ]
    rows.extend([
        {
            "source_type": "support_ticket",
            "source_title": "Mortgage issue",
            "evidence": [{
                "text": "My mortgage servicer will not explain the payoff quote.",
                "source_id": "mortgage-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Mortgage issue",
            "evidence": [{
                "text": "My mortgage servicer still will not explain the payoff quote.",
                "source_id": "mortgage-2",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Debt collection issue",
            "evidence": [{
                "text": "A debt collector is asking me to pay a debt I do not recognize.",
                "source_id": "debt-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Debt collection issue",
            "evidence": [{
                "text": "A debt collector keeps asking me to pay a debt I do not recognize.",
                "source_id": "debt-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    result = build_ticket_faq_markdown(rows, max_items=2)

    assert [item["topic"] for item in result.items] == [
        "credit report disputes",
        "other support issues",
    ]
    assert result.items[1]["source_ids"] == ("debt-1", "debt-2", "mortgage-1", "mortgage-2")
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }


def test_build_ticket_faq_markdown_preserves_top_group_when_single_item_cap_overflows() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": f"Credit report dispute {index}",
                "evidence": [{
                    "text": "My credit report has the wrong balance on my account.",
                    "source_id": f"credit-{index}",
                    "source_type": "support_ticket",
                }],
            }
            for index in range(1, 4)
        ] + [
            {
                "source_type": "support_ticket",
                "source_title": "Debt collection issue",
                "evidence": [{
                    "text": "A collector is asking me to pay a debt I do not recognize.",
                    "source_id": "debt-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Debt collection issue",
                "evidence": [{
                    "text": "A collector keeps asking me to pay a debt I do not recognize.",
                    "source_id": "debt-2",
                    "source_type": "support_ticket",
                }],
            },
        ],
        max_items=1,
    )

    assert len(result.items) == 1
    assert result.items[0]["topic"] == "credit report disputes"
    assert result.items[0]["source_ids"] == ("credit-1", "credit-2", "credit-3", "debt-1", "debt-2")
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }


def test_build_ticket_faq_markdown_handles_1000_cfpb_style_rows_without_archive() -> None:
    rows = []
    next_id = 1

    def add_rows(count: int, *, source_title: str, text: str, pain_point: str | None = None) -> None:
        nonlocal next_id
        for index in range(count):
            source_id = f"cfpb:{next_id}"
            next_id += 1
            row = {
                "source_type": "support_ticket",
                "source_title": source_title,
                "evidence": [{
                    "text": text.format(index=index + 1),
                    "source_id": source_id,
                    "source_type": "support_ticket",
                    "source_title": source_title,
                }],
            }
            if pain_point:
                row["pain_points"] = [pain_point]
            rows.append(row)

    add_rows(
        600,
        source_title="Credit reporting, credit repair services, or other personal consumer reports - Incorrect information on your report",
        text="My credit report has incorrect information on one of my accounts.",
    )
    add_rows(
        150,
        source_title="Debt collection - Attempts to collect debt not owed",
        text="A debt collector says I owe a debt I do not recognize.",
    )
    add_rows(
        100,
        source_title="Credit card or prepaid card - Fees or interest",
        text="I paid the XX/XX/2019 credit card installment of {{$100.00}} but it was not credited.",
    )
    add_rows(
        50,
        source_title="Mortgage - Trouble during payment process",
        text="My mortgage servicer will not explain the payment issue on my loan.",
    )
    add_rows(
        30,
        source_title="Checking or savings account - Opening an account",
        text="I was trying to open an account and the bank declined me because of early warning services.",
    )
    add_rows(
        20,
        source_title="Checking or savings account - Closing an account",
        text="I need to close an account and recover the remaining balance.",
    )
    add_rows(
        15,
        source_title="Credit card or prepaid card - Getting a credit card",
        text="I applied for a credit card and the issuer could not confirm my identity.",
    )
    add_rows(
        10,
        source_title="Credit card or prepaid card - Advertising",
        text="The advertising offer for this prepaid card seems wrong.",
    )
    add_rows(
        8,
        source_title="Checking or savings account - Managing an account",
        text="The bank website malfunction blocked my account activity.",
        pain_point="Managing an account",
    )
    add_rows(
        7,
        source_title="Checking or savings account - Customer service",
        text="I am not \" allowed '' to speak to a human.",
        pain_point="Customer service",
    )
    add_rows(
        5,
        source_title="Money transfer, virtual currency, or money service - Other transaction problem",
        text="A transaction was scheduled incorrectly and the company will not explain it.",
    )
    add_rows(
        3,
        source_title="Money transfer, virtual currency, or money service - Other service problem",
        text="I received an email with another customer's information.",
    )
    add_rows(
        2,
        source_title="Money transfer, virtual currency, or money service - Wire transfer problem",
        text="The transfer was delayed and no one explained the status.",
        pain_point="Wire transfer problem",
    )

    assert len(rows) == 1000

    result = build_ticket_faq_markdown(
        rows,
        max_items=12,
        max_evidence_per_item=5,
    )

    questions = [item["question"] for item in result.items]
    opening = next(item for item in result.items if item["topic"] == "opening an account")

    assert result.ticket_source_count == 1000
    assert len(result.items) == 12
    assert sum(item["ticket_count"] for item in result.items) == 1000
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert all(item["question_source"] != "topic_fallback" for item in result.items)
    assert all("XX/XX/2019" not in question for question in questions)
    assert all("allowed ''" not in question for question in questions)
    assert opening["steps"][0].startswith("Review the cited ticket evidence")
    assert opening["answer_evidence_status"] == "draft_needs_review"
    assert "Export or Download" not in " ".join(opening["steps"])
    assert "export is missing" not in opening["when_to_contact_support"]


def test_build_ticket_faq_markdown_uses_financial_contact_guidance_for_cfpb_account_topics() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Checking or savings account - Opening an account",
                "evidence": [{
                    "text": "I was trying to open an account and the bank declined me because of early warning services.",
                    "source_id": "cfpb:1",
                    "source_type": "support_ticket",
                    "source_title": "Checking or savings account - Opening an account",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Checking or savings account - Opening an account",
                "evidence": [{
                    "text": "I was trying to open an account and the bank also declined me because of early warning services.",
                    "source_id": "cfpb:2",
                    "source_type": "support_ticket",
                    "source_title": "Checking or savings account - Opening an account",
                }],
            },
        ]
    )

    assert result.items[0]["topic"] == "opening an account"
    assert result.items[0]["steps"][0].startswith("Review the cited ticket evidence")
    assert result.items[0]["answer_evidence_status"] == "draft_needs_review"
    assert "Export or Download" not in result.markdown
    assert "export is missing" not in result.markdown


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
            },
            {
                "source_type": "support_ticket",
                "source_title": "Profile update",
                "evidence": [{
                    "text": "How do we change our email before renewal?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert [item["topic"] for item in result.items] == ["email and profile updates"]


def test_build_ticket_faq_markdown_escapes_pipe_once_in_article_sections() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Export format",
                "evidence": [{
                    "text": "How do I export the A | B report?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Export format",
                "evidence": [{
                    "text": "How do we export the A | B report?",
                    "source_id": "ticket-2",
                    "source_type": "support_ticket",
                }],
            },
        ]
    )

    assert "A \\| B report" in result.markdown
    assert "A \\\\| B report" not in result.markdown


def test_ticket_faq_markdown_renders_action_and_source_lists_from_packaged_rows() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities)
    rendered = _RenderedFAQHTML()
    rendered.feed(markdown.markdown(result.markdown))

    assert rendered.h1 == ["Customer Ticket FAQ"]
    assert rendered.h2 == [
        "1. How do we export the campaign reporting dashboard before renewal?",
        "2. How do I change my login email?",
    ]
    assert rendered.strong.count("What to do next:") == 2
    assert rendered.strong.count("When to contact support:") == 2
    assert rendered.strong.count("Sources:") == 2
    assert rendered.ol_count == 2
    assert rendered.ul_count == 2
    assert any(
        "Customers are asking about email and profile updates across 2 ticket sources."
        in paragraph
        for paragraph in rendered.paragraphs
    )
    assert any("the field is locked" in paragraph for paragraph in rendered.paragraphs)
    assert any(
        "Draft the customer-facing steps from a verified help article"
        in item
        for item in rendered.list_items
    )
    assert any(
        "ticket-acme-1 - Change login email" in item
        for item in rendered.list_items
    )
    assert len(result.items) == 2
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
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
                "createdAt": "2026-05-02T12:00:00Z",
                "pain_points": ["login"],
                "evidence": [{
                    "text": "How do we update our login email?",
                    "source_id": "ticket-new-2",
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

    assert result.ticket_source_count == 2
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
                },
                {
                    "ticket_id": "ticket-2",
                    "subject": "Login email change",
                    "source_type": "ticket",
                    "message": "How do we change our email address?",
                    "pain_category": "login",
                },
            ]
        },
        max_items=2,
    )

    assert result.as_dict()["generated"] == 1
    assert "How do I change my email address?" in result.markdown
    assert "`ticket-1` - Login email change" in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_config_threads_embedding_port() -> None:
    service = TicketFAQMarkdownService(
        TicketFAQMarkdownConfig(
            embedding_port=_StubEmbeddingPort({
                "How do I get my money back?": (1.0, 0.0),
                "What is the process for a refund?": (0.99, 0.01),
            })
        )
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [
                {
                    "ticket_id": "refund-a",
                    "subject": "Refund request",
                    "source_type": "ticket",
                    "message": "How do I get my money back?",
                    "pain_category": "billing",
                },
                {
                    "ticket_id": "refund-b",
                    "subject": "Refund request",
                    "source_type": "ticket",
                    "message": "What is the process for a refund?",
                    "pain_category": "billing",
                },
            ]
        },
        max_items=0,
    )

    assert len(result.items) == 1
    assert result.items[0]["source_ids"] == ("refund-a", "refund-b")


@pytest.mark.asyncio
async def test_ticket_faq_service_groups_package_cluster_hints_for_raw_rows() -> None:
    package = build_support_ticket_input_package([
        {
            "ticket_id": "zd-1",
            "subject": "Password reset help",
            "description": "<p>How do I reset my password from the login screen?</p>",
        },
        {
            "ticket_id": "zd-2",
            "subject": "Password reset not working",
            "description": "I cannot reset my password from the login screen",
        },
        {
            "ticket_id": "hs-1",
            "subject": "Change email address",
            "description": "Where do I update my email?",
        },
        {
            "ticket_id": "hs-2",
            "subject": "Update account email",
            "description": "Need to change email address",
        },
    ])
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material=package.inputs["source_material"],
        max_items=10,
    )

    assert [(item["topic"], item["ticket_count"]) for item in result.items] == [
        ("login password reset", 2),
        ("email update", 2),
    ]
    assert all(
        item["answer_evidence_status"] == "draft_needs_review"
        for item in result.items
    )
    assert all(item["answer"].strip() for item in result.items)
    assert "<p>" not in result.markdown
    assert "`zd-1` - Password reset help" in result.markdown
    assert "`hs-2` - Update account email" in result.markdown


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
                    "message": "A new user needs another invite link after the invite link expired.",
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
        source_material=[
            {
                "source_id": "review-1",
                "source_type": "review",
                "text": "Export settings are hard to find.",
                "pain_category": "exports",
            },
            {
                "source_id": "review-2",
                "source_type": "review",
                "text": "The export settings are hard to find.",
                "pain_category": "exports",
            },
        ],
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
            {
                "ticket_id": "ticket-3",
                "source_type": "ticket",
                "subject": "Export issue",
                "message": "The export still keeps timing out.",
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
            "rows": [
                {
                    "ticket_id": "ticket-1",
                    "source_type": "ticket",
                    "message": "The dashboard export is not working.",
                    "pain_category": "exports",
                },
                {
                    "ticket_id": "ticket-2",
                    "source_type": "ticket",
                    "message": "The dashboard export is still not working.",
                    "pain_category": "exports",
                },
            ],
        },
    )

    assert result.as_dict()["generated"] == 1
    assert "The dashboard export is not working." in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_accepts_search_log_source_material() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "search_logs": [
                {
                    "query_id": "search-1",
                    "search_query": "How do I export attribution report?",
                    "results_count": 0,
                    "zero_results": True,
                },
                {
                    "query_id": "search-2",
                    "search_query": "export attribution reports",
                    "results_count": 2,
                },
            ]
        },
    )

    assert result.as_dict()["generated"] == 1
    assert result.ticket_source_count == 2
    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["source_ids"] == ("search-1", "search-2")
    assert result.items[0]["frequency"] == 2
    assert "`search-1`" in result.markdown
    assert "How do I export attribution report?" in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_accepts_chat_transcript_source_material() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "chats": [
                {
                    "chat_id": "chat-1",
                    "subject": "Attribution export",
                    "messages": [
                        {
                            "speaker": "customer",
                            "message": "How do I export the attribution dashboard?",
                        },
                        {
                            "speaker": "agent",
                            "message": "I can send the export steps.",
                        },
                    ],
                },
                {
                    "chat_id": "chat-2",
                    "subject": "Attribution export",
                    "messages": [
                        {
                            "speaker": "customer",
                            "message": "How do we export the attribution dashboard?",
                        },
                        {
                            "speaker": "agent",
                            "message": "I can send the export steps.",
                        },
                    ],
                },
            ]
        },
    )

    assert result.as_dict()["generated"] == 1
    assert result.ticket_source_count == 2
    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["source_ids"] == ("chat-1", "chat-2")
    assert result.items[0]["question"] == "How do I export the attribution dashboard?"
    assert "`chat-1` - Attribution export" in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_accepts_sales_objection_source_material() -> None:
    service = TicketFAQMarkdownService()

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "sales_objections": [
                {
                    "objection_id": "obj-1",
                    "objection_text": "We cannot export attribution reports before renewal.",
                },
                {
                    "objection_id": "obj-2",
                    "objection_text": "We cannot export attribution reports before the renewal.",
                },
            ]
        },
    )

    assert result.as_dict()["generated"] == 1
    assert result.ticket_source_count == 2
    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["source_ids"] == ("obj-1", "obj-2")
    assert result.items[0]["question"] == "How do we export attribution reports before renewal?"
    assert "`obj-1`" in result.markdown


@pytest.mark.asyncio
async def test_ticket_faq_service_accepts_documentation_terms() -> None:
    service = TicketFAQMarkdownService(
        TicketFAQMarkdownConfig(documentation_terms=("Download report",))
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "message": "How do I export attribution data?",
                    "pain_category": "exports",
                },
                {
                    "ticket_id": "ticket-2",
                    "message": "How do we export attribution data?",
                    "pain_category": "exports",
                },
            ]
        },
    )

    assert result.items[0]["term_mappings"][0]["customer_term"] == "export"
    assert result.items[0]["term_mappings"][0]["documentation_term"] == "Download report"


@pytest.mark.asyncio
async def test_ticket_faq_service_accepts_custom_vocabulary_gap_rules() -> None:
    repository = _FAQRepository()
    service = TicketFAQMarkdownService(
        TicketFAQMarkdownConfig(
            documentation_terms=("Single sign-on setup",),
            vocabulary_gap_rules=(("SSO", "single sign-on"),),
        ),
        ticket_faqs=repository,
    )

    result = await service.generate(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        source_material={
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "message": "How do I enable SSO for my team?",
                    "pain_category": "authentication",
                },
                {
                    "ticket_id": "ticket-2",
                    "message": "How do we enable SSO for our team?",
                    "pain_category": "authentication",
                },
            ]
        },
    )

    assert result.items[0]["term_mappings"][0]["customer_term"] == "SSO"
    assert result.items[0]["term_mappings"][0]["documentation_term"] == "Single sign-on setup"
    assert repository.saved[0]["drafts"][0].metadata["vocabulary_gap_rules"] == [
        ["SSO", "single sign-on"]
    ]


@pytest.mark.asyncio
async def test_ticket_faq_service_saves_generated_markdown_when_repository_configured() -> None:
    repository = _FAQRepository()
    service = TicketFAQMarkdownService(ticket_faqs=repository)

    result = await service.generate(
        scope=TenantScope(account_id="acct-1", user_id="user-1"),
        target_mode="vendor_retention",
        source_material=[
            {
                "ticket_id": "ticket-1",
                "source_type": "ticket",
                "created_at": "2026-05-01",
                "message": "The attribution export is missing renewals.",
                "pain_category": "exports",
            },
            {
                "ticket_id": "ticket-2",
                "source_type": "ticket",
                "created_at": "2026-05-02",
                "message": "The attribution export is missing renewals.",
                "pain_category": "exports",
            },
        ],
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
        "chat",
        "chat_transcript",
        "conversation",
        "transcript",
        "sales_call",
        "meeting",
        "sales_objection",
        "objection",
        "complaint",
        "search_log",
        "search_query",
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
        "resolution_evidence_scoped": False,
    }
    with pytest.raises(ValueError, match="max_items must be positive or 0 for unlimited"):
        build_ticket_faq_markdown([], max_items=-1)
    with pytest.raises(ValueError, match="max_evidence_per_item must be positive"):
        build_ticket_faq_markdown([], max_evidence_per_item=0)


def test_build_ticket_faq_markdown_accepts_ticket_source_type_alias() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "ticket",
            "pain_points": ["billing"],
            "evidence": [{"text": "I need help with billing.", "source_id": "ticket-1", "source_type": "ticket"}],
        },
        {
            "source_type": "ticket",
            "pain_points": ["billing"],
            "evidence": [{"text": "I need help with billing.", "source_id": "ticket-2", "source_type": "ticket"}],
        },
    ])

    assert result.ticket_source_count == 2
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2")
    assert "I need help with billing." in result.markdown


def test_build_ticket_faq_markdown_uses_financial_support_guidance_for_cfpb_shaped_rows() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Credit card or prepaid card - Fees or interest",
                "pain_points": ["Fees or interest"],
                "evidence": [{
                    "text": (
                        "I logged into my account and saw a foreign transaction "
                        "fee that should not have been charged."
                    ),
                    "source_id": "cfpb:3559709",
                    "source_type": "support_ticket",
                    "source_title": "Credit card or prepaid card - Fees or interest",
                }],
            },
            {
                "source_type": "support_ticket",
                "source_title": "Credit card or prepaid card - Fees or interest",
                "pain_points": ["Fees or interest"],
                "evidence": [{
                    "text": (
                        "I logged into my account and saw another foreign "
                        "transaction fee that should not have been charged."
                    ),
                    "source_id": "cfpb:3205066",
                    "source_type": "support_ticket",
                    "source_title": "Credit card or prepaid card - Fees or interest",
                }],
            },
        ],
        support_contact="https://www.consumerfinance.gov/complaint/",
    )

    markdown = result.markdown
    assert "Review the cited ticket evidence and confirm the policy-approved answer" in markdown
    assert "Draft the customer-facing steps from a verified help article" in markdown
    assert "charge, fee, payment, balance, or dispute still looks wrong" in markdown
    assert "Open your profile, account settings, or login settings" not in markdown
    assert "https://www.consumerfinance.gov/complaint/" in markdown


def test_build_ticket_faq_markdown_uses_debt_collection_guidance_before_account_guidance() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "complaint",
                "pain_points": ["Attempts to collect debt not owed"],
                "evidence": [{
                    "text": "I received a collection letter for a debt I do not owe.",
                    "source_id": "cfpb:3182593",
                    "source_type": "complaint",
                }],
            },
            {
                "source_type": "complaint",
                "pain_points": ["Attempts to collect debt not owed"],
                "evidence": [{
                    "text": "I received another collection letter for a debt I do not owe.",
                    "source_id": "cfpb:3182594",
                    "source_type": "complaint",
                }],
            },
        ],
        support_contact="https://example.com/support",
    )

    markdown = result.markdown
    assert result.items[0]["topic"] == "debt collection disputes"
    assert "collector will not validate the debt" in markdown
    assert "Draft the customer-facing steps from a verified help article" in markdown
    assert "Open your profile, account settings, or login settings" not in markdown
    assert "Open the reporting or analytics area" not in markdown


def test_build_ticket_faq_markdown_uses_credit_report_guidance_before_reporting_guidance() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "complaint",
                "pain_points": ["Incorrect information on your report"],
                "evidence": [{
                    "text": "There are many mistakes appearing in my credit report.",
                    "source_id": "cfpb:3187954",
                    "source_type": "complaint",
                }],
            },
            {
                "source_type": "complaint",
                "pain_points": ["Incorrect information on your report"],
                "evidence": [{
                    "text": "There are many mistakes still appearing in my credit report.",
                    "source_id": "cfpb:3187955",
                    "source_type": "complaint",
                }],
            },
        ],
        support_contact="https://example.com/support",
    )

    markdown = result.markdown
    assert result.items[0]["topic"] == "credit report disputes"
    assert "incorrect after you dispute it" in markdown
    assert "Draft the customer-facing steps from a verified help article" in markdown
    assert "Open the reporting or analytics area" not in markdown


def test_build_ticket_faq_markdown_replaces_vague_questions_with_source_policy_question() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "pain_points": ["Incorrect information on your report"],
            "evidence": [{
                "text": "Need help? There are mistakes appearing in my credit report.",
                "source_id": "cfpb:3187954",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "pain_points": ["Incorrect information on your report"],
            "evidence": [{
                "text": "Need help? There are mistakes still appearing in my credit report.",
                "source_id": "cfpb:3187955",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["question"] == "What should I do if information on my credit report is wrong?"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "## 1. Need help?" not in result.markdown


def test_build_ticket_faq_markdown_uses_substantive_question_after_vague_opener() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "pain_points": ["login reset"],
            "evidence": [{
                "text": "Need help? I cannot reset my password.",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "pain_points": ["login reset"],
            "evidence": [{
                "text": "I cannot reset my password.",
                "source_id": "ticket-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "## 1. Need help?" not in result.markdown


def test_build_ticket_faq_markdown_does_not_classify_generic_investigation_as_credit_report() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "pain_points": ["reporting friction"],
            "evidence": [{
                "text": "Need help? I cannot export the investigation dashboard report.",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "pain_points": ["reporting friction"],
            "evidence": [{
                "text": "I cannot export the investigation dashboard report.",
                "source_id": "ticket-2",
                "source_type": "support_ticket",
            }],
        },
    ])

    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["question"] == "How do I export the investigation dashboard report?"
    assert "Draft the customer-facing steps from a verified help article" in result.markdown
    assert "credit bureau" not in result.markdown


def test_build_ticket_faq_markdown_does_not_treat_cfpb_report_as_saas_reporting() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "pain_points": ["Opening an account"],
            "evidence": [{
                "text": (
                    "I was trying to open an account and the bank declined me after I sent "
                    "my identity theft report."
                ),
                "source_id": "cfpb:3173042",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "pain_points": ["Opening an account"],
            "evidence": [{
                "text": (
                    "I was trying to open an account and the bank again declined me after I sent "
                    "my identity theft report."
                ),
                "source_id": "cfpb:3173043",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["topic"] == "opening an account"
    assert result.items[0]["question_source"] == "customer_wording"
    assert "Open the reporting or analytics area" not in result.markdown
    assert "Draft the customer-facing steps from a verified help article" in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_credit_report_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "credit_reporting.csv",
        [
            {
                "Complaint ID": "3181474",
                "Product": "Credit reporting, credit repair services, or other personal consumer reports",
                "Issue": "Improper use of your report",
                "Consumer complaint narrative": (
                    "The inquiries are a result of identity theft and should not remain on my report."
                ),
                "Company": "Example Bureau",
            },
            {
                "Complaint ID": "3181475",
                "Product": "Credit reporting, credit repair services, or other personal consumer reports",
                "Issue": "Improper use of your report",
                "Consumer complaint narrative": (
                    "These inquiries are a result of identity theft and should not remain on my report."
                ),
                "Company": "Example Bureau",
            },
        ],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "credit report disputes"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert "Draft the customer-facing steps from a verified help article" in result.markdown
    assert "Open the reporting or analytics area" not in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_debt_collection_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "debt_collection.csv",
        [
            {
                "Complaint ID": "3177182",
                "Product": "Debt collection",
                "Issue": "Communication tactics",
                "Consumer complaint narrative": (
                    "The collector keeps calling and asking me to confirm my email address for a debt I do not owe."
                ),
                "Company": "Example Collector",
            },
            {
                "Complaint ID": "3177183",
                "Product": "Debt collection",
                "Issue": "Communication tactics",
                "Consumer complaint narrative": (
                    "The collector keeps calling and asking me to confirm my email address for a debt I do not owe at all."
                ),
                "Company": "Example Collector",
            },
        ],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "debt collection disputes"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "Draft the customer-facing steps from a verified help article" in result.markdown
    assert "Open your profile, account settings, or login settings" not in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_mortgage_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "mortgage.csv",
        [
            {
                "Complaint ID": "3178554",
                "Product": "Mortgage",
                "Issue": "Struggling to pay mortgage",
                "Consumer complaint narrative": (
                    "The servicer posted a foreclosure notice even though the modification documents were submitted."
                ),
                "Company": "Example Servicer",
            },
            {
                "Complaint ID": "3178555",
                "Product": "Mortgage",
                "Issue": "Struggling to pay mortgage",
                "Consumer complaint narrative": (
                    "The servicer posted another foreclosure notice even though the modification documents were submitted."
                ),
                "Company": "Example Servicer",
            },
        ],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "mortgage servicing issues"
    assert result.items[0]["question"] == (
        "What should I do if my mortgage servicer will not fix a payment, payoff, foreclosure, or modification issue?"
    )
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
        "resolution_evidence_scoped": True,
    }
    assert "Draft the customer-facing steps from a verified help article" in result.markdown
    assert "Open the bill, statement, payment history, or dispute record" not in result.markdown


def test_build_ticket_faq_markdown_does_not_treat_generic_loan_modification_as_mortgage(
    tmp_path: Path,
) -> None:
    source = _write_source_csv(
        tmp_path,
        "vehicle_loan.csv",
        [
            {
                "Complaint ID": "vehicle-1",
                "Product": "Vehicle loan or lease",
                "Issue": "Managing the loan or lease",
                "Consumer complaint narrative": (
                    "I need help with a loan modification and payment dispute on my auto loan."
                ),
                "Company": "Example Auto Lender",
            },
            {
                "Complaint ID": "vehicle-2",
                "Product": "Vehicle loan or lease",
                "Issue": "Managing the loan or lease",
                "Consumer complaint narrative": (
                    "I still need help with a loan modification and payment dispute on my auto loan."
                ),
                "Company": "Example Auto Lender",
            },
        ],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "billing and payments"
    assert "mortgage servicer" not in result.markdown
    assert "Gather the mortgage statement" not in result.markdown
    assert "Draft the customer-facing steps from a verified help article" in result.markdown


def test_build_ticket_faq_markdown_rejects_complaint_process_boilerplate_question() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "Product": "Mortgage",
            "Issue": "Struggling to pay mortgage",
            "evidence": [{
                "text": (
                    "My husband and I have submitted several complaints through the CFPB. "
                    "The servicer still has not reviewed the modification documents."
                ),
                "source_id": "cfpb:3178270",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Mortgage",
            "Issue": "Struggling to pay mortgage",
            "evidence": [{
                "text": (
                    "My husband and I have submitted several complaints through the CFPB. "
                    "The servicer has still not reviewed the modification documents."
                ),
                "source_id": "cfpb:3178271",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["topic"] == "mortgage servicing issues"
    assert result.items[0]["question"] == (
        "What should I do if my mortgage servicer will not fix a payment, payoff, foreclosure, or modification issue?"
    )
    assert result.items[0]["question_source"] == "source_policy"
    assert "## 1. What should I do if my mortgage servicer" in result.markdown
    assert "## 1. What should I do if my husband" not in result.markdown


def test_build_ticket_faq_markdown_rejects_all_caps_customer_question() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Advertising",
            "evidence": [{
                "text": (
                    "HOW WOULD I HAVE KNOWN? The promotional payment option was not "
                    "available on the statement."
                ),
                "source_id": "cfpb:3173636",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Advertising",
            "evidence": [{
                "text": (
                    "HOW WOULD I HAVE KNOWN? The promotional payment option was not "
                    "available on my statement."
                ),
                "source_id": "cfpb:3173637",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["topic"] == "advertising"
    assert result.items[0]["question"] == (
        "What should I do if a financial product advertisement or offer seems wrong?"
    )
    assert result.items[0]["question_source"] == "source_policy"
    assert "## 1. HOW WOULD I HAVE KNOWN?" not in result.markdown


def test_build_ticket_faq_markdown_rejects_complaint_about_as_customer_question() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "Product": "Checking or savings account",
            "Issue": "Opening an account",
            "evidence": [{
                "text": (
                    "My complaint is about Capital One opening the Money Market Account "
                    "and getting the Bonus."
                ),
                "source_id": "cfpb:3172831",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Checking or savings account",
            "Issue": "Opening an account",
            "evidence": [{
                "text": (
                    "My complaint is about Capital One opening the Money Market Account "
                    "and not getting the Bonus."
                ),
                "source_id": "cfpb:3172832",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["topic"] == "opening an account"
    assert result.items[0]["question"] == (
        "What should I do if a bank will not open an account or says my account was opened incorrectly?"
    )
    assert result.items[0]["question_source"] == "source_policy"
    assert "Capital One opening the Money Market Account" not in result.markdown.splitlines()[2]


def test_build_ticket_faq_markdown_summarizes_customer_question_source_row() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Problem with a purchase shown on your statement",
            "evidence": [{
                "text": "There is no proof that I received the package shown on my statement.",
                "source_id": "cfpb:3584679",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Problem with a purchase shown on your statement",
            "evidence": [{
                "text": (
                    "How am I being held responsible for a package for which there "
                    "is no proof that I received?"
                ),
                "source_id": "cfpb:3442136",
                "source_type": "complaint",
            }],
        },
    ])

    assert result.items[0]["question"] == (
        "How am I being held responsible for a package for which there is no proof that I received?"
    )
    assert result.items[0]["question_source"] == "customer_wording"
    assert "The clearest customer wording is \"How am I being held responsible" in result.items[0]["summary"]
    assert "The clearest customer wording is \"They call at all hours" not in result.items[0]["summary"]


def test_build_ticket_faq_markdown_separates_contact_complaints_from_billing_disputes() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "complaint",
            "Product": "Student loan",
            "Issue": "Dealing with your lender or servicer",
            "evidence": [{
                "text": "They call at all hours and on the weekends using various numbers.",
                "source_id": "cfpb:3584679",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Student loan",
            "Issue": "Dealing with your lender or servicer",
            "evidence": [{
                "text": "They call at all hours and on weekends using various numbers.",
                "source_id": "cfpb:3584680",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Problem with a purchase shown on your statement",
            "evidence": [{
                "text": (
                    "How am I being held responsible for a package for which there "
                    "is no proof that I received?"
                ),
                "source_id": "cfpb:3442136",
                "source_type": "complaint",
            }],
        },
        {
            "source_type": "complaint",
            "Product": "Credit card or prepaid card",
            "Issue": "Problem with a purchase shown on your statement",
            "evidence": [{
                "text": "There is no proof that I received the package shown on my statement.",
                "source_id": "cfpb:3442137",
                "source_type": "complaint",
            }],
        },
    ])

    assert [item["topic"] for item in result.items] == [
        "billing and payments",
        "communication and contact issues",
    ]
    assert result.items[0]["source_ids"] == ("cfpb:3442136", "cfpb:3442137")
    assert result.items[1]["source_ids"] == ("cfpb:3584679", "cfpb:3584680")
    assert "Review the cited ticket evidence" in result.items[1]["steps"][0]
    assert result.items[1]["answer_evidence_status"] == "draft_needs_review"
    assert "keeps contacting you" in result.items[1]["when_to_contact_support"]


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
    assert "across 2 ticket sources" in result.markdown


def test_build_ticket_faq_markdown_counts_distinct_source_ids_per_item() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "pain_points": ["exports"],
            "evidence": [
                {"text": "Export failed on Monday.", "source_id": "ticket-1", "source_type": "support_ticket"},
                {"text": "Export failed again on Monday.", "source_id": "ticket-1", "source_type": "support_ticket"},
            ],
        },
        {
            "source_type": "support_ticket",
            "pain_points": ["exports"],
            "evidence": [
                {"text": "Export failed on Monday.", "source_id": "ticket-2", "source_type": "support_ticket"},
            ],
        },
    ])

    assert result.items[0]["evidence_count"] == 3
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2")
    assert result.items[0]["frequency"] == 2
    assert result.items[0]["failure_risk_score"] == 1
    assert result.items[0]["opportunity_score"] == 4
    assert result.ticket_source_count == 2
    assert "Customers are asking about reporting friction" in result.markdown


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
                "text": "How can I reset my password?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
                "source_title": "Password reset again",
            },
            {
                "text": "How do I reset my account password?",
                "source_id": "ticket-2",
                "source_type": "support_ticket",
            },
        ],
    }])

    assert result.ticket_source_count == 2
    assert result.items[0]["source_ids"] == ("ticket-1", "ticket-2")
    assert result.output_checks["condensed"] is True


def test_build_ticket_faq_markdown_counts_unidentified_source_rows_once() -> None:
    result = build_ticket_faq_markdown([
        {
            "source_type": "support_ticket",
            "source_title": "Login reset",
            "evidence": [
                {"text": "How do I reset my password?", "source_type": "support_ticket"},
                {"text": "How do I reset my account password?", "source_type": "support_ticket"},
            ],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Login reset",
            "evidence": [
                {"text": "How can I reset my password?", "source_type": "support_ticket"},
            ],
        },
    ])

    assert result.ticket_source_count == 2
    assert result.items[0]["source_ids"] == ("row:1", "row:2")
    assert result.output_checks["condensed"] is True


def test_build_ticket_faq_markdown_condenses_overflow_sources_instead_of_truncating() -> None:
    opportunities = [
        {
            "source_type": "support_ticket",
            "source_title": f"Unique issue {index}",
            "evidence": [{
                "text": f"How do I handle unique issue {index}?",
                "source_id": f"ticket-{index}{suffix}",
                "source_type": "support_ticket",
            }],
        }
        for index in range(1, 10)
        for suffix in ("", "-b")
    ]

    result = build_ticket_faq_markdown(opportunities, max_items=8)

    assert len(result.items) == 8
    assert result.ticket_source_count == 18
    assert result.output_checks["uses_user_vocabulary"] is True
    assert result.output_checks["condensed"] is True
    assert result.items[-1]["topic"] == "other support issues"
    assert result.items[-1]["source_ids"] == ("ticket-8", "ticket-8-b", "ticket-9", "ticket-9-b")


def test_build_ticket_faq_markdown_treats_zero_max_items_as_unlimited() -> None:
    opportunities = [
        {
            "source_type": "support_ticket",
            "source_title": f"Unique issue {index}",
            "evidence": [{
                "text": f"How do I handle unique issue {index}?",
                "source_id": f"ticket-{index}{suffix}",
                "source_type": "support_ticket",
            }],
        }
        for index in range(1, 10)
        for suffix in ("", "-b")
    ]

    result = build_ticket_faq_markdown(opportunities, max_items=0)

    assert len(result.items) == 9
    assert result.ticket_source_count == 18
    assert "other support issues" not in {item["topic"] for item in result.items}
    assert result.items[-1]["source_ids"] == ("ticket-9", "ticket-9-b")


def test_ticket_faq_cli_writes_markdown_file(tmp_path: Path) -> None:
    output = tmp_path / "ticket_faq.md"
    result_output = tmp_path / "ticket_faq_result.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(SUPPORT_TICKET_CSV),
            "--source-format",
            "csv",
            "--title",
            "Support FAQ",
            "--support-contact",
            "1-800-555-0100",
            "--output",
            str(output),
            "--result-output",
            str(result_output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == ""
    markdown = output.read_text(encoding="utf-8")
    assert markdown.startswith("# Support FAQ")
    assert "Ticket sources used: 4" in markdown
    assert "ticket-acme-1" in markdown
    assert "contact support at 1-800-555-0100" in markdown
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["status"] == "ok"
    assert result["source_count"] == 4
    assert result["ticket_source_count"] == 4
    assert result["failed_output_checks"] == []
    assert result["output"]["markdown_path"] == str(output)
    assert result["diagnostics"]["question_source_counts"] == {
        "customer_wording": 2,
    }
    assert result["diagnostics"]["source_mix"] == {
        "source_channel_counts": {"support_tickets": 4},
        "source_type_counts": {"support_ticket": 4},
        "weighted_source_volume": 4,
        "weighted_source_volume_by_channel": {"support_tickets": 4},
        "weighted_source_volume_by_type": {"support_ticket": 4},
        "zero_result_search_source_count": 0,
    }
    assert result["diagnostics"]["run_summary"] == {
        "status": "ok",
        "source_count": 4,
        "ticket_source_count": 4,
        "generated": 2,
        "weighted_source_volume": 4,
        "source_channel_counts": {"support_tickets": 4},
        "zero_result_search_source_count": 0,
        "output_checks": {
            "passed": 4,
            "failed": 0,
            "total": 4,
            "failed_checks": [],
        },
        "vocabulary_gaps": {
            "term_mapping_count": 0,
            "mapped_topic_count": 0,
            "zero_result_mapping_count": 0,
            "max_opportunity_score": 0,
            "top_customer_terms": [],
        },
        "item_score_distribution": {
            "count": 2,
            "min": 2,
            "max": 4,
            "average": 3.0,
            "bands": {
                "zero": 0,
                "low_1_to_4": 2,
                "medium_5_to_9": 0,
                "high_10_plus": 0,
            },
        },
        "warning_count": 0,
    }
    assert result["diagnostics"]["ticket_counts"] == [2, 2]
    assert result["diagnostics"]["term_mapping_count"] == 0
    assert result["diagnostics"]["term_mappings"] == []
    assert result["diagnostics"]["items"][0] == {
        "rank": 1,
        "topic": "reporting friction",
        "question": "How do we export the campaign reporting dashboard before renewal?",
        "question_source": "customer_wording",
        "frequency": 2,
        "weighted_frequency": 2,
        "failure_risk_score": 1,
        "failure_risk_signals": ["blocked_access"],
        "opportunity_score": 4,
        "ticket_count": 2,
        "evidence_count": 2,
        "source_id_count": 2,
        "first_source_id": "ticket-northstar-1",
        "source_type_counts": {"support_ticket": 2},
        "source_channel_counts": {"support_tickets": 2},
        "weighted_source_volume_by_type": {"support_ticket": 2},
        "weighted_source_volume_by_channel": {"support_tickets": 2},
        "step_count": 3,
        "term_mapping_count": 0,
    }


def test_ticket_faq_cli_writes_vocabulary_gap_result_diagnostics(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Download report",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    assert "**Vocabulary gaps:**" in completed.stdout
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == ["Download report"]
    assert result["diagnostics"]["term_mapping_count"] == 1
    assert result["diagnostics"]["term_mappings"] == [{
        "rank": 1,
        "topic": "reporting friction",
        "customer_term": "export",
        "documentation_term": "Download report",
        "source_id_count": 2,
        "zero_result_source_count": 0,
        "failure_risk_score": 0,
        "failure_risk_signals": [],
        "opportunity_score": 2,
        "first_source_id": "ticket-1",
    }]
    assert result["diagnostics"]["run_summary"]["vocabulary_gaps"] == {
        "term_mapping_count": 1,
        "mapped_topic_count": 1,
        "zero_result_mapping_count": 0,
        "max_opportunity_score": 2,
        "top_customer_terms": ["export"],
    }
    assert result["diagnostics"]["items"][0]["term_mapping_count"] == 1


def test_ticket_faq_cli_writes_source_mix_result_diagnostics(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "mixed_sources.csv",
        [
            {
                "ticket_id": "ticket-1",
                "subject": "Export blocked",
                "description": "I cannot export the report.",
            },
            {
                "query_id": "search-1",
                "search_query": "export report",
                "results_count": "0",
                "search_count": "25",
            },
            {
                "query_id": "search-1",
                "search_query": "export attribution report",
                "search_count": "10",
            },
            {
                "chat_id": "chat-1",
                "message": "How do I change my login email?",
            },
            {
                "sales_objection_id": "objection-1",
                "sales_objection": "Prospect asked: can we export the report?",
            },
        ],
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["source_count"] == 5
    assert result["diagnostics"]["source_mix"] == {
        "source_channel_counts": {
            "chats": 1,
            "sales_inputs": 1,
            "search_logs": 2,
            "support_tickets": 1,
        },
        "source_type_counts": {
            "chat": 1,
            "sales_objection": 1,
            "search_log": 2,
            "support_ticket": 1,
        },
        "weighted_source_volume": 28,
        "weighted_source_volume_by_channel": {
            "chats": 1,
            "sales_inputs": 1,
            "search_logs": 25,
            "support_tickets": 1,
        },
        "weighted_source_volume_by_type": {
            "chat": 1,
            "sales_objection": 1,
            "search_log": 25,
            "support_ticket": 1,
        },
        "zero_result_search_source_count": 1,
    }
    assert result["diagnostics"]["run_summary"]["weighted_source_volume"] == 28
    assert result["diagnostics"]["run_summary"]["source_channel_counts"] == {
        "chats": 1,
        "sales_inputs": 1,
        "search_logs": 2,
        "support_tickets": 1,
    }
    assert result["diagnostics"]["run_summary"]["zero_result_search_source_count"] == 1
    assert result["diagnostics"]["run_summary"]["output_checks"]["failed"] == 0
    assert (
        result["diagnostics"]["run_summary"]["item_score_distribution"]["count"]
        == result["generated"]
    )
    reporting_item = result["diagnostics"]["items"][0]
    assert reporting_item["topic"] == "reporting friction"
    assert reporting_item["source_type_counts"] == {
        "sales_objection": 1,
        "search_log": 1,
        "support_ticket": 1,
    }
    assert reporting_item["source_channel_counts"] == {
        "sales_inputs": 1,
        "search_logs": 1,
        "support_tickets": 1,
    }
    assert reporting_item["weighted_source_volume_by_type"] == {
        "sales_objection": 1,
        "search_log": 25,
        "support_ticket": 1,
    }
    assert reporting_item["weighted_source_volume_by_channel"] == {
        "sales_inputs": 1,
        "search_logs": 25,
        "support_tickets": 1,
    }


def test_ticket_faq_cli_accepts_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Billing center",
        "--documentation-term-file",
        str(FAQ_DOCUMENTATION_TERMS),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_term_files"] == [
        str(FAQ_DOCUMENTATION_TERMS)
    ]
    assert result["config"]["documentation_terms"] == [
        "Billing center",
        "Single sign-on setup",
        "Download report",
        "Dashboard analytics",
    ]
    assert result["diagnostics"]["term_mappings"][0]["customer_term"] == "export"
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_accepts_json_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.json"
    term_file.write_text(
        json.dumps({
            "documentation_terms": ["Single sign-on setup"],
            "documents": [
                {"title": "Download report"},
                {"heading": "Dashboard analytics"},
            ],
        }),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == [
        "Single sign-on setup",
        "Download report",
        "Dashboard analytics",
    ]
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_accepts_nested_json_documentation_term_values(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.json"
    term_file.write_text(
        json.dumps({
            "documentation_terms": [
                {"title": "Download report"},
                {"heading": "Dashboard analytics"},
            ]
        }),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == [
        "Download report",
        "Dashboard analytics",
    ]
    assert not any(
        term.startswith("{") for term in result["config"]["documentation_terms"]
    )


def test_ticket_faq_cli_accepts_jsonl_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.jsonl"
    term_file.write_text(
        "\n".join((
            json.dumps("Single sign-on setup"),
            json.dumps({"page_title": "Download report"}),
            json.dumps({"term": "Dashboard analytics"}),
            "",
        )),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == [
        "Single sign-on setup",
        "Download report",
        "Dashboard analytics",
    ]
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_accepts_csv_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.csv"
    term_file.write_text(
        "\n".join((
            "title,slug",
            "Single sign-on setup,sso",
            "Download report,download-report",
            "Dashboard analytics,dashboard",
            "",
        )),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == [
        "Single sign-on setup",
        "Download report",
        "Dashboard analytics",
    ]
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_accepts_suffixless_csv_documentation_term_file_with_format(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms_export"
    term_file.write_text(
        "\n".join((
            "title,slug",
            "Single sign-on setup,sso",
            "Download report,download-report",
            "Dashboard analytics,dashboard",
            "",
        )),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--documentation-term-format",
        "csv",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_term_format"] == "csv"
    assert result["config"]["documentation_terms"] == [
        "Single sign-on setup",
        "Download report",
        "Dashboard analytics",
    ]
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_accepts_bom_and_multiline_csv_documentation_term_file(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.csv"
    term_file.write_text(
        '\ufefftitle,body\n"Download report","First line\nsecond line"\n',
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["documentation_terms"] == ["Download report"]
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Download report"
    )


def test_ticket_faq_cli_rejects_missing_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(tmp_path / "missing_terms.txt"),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file not found:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_empty_documentation_term_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.txt"
    term_file.write_text("\n# headings export placeholder\n  \n", encoding="utf-8")

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file contains no terms:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_unrecognized_csv_documentation_term_fields(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.csv"
    term_file.write_text(
        "\n".join((
            "url,slug",
            "https://help.example/export,export",
            "",
        )),
        encoding="utf-8",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file has no recognized term fields:" in completed.stderr
    assert "expected one of: documentation_term" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_unrecognized_json_documentation_term_fields(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.json"
    term_file.write_text(
        json.dumps({"documents": [{"url": "https://help.example/export"}]}),
        encoding="utf-8",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file has no recognized term fields:" in completed.stderr
    assert "expected one of: documentation_term" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_unrecognized_jsonl_documentation_term_fields(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.jsonl"
    term_file.write_text(
        json.dumps({"url": "https://help.example/export"}) + "\n",
        encoding="utf-8",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file has no recognized term fields:" in completed.stderr
    assert "expected one of: documentation_term" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_applies_documentation_term_format_to_suffixless_file(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms_export"
    term_file.write_text(
        "title,slug\nDownload report,download-report\n",
        encoding="utf-8",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
        "--documentation-term-format",
        "json",
    )

    assert completed.returncode == 1
    assert "--documentation-term-file must be valid JSON:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_malformed_json_documentation_term_file(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.json"
    term_file.write_text("{", encoding="utf-8")

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file must be valid JSON:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_malformed_jsonl_documentation_term_file(
    tmp_path: Path,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
        "ticket-2,2026-05-01,Attribution export,How do we export attribution data?,exports",
    )
    term_file = tmp_path / "terms.jsonl"
    term_file.write_text('{"title": "Download report"}\n{', encoding="utf-8")

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term-file",
        str(term_file),
    )

    assert completed.returncode == 1
    assert "--documentation-term-file must be valid JSONL:" in completed.stderr
    assert "line 2" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_accepts_custom_vocabulary_gap_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO setup,How do I enable SSO for my team?,authentication",
        "ticket-2,2026-05-01,SSO setup,How do we enable SSO for our team?,authentication",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Single sign-on setup",
        "--vocabulary-gap-rule",
        "SSO,single sign-on",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["vocabulary_gap_rules"] == [["SSO", "single sign-on"]]
    assert result["diagnostics"]["term_mapping_count"] == 1
    assert result["diagnostics"]["term_mappings"][0]["customer_term"] == "SSO"
    assert (
        result["diagnostics"]["term_mappings"][0]["documentation_term"]
        == "Single sign-on setup"
    )


def test_ticket_faq_cli_rejects_single_term_vocabulary_gap_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO setup,How do I enable SSO for my team?,authentication",
        "ticket-2,2026-05-01,SSO setup,How do we enable SSO for our team?,authentication",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--vocabulary-gap-rule",
        "SSO",
    )

    assert completed.returncode == 1
    assert (
        "--vocabulary-gap-rule must include at least two comma-separated terms"
        in completed.stderr
    )


def test_ticket_faq_cli_rejects_case_duplicate_vocabulary_gap_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Export setup,How do I export data?,exports",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--vocabulary-gap-rule",
        "Export,export",
    )

    assert completed.returncode == 1
    assert (
        "--vocabulary-gap-rule must include at least two comma-separated terms"
        in completed.stderr
    )
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_accepts_custom_intent_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Warehouse sync lag,The CRM warehouse sync is delayed every morning by connector lag.,sync",
        "ticket-2,2026-05-01,Connector lag,The CRM connector lag keeps the warehouse sync delayed every morning.,sync",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--intent-rule",
        "data freshness=warehouse sync,connector lag",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "data freshness", "keywords": ["warehouse sync", "connector lag"]}
    ]
    assert result["diagnostics"]["items"][0]["topic"] == "data freshness"
    assert result["diagnostics"]["items"][0]["ticket_count"] == 2


def test_ticket_faq_cli_prioritizes_custom_intent_rule_over_defaults(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Export delay,The attribution export is late.,exports",
        "ticket-2,2026-05-01,Export delay,The attribution export is late again.,exports",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--intent-rule",
        "custom reporting=attribution export",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["diagnostics"]["items"][0]["topic"] == "custom reporting"


@pytest.mark.parametrize(
    "rule",
    [
        "data freshness",
        "=warehouse sync",
        "data freshness=",
        "data freshness=,",
    ],
)
def test_ticket_faq_cli_rejects_invalid_intent_rule(tmp_path: Path, rule: str) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--intent-rule",
        rule,
    )

    assert completed.returncode == 1
    assert (
        "--intent-rule must use topic=keyword,keyword with at least one keyword"
        in completed.stderr
    )
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_dedupes_custom_intent_keywords_by_case(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--intent-rule",
        "data freshness=Warehouse Sync,warehouse sync",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "data freshness", "keywords": ["Warehouse Sync"]}
    ]
    assert result["diagnostics"]["items"][0]["topic"] == "data freshness"


def test_ticket_faq_cli_uses_first_matching_custom_intent_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--intent-rule",
        "first topic=warehouse sync",
        "--intent-rule",
        "second topic=warehouse sync",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "first topic", "keywords": ["warehouse sync"]},
        {"topic": "second topic", "keywords": ["warehouse sync"]},
    ]
    assert result["diagnostics"]["items"][0]["topic"] == "first topic"


def test_ticket_faq_cli_accepts_json_rule_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO sync,How do I enable SSO after warehouse sync?,sync",
        "ticket-2,2026-05-01,SSO sync,How do we enable SSO after warehouse sync?,sync",
    )
    rule_file = tmp_path / "faq_rules.json"
    rule_file.write_text(
        json.dumps({
            "intent_rules": [
                {"topic": "data freshness", "keywords": ["warehouse sync"]}
            ],
            "vocabulary_gap_rules": [["SSO", "single sign-on"]],
        }),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Single sign-on setup",
        "--rule-file",
        str(rule_file),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["rule_files"] == [str(rule_file)]
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "data freshness", "keywords": ["warehouse sync"]}
    ]
    assert result["config"]["vocabulary_gap_rules"] == [["SSO", "single sign-on"]]
    assert result["diagnostics"]["items"][0]["topic"] == "data freshness"
    assert result["diagnostics"]["term_mappings"][0]["customer_term"] == "SSO"


def test_ticket_faq_cli_checked_rule_file_example_affects_output(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO sync,How do I enable SSO after warehouse sync?,sync",
        "ticket-2,2026-05-01,SSO sync,How do we enable SSO after warehouse sync?,sync",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Single sign-on setup",
        "--rule-file",
        str(FAQ_CUSTOM_RULES),
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["rule_files"] == [str(FAQ_CUSTOM_RULES)]
    assert result["config"]["custom_intent_rules"] == [
        {
            "topic": "data freshness",
            "keywords": ["warehouse sync", "connector lag"],
        }
    ]
    assert result["config"]["vocabulary_gap_rules"] == [["SSO", "single sign-on"]]
    assert result["diagnostics"]["items"][0]["topic"] == "data freshness"
    assert result["diagnostics"]["term_mappings"][0] == {
        "customer_term": "SSO",
        "documentation_term": "Single sign-on setup",
        "failure_risk_score": 0,
        "failure_risk_signals": [],
        "first_source_id": "ticket-1",
        "opportunity_score": 2,
        "rank": 1,
        "source_id_count": 2,
        "topic": "data freshness",
        "zero_result_source_count": 0,
    }


def test_ticket_faq_cli_rule_flags_take_precedence_over_rule_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )
    rule_file = tmp_path / "faq_rules.json"
    rule_file.write_text(
        json.dumps({
            "intent_rules": [
                {"topic": "file topic", "keywords": ["warehouse sync"]}
            ]
        }),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--rule-file",
        str(rule_file),
        "--intent-rule",
        "cli topic=warehouse sync",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["custom_intent_rules"] == [
        {"topic": "cli topic", "keywords": ["warehouse sync"]},
        {"topic": "file topic", "keywords": ["warehouse sync"]},
    ]
    assert result["diagnostics"]["items"][0]["topic"] == "cli topic"


def test_ticket_faq_cli_vocabulary_flags_take_precedence_over_rule_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO setup,How do I enable SSO for my team?,auth",
        "ticket-2,2026-05-01,SSO setup,How do we enable SSO for our team?,auth",
    )
    rule_file = tmp_path / "faq_rules.json"
    rule_file.write_text(
        json.dumps({"vocabulary_gap_rules": [["login", "authentication"]]}),
        encoding="utf-8",
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Single sign-on setup",
        "--rule-file",
        str(rule_file),
        "--vocabulary-gap-rule",
        "SSO,single sign-on",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["config"]["vocabulary_gap_rules"] == [
        ["SSO", "single sign-on"],
        ["login", "authentication"],
    ]
    assert result["diagnostics"]["term_mappings"][0]["customer_term"] == "SSO"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "--rule-file must contain a JSON object"),
        ({"unknown": []}, "--rule-file contains unsupported key(s): unknown"),
        ({"intent_rules": "bad"}, "--rule-file intent_rules must be an array"),
        (
            {"intent_rules": ["bad"]},
            "--rule-file intent_rules[1] must be an object",
        ),
        (
            {"intent_rules": [{"topic": "data freshness", "keywords": "bad"}]},
            "--rule-file intent_rules[1].keywords must be an array",
        ),
        (
            {"intent_rules": [{"topic": "data freshness", "keywords": []}]},
            "--rule-file intent_rules[1] is invalid",
        ),
        (
            {"intent_rules": [{"topic": "data=freshness", "keywords": ["sync"]}]},
            "--rule-file intent_rules[1].topic cannot contain delimiter",
        ),
        (
            {"intent_rules": [{"topic": "data freshness", "keywords": ["sync,lag"]}]},
            "--rule-file intent_rules[1].keywords cannot contain delimiter",
        ),
        (
            {"intent_rules": [{"topic": "data freshness", "keywords": [5]}]},
            "--rule-file intent_rules[1].keywords must contain string values",
        ),
        (
            {"vocabulary_gap_rules": "bad"},
            "--rule-file vocabulary_gap_rules must be an array",
        ),
        (
            {"vocabulary_gap_rules": ["bad"]},
            "--rule-file vocabulary_gap_rules[1] must be an array",
        ),
        (
            {"vocabulary_gap_rules": [["SSO"]]},
            "--rule-file vocabulary_gap_rules[1] is invalid",
        ),
        (
            {"vocabulary_gap_rules": [["SSO,login", "single sign-on"]]},
            "--rule-file vocabulary_gap_rules[1] cannot contain delimiter",
        ),
        (
            {"vocabulary_gap_rules": [["SSO", None]]},
            "--rule-file vocabulary_gap_rules[1] must contain string values",
        ),
    ],
)
def test_ticket_faq_cli_rejects_invalid_rule_file(
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )
    rule_file = tmp_path / "faq_rules.json"
    rule_file.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run_ticket_faq_cli(
        source,
        "--rule-file",
        str(rule_file),
    )

    assert completed.returncode == 1
    assert message in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_missing_rule_file(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )

    completed = _run_ticket_faq_cli(
        source,
        "--rule-file",
        str(tmp_path / "missing.json"),
    )

    assert completed.returncode == 1
    assert "--rule-file not found:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_rejects_malformed_rule_file_json(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Sync lag,The warehouse sync is delayed.,sync",
        "ticket-2,2026-05-01,Sync lag,The warehouse sync is delayed again.,sync",
    )
    rule_file = tmp_path / "faq_rules.json"
    rule_file.write_text("{", encoding="utf-8")

    completed = _run_ticket_faq_cli(
        source,
        "--rule-file",
        str(rule_file),
    )

    assert completed.returncode == 1
    assert "--rule-file must be valid JSON:" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_ticket_faq_cli_sorts_vocabulary_gap_result_diagnostics_by_impact(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "searches.csv",
        [
            {
                "query_id": "search-1",
                "search_query": "How do I export attribution report?",
                "search_count": "25",
                "results_count": "0",
            },
            {
                "ticket_id": "ticket-1",
                "description": "I cannot export attribution data.",
                "pain_category": "exports",
            },
            {
                "ticket_id": "ticket-2",
                "description": "Where can I find my bill?",
                "pain_category": "billing",
            },
            {
                "ticket_id": "ticket-3",
                "description": "Where do I find my bill?",
                "pain_category": "billing",
            },
        ],
    )
    result_output = tmp_path / "ticket_faq_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--documentation-term",
        "Download report",
        "--documentation-term",
        "Invoice settings",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 0
    result = json.loads(result_output.read_text(encoding="utf-8"))
    mappings = result["diagnostics"]["term_mappings"]
    assert [mapping["customer_term"] for mapping in mappings] == ["export", "bill"]
    assert mappings[0]["opportunity_score"] == 78
    assert mappings[0]["zero_result_source_count"] == 1
    assert mappings[1]["opportunity_score"] == 2
    assert mappings[1]["zero_result_source_count"] == 0
    assert result["diagnostics"]["run_summary"]["vocabulary_gaps"] == {
        "term_mapping_count": 2,
        "mapped_topic_count": 2,
        "zero_result_mapping_count": 1,
        "max_opportunity_score": 78,
        "top_customer_terms": ["export", "bill"],
    }


def test_ticket_faq_cli_filters_csv_to_date_window(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
            "ticket-new-2,2026-05-02,Login email,How do we change our login email?,login",
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
    assert "Ticket sources used: 2" in completed.stdout


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


def test_ticket_faq_cli_requires_output_checks_for_packaged_rows() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(SUPPORT_TICKET_CSV),
            "--source-format",
            "csv",
            "--require-output-checks",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "How do I change my login email?" in completed.stdout
    assert "FAQ output checks failed" not in completed.stderr


def test_ticket_faq_cli_fails_required_output_checks_for_weak_rows(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-1,2026-05-01,Unique one,The export button moved.,exports",
            "ticket-2,2026-05-01,Unique two,Billing receipt is missing.,billing",
    )
    result_output = tmp_path / "failed_result.json"

    completed = _run_ticket_faq_cli(
        source,
        "--require-output-checks",
        "--result-output",
        str(result_output),
    )

    assert completed.returncode == 1
    assert "FAQ output checks failed" in completed.stderr
    assert "condensed" in completed.stderr
    assert "uses_user_vocabulary" in completed.stderr
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["status"] == "failed_output_checks"
    # #1460: two one-off questions are no longer billed as repeat FAQ items,
    # so no items are generated and every output check fails.
    assert result["failed_output_checks"] == [
        "condensed",
        "has_action_items",
        "resolution_evidence_scoped",
        "uses_user_vocabulary",
    ]
    assert result["generated"] == 0
    assert result["diagnostics"]["rendered_ticket_source_count"] == 0
    assert result["diagnostics"]["unrepresented_ticket_sources"] == 2
    assert result["diagnostics"]["output_check_details"] == [
        {
            "check": "condensed",
            "passed": False,
            "why": (
                "Some ticket sources were not represented in generated FAQ items. "
                "ticket_source_count=2, rendered_ticket_source_count=0."
            ),
        },
        {
            "check": "has_action_items",
            "passed": False,
            "why": "One or more FAQ items did not include actionable next steps.",
        },
        {
            "check": "resolution_evidence_scoped",
            "passed": False,
            "why": (
                "One or more proven answers used resolution evidence outside "
                "its selected question scope."
            ),
        },
        {
            "check": "uses_user_vocabulary",
            "passed": False,
            "why": (
                "One or more FAQ questions did not come from customer wording "
                "or source policy."
            ),
        },
    ]
    assert result["diagnostics"]["run_summary"]["status"] == "failed_output_checks"
    assert result["diagnostics"]["run_summary"]["output_checks"] == {
        "passed": 0,
        "failed": 4,
        "total": 4,
        "failed_checks": [
            "condensed",
            "has_action_items",
            "resolution_evidence_scoped",
            "uses_user_vocabulary",
        ],
    }
    assert result["diagnostics"]["run_summary"]["generated"] == 0
    assert result["diagnostics"]["run_summary"]["item_score_distribution"]["count"] == 0
    non_repeat_warnings = [
        warning
        for warning in result["diagnostics"]["warnings"]
        if warning["code"] == "non_repeat_tickets_excluded"
    ]
    assert len(non_repeat_warnings) == 1
    assert non_repeat_warnings[0]["ticket_count"] == 2
    assert non_repeat_warnings[0]["question_count"] == 2


def test_ticket_faq_cli_rejects_as_of_date_without_window(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
    )
    completed = _run_ticket_faq_cli(source, "--as-of-date", "2026-05-20")

    assert completed.returncode == 1
    assert "--as-of-date requires --window-days" in completed.stderr


# --- Held-out publishability corpus for the #1466 Option 1 inversion ---------
# Pinned per the operator decision: the gate must publish realistic answers
# across varied verbs and symptom/fix synonym pairs (not just the four cited
# examples), and keep the honesty-floor reject classes rejected. These are
# probed directly against _resolution_text_is_publishable so a regression in
# the gate fails here, not only in an end-to-end assertion.

_HELD_OUT_PUBLISHABLE = (
    # varied imperative verbs, including verbs absent from the action-term list
    ("Why is my sync slow?", "Schedule the sync to run during off-peak hours."),
    ("How do I keep my view?", "Pin the saved view to your dashboard so it loads first."),
    ("I never got the email", "Check the spam folder for the confirmation email and mark it not spam."),
    ("My export times out", "Narrow the export window to under 90 days and retry the export."),
    ("Wrong team got my ticket", "Forward the ticket to the billing queue from the actions menu."),
    ("Who owns this case?", "Assign the case to yourself, then add an internal label."),
    ("Change workspace name", "Rename the workspace under Settings then General."),
    ("Too many old threads", "Archive the old thread and start a fresh conversation."),
    ("Form will not submit", "Submit the form again after clearing the browser cache."),
    ("Stop my plan", "Cancel the subscription from Billing then Plan before the renewal date."),
    # symptom/fix synonym pairs (topic overlap demoted -> still publishable)
    ("I cannot log in", "Reset the SSO certificate and re-enable single sign-on for the org."),
    ("App keeps crashing", "Clear the cache and restart the app to stop the crash."),
    ("I was charged twice", "Open Billing, find the duplicate charge, and request a refund."),
    # past-tense / first-person action narration
    ("How do I update MFA settings?", "I enabled MFA and configured the authenticator, then updated the settings."),
    # numbered steps
    ("Sensor offline", "1. Open Settings. 2. Tap Devices. 3. Re-pair the sensor."),
    # "to fix this," instruction preamble
    ("Stuck on payment", "To fix this, clear the saved card and add it again under Billing."),
    # Guards for the contact-redirect / question reject classes (#1466 round 7):
    # real steps that merely contain "send"/"message", or a trailing redirect or
    # question after a genuine step, must still publish on the step.
    ("Cannot reach owner", "Message the channel owner to request access."),
    ("Where do I send it?", "Send the report to your team from the Export menu."),
    ("App errors out", "Reset the cache, then DM us if the error persists."),
    ("Service is down", "Check the logs. Did it work? If not, restart the service."),
    # A real UI-path instruction must still publish after the structural
    # recognizers were moved behind the per-sentence rejects (round-7 BLOCKER).
    ("Disable airplane mode", "Go to Settings then Phone and toggle airplane mode off."),
)

_HELD_OUT_REJECTED = (
    ("anything", "Customer did not respond, closing this out."),
    ("anything", "Sent an update to the customer."),
    ("anything", "Replied to the requester with the latest status details."),
    ("anything", "Provided the customer an update on the timeline."),
    ("anything", "Escalated to T2 for further review."),
    ("anything", "Refunded per policy 4.2."),
    ("anything", "Thank you for contacting support. We received your request."),
    ("anything", "We have closed your ticket as resolved."),
    ("anything", "Done."),
    ("Why was I charged?", "The bank charged a fee after I closed the account."),
    # Declarative status / opinion commentary must NOT publish as a step (the
    # inversion's over-accept flip-side: the imperative object-shape must not
    # match a clause subject + copula). Round-6 review MAJOR.
    ("anything", "Honestly this is a known issue and pretty annoying for everyone."),
    ("anything", "That is the expected behavior for the free tier."),
    ("anything", "Unfortunately the issue is a duplicate of an existing bug."),
    ("anything", "Is the account still active on your end?"),
    # Contact-channel redirection: a hand-off to a human over a private channel
    # is imperative-shaped but answers nothing. Surfaced running the gate over
    # real support replies (Twitter brand replies), round 7.
    ("My account is locked", "Please send me a private message so we can help."),
    ("I need help", "Have your friend message us."),
    ("Where is my order?", "Send us a Direct Message with your order number."),
    ("Login broken", "DM us your account email and we'll investigate."),
    # Answer-is-a-question: a diagnostic prompt back to the requester is not a
    # step they can follow (a non-copula interrogative, complementing the
    # existing "Is the account still active?" copula case above).
    ("Internet keeps dropping", "Did the lights change on the router when this happened?"),
    # Structural recognizers must not bypass the per-sentence rejects (round-7
    # BLOCKER): a UI-path or numbered diagnostic question, and a UI-path
    # declarative, are non-answers even though they carry a step-like shape.
    ("Phone carrier", "Did Settings then Phone show the carrier toggle?"),
    ("Router lights", "1. Did the lights change on the router?"),
    ("Billing location", "The issue is in Billing then Plan and not your account."),
)


@pytest.mark.parametrize(("question_text", "resolution_text"), _HELD_OUT_PUBLISHABLE)
def test_resolution_gate_publishes_held_out_real_answers(
    question_text: str, resolution_text: str
) -> None:
    assert _resolution_text_is_publishable(
        resolution_text, question_text=question_text
    ) is True


@pytest.mark.parametrize(("question_text", "resolution_text"), _HELD_OUT_REJECTED)
def test_resolution_gate_rejects_held_out_non_answers(
    question_text: str, resolution_text: str
) -> None:
    assert _resolution_text_is_publishable(
        resolution_text, question_text=question_text
    ) is False


def test_resolution_gate_drafts_held_out_answers_end_to_end() -> None:
    # drafted_answer_count must not regress: a repeated question whose two
    # tickets carry a held-out publishable instruction surfaces as a drafted
    # (resolution_evidence) answer through the full builder.
    question = "How do I rotate my API key?"
    answer = "Open Settings then Developers, revoke the old key, and create a new one."
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "API key rotation",
            "evidence": [
                {"text": question, "source_id": "held-1", "source_type": "support_ticket", "resolution_text": answer},
                {"text": question, "source_id": "held-2", "source_type": "support_ticket", "resolution_text": answer},
            ],
        }]
    )
    item = result.items[0]
    assert item["answer_evidence_status"] == "resolution_evidence"
    assert item["resolution_evidence_scope"] == "scoped"
