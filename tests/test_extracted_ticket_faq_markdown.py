from __future__ import annotations

import csv
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
        "Open the reporting or analytics area and choose the date range you need.",
        "Look for an Export or Download option, then ask an admin to check your role and plan access if it is missing.",
        "If it still does not work, contact support at 1-800-555-0100 and include the cited ticket details.",
    )
    assert result.items[0]["failure_risk_score"] == 1
    assert result.items[0]["failure_risk_signals"] == ("blocked_access",)
    assert result.items[0]["opportunity_score"] == 4
    assert "the export is missing" in result.items[0]["when_to_contact_support"]
    assert result.items[0]["evidence_quotes"] == (
        '`ticket-northstar-1` - Export campaign reports: "How do we export campaign attribution data before renewal?"',
        '`ticket-northstar-2` - Reporting dashboard export: "We cannot export the reporting dashboard for analysts."',
    )
    assert "How do I change my login email?" in result.markdown
    assert "How do we export campaign attribution data before renewal?" in result.markdown
    assert "`ticket-acme-1` - Change login email" in result.markdown
    assert "**What to do next:**" in result.markdown
    assert "self-service" not in result.markdown.lower()
    assert "contact support at 1-800-555-0100" in result.markdown
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }


def test_build_ticket_faq_markdown_does_not_invent_support_contact() -> None:
    loaded = load_source_campaign_opportunities_from_file(SUPPORT_TICKET_CSV, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities)

    assert "contact support at" not in result.markdown.lower()
    assert "If it still does not work, contact support and include the cited ticket details." in result.markdown


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
                "text": "Where can I edit the email address?",
                "source_id": "ticket-email-3",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Billing failure",
            "evidence": [{
                "text": "The payment failed and the balance is wrong.",
                "source_id": "ticket-billing-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Billing locked",
            "evidence": [{
                "text": "I cannot pay because the billing page is locked.",
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
                "text": "Where do I edit the email address?",
                "source_id": "ticket-email-3",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Webhook failure",
            "evidence": [{
                "text": "The API webhook failed.",
                "source_id": "ticket-integration-1",
                "source_type": "support_ticket",
            }],
        },
        {
            "source_type": "support_ticket",
            "source_title": "Webhook error",
            "evidence": [{
                "text": "The integration sync shows the wrong status.",
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
            "evidence": [{
                "text": "export attribution report",
                "source_id": "search-export-1",
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
    ])

    assert [item["topic"] for item in result.items] == [
        "reporting friction",
        "email and profile updates",
    ]
    assert result.items[0]["frequency"] == 20
    assert result.items[0]["weighted_frequency"] == 20
    assert result.items[0]["ticket_count"] == 1
    assert result.items[0]["source_ids"] == ("search-export-1",)
    assert result.items[0]["opportunity_score"] == 20
    assert result.items[1]["frequency"] == 2
    assert result.items[1]["weighted_frequency"] == 2


def test_build_ticket_faq_markdown_prefers_explicit_aggregate_weight_fields() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "search_log",
        "query_id": "search-export-1",
        "search_query": "export attribution report",
        "frequency": "1",
        "search_count": "25",
        "evidence": [{
            "text": "export attribution report",
            "source_id": "search-export-1",
            "source_type": "search_log",
        }],
    }])

    assert result.items[0]["frequency"] == 25
    assert result.items[0]["weighted_frequency"] == 25
    assert result.items[0]["ticket_count"] == 1


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
                "text": "download attribution report",
                "source_id": "search-export-1",
                "source_type": "search_log",
                "source_weight": "200",
            },
            {
                "text": "dashboard attribution report",
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
            "search_query": "export dashboard attribution",
            "result_count": "0",
            "evidence": [{
                "text": "export dashboard attribution",
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
                "text": "Where can I edit the email address?",
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
        [{
            "source_type": "support_ticket",
            "source_title": "Attribution dashboard",
            "evidence": [{
                "text": "How do I export the attribution dashboard?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        }],
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
            "source_id_count": 1,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 1,
            "first_source_id": "ticket-1",
        },
        {
            "customer_term": "dashboard",
            "documentation_term": "Analytics",
            "suggestion": (
                'Add "dashboard" as alternate phrasing for "Analytics" '
                "in FAQ headings and answer text."
            ),
            "source_id_count": 1,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 1,
            "first_source_id": "ticket-1",
        },
    )
    assert "**Vocabulary gaps:**" in result.markdown
    assert 'Customers say "export"; documentation says "Download report".' in result.markdown
    assert "(Seen in 1 source(s); mapping score 1.)" in result.markdown


def test_build_ticket_faq_markdown_accepts_custom_vocabulary_gap_rules() -> None:
    result = build_ticket_faq_markdown(
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
            "source_id_count": 1,
            "zero_result_source_count": 0,
            "failure_risk_score": 0,
            "failure_risk_signals": (),
            "opportunity_score": 1,
            "first_source_id": "ticket-1",
        },
    )
    assert 'Customers say "SSO"; documentation says "Single sign-on setup".' in result.markdown


def test_build_ticket_faq_markdown_prioritizes_custom_vocabulary_gap_rules() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export dashboard bill SSO",
            "evidence": [{
                "text": "How do I export dashboard bill data after SSO setup?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        }],
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

    assert result.source_count == 2
    assert result.ticket_source_count == 1
    assert result.items[0]["source_ids"] == ("ticket-1",)
    assert result.items[0]["term_mappings"][0]["customer_term"] == "bill"
    assert result.items[0]["term_mappings"][0]["documentation_term"] == "Invoice settings"
    assert "`doc-1`" not in result.markdown


def test_build_ticket_faq_markdown_skips_vocabulary_gap_when_docs_match_customer_term() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "support_ticket",
            "source_title": "Export report",
            "evidence": [{
                "text": "How do I export the attribution dashboard?",
                "source_id": "ticket-1",
                "source_type": "support_ticket",
            }],
        }],
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
                "source_title": "Loan payment issue",
                "pain_points": ["Fees"],
                "evidence": [{
                    "text": "My payment was applied to the wrong loan balance.",
                    "source_id": "cfpb:2",
                    "source_type": "support_ticket",
                    "source_title": "Loan payment issue",
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
    }
    assert "Open the bill, statement, payment history, or dispute record" in result.markdown
    assert "contact support at https://example.com/support" in result.markdown


def test_build_ticket_faq_markdown_uses_source_policy_question_when_customer_wording_is_missing() -> None:
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
    assert result.items[0]["question"] == "What should I do about reporting friction?"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks["uses_user_vocabulary"] is True


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
            }
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
            }
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


def test_build_ticket_faq_markdown_condenses_tail_groups_when_item_cap_is_lower_than_topics() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "source_title": "Credit report dispute",
            "evidence": [{
                "text": f"My credit report has the wrong balance on account {index}.",
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
            "source_title": "Debt collection issue",
            "evidence": [{
                "text": "A debt collector is asking me to pay a debt I do not recognize.",
                "source_id": "debt-1",
                "source_type": "support_ticket",
            }],
        },
    ])

    result = build_ticket_faq_markdown(rows, max_items=2)

    assert [item["topic"] for item in result.items] == [
        "credit report disputes",
        "other support issues",
    ]
    assert result.items[1]["source_ids"] == ("debt-1", "mortgage-1")
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }


def test_build_ticket_faq_markdown_preserves_top_group_when_single_item_cap_overflows() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": f"Credit report dispute {index}",
                "evidence": [{
                    "text": f"My credit report has the wrong balance on account {index}.",
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
        ],
        max_items=1,
    )

    assert len(result.items) == 1
    assert result.items[0]["topic"] == "credit report disputes"
    assert result.items[0]["source_ids"] == ("credit-1", "credit-2", "credit-3", "debt-1")
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
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
        text="My credit report has incorrect information on account {index}.",
    )
    add_rows(
        150,
        source_title="Debt collection - Attempts to collect debt not owed",
        text="A debt collector says I owe a debt I do not recognize for account {index}.",
    )
    add_rows(
        100,
        source_title="Credit card or prepaid card - Fees or interest",
        text="I paid the XX/XX/2019 credit card installment of {{$100.00}} but it was not credited.",
    )
    add_rows(
        50,
        source_title="Mortgage - Trouble during payment process",
        text="My mortgage servicer will not explain the payment issue for loan {index}.",
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
    }
    assert all(item["question_source"] != "topic_fallback" for item in result.items)
    assert all("XX/XX/2019" not in question for question in questions)
    assert all("allowed ''" not in question for question in questions)
    assert opening["steps"][0].startswith("Gather the application")
    assert "Export or Download" not in " ".join(opening["steps"])
    assert "export is missing" not in opening["when_to_contact_support"]


def test_build_ticket_faq_markdown_uses_financial_steps_for_cfpb_account_topics() -> None:
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
                    "text": "The bonus for opening my money market account was never paid.",
                    "source_id": "cfpb:2",
                    "source_type": "support_ticket",
                    "source_title": "Checking or savings account - Opening an account",
                }],
            },
        ]
    )

    assert result.items[0]["topic"] == "opening an account"
    assert result.items[0]["steps"][0].startswith("Gather the application")
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
            }
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
            }
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
        "1. How do we export campaign attribution data before renewal?",
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
        "Open the reporting or analytics area and choose the date range you need."
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
                    "search_query": "export dashboard attribution",
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
                }
            ]
        },
    )

    assert result.as_dict()["generated"] == 1
    assert result.ticket_source_count == 1
    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["source_ids"] == ("chat-1",)
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
                }
            ]
        },
    )

    assert result.as_dict()["generated"] == 1
    assert result.ticket_source_count == 1
    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["source_ids"] == ("obj-1",)
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
                }
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
                }
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


def test_build_ticket_faq_markdown_uses_financial_steps_for_cfpb_shaped_rows() -> None:
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
                "source_title": "Vehicle loan or lease - Managing the loan or lease",
                "pain_points": ["Managing the loan or lease"],
                "evidence": [{
                    "text": (
                        "My loan payoff balance and payment extension do not "
                        "match what the representative told me."
                    ),
                    "source_id": "cfpb:3205066",
                    "source_type": "support_ticket",
                    "source_title": "Vehicle loan or lease - Managing the loan or lease",
                }],
            },
        ],
        support_contact="https://www.consumerfinance.gov/complaint/",
    )

    markdown = result.markdown
    assert "Open the bill, statement, payment history, or dispute record connected to the issue." in markdown
    assert "Compare the charge, fee, payment, or balance against your receipt, contract, or written confirmation." in markdown
    assert "charge, fee, payment, balance, or dispute still looks wrong" in markdown
    assert "Open your profile, account settings, or login settings" not in markdown
    assert "https://www.consumerfinance.gov/complaint/" in markdown


def test_build_ticket_faq_markdown_uses_debt_collection_steps_before_account_steps() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "complaint",
            "pain_points": ["Attempts to collect debt not owed"],
            "evidence": [{
                "text": "I received a collection letter for a debt I do not owe.",
                "source_id": "cfpb:3182593",
                "source_type": "complaint",
            }],
        }],
        support_contact="https://example.com/support",
    )

    markdown = result.markdown
    assert result.items[0]["topic"] == "debt collection disputes"
    assert "Ask the collector in writing to identify the original creditor" in markdown
    assert "Compare the notice with your payment, settlement, insurance, or provider records" in markdown
    assert "Open your profile, account settings, or login settings" not in markdown
    assert "Open the reporting or analytics area" not in markdown


def test_build_ticket_faq_markdown_uses_credit_report_steps_before_reporting_steps() -> None:
    result = build_ticket_faq_markdown(
        [{
            "source_type": "complaint",
            "pain_points": ["Incorrect information on your report"],
            "evidence": [{
                "text": "There are many mistakes appearing in my credit report.",
                "source_id": "cfpb:3187954",
                "source_type": "complaint",
            }],
        }],
        support_contact="https://example.com/support",
    )

    markdown = result.markdown
    assert result.items[0]["topic"] == "credit report disputes"
    assert "Get your latest credit reports and mark the account" in markdown
    assert "File a dispute with the credit bureau and the company that supplied the information" in markdown
    assert "Open the reporting or analytics area" not in markdown


def test_build_ticket_faq_markdown_replaces_vague_questions_with_source_policy_question() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "complaint",
        "pain_points": ["Incorrect information on your report"],
        "evidence": [{
            "text": "Need help? There are mistakes appearing in my credit report.",
            "source_id": "cfpb:3187954",
            "source_type": "complaint",
        }],
    }])

    assert result.items[0]["question"] == "What should I do if information on my credit report is wrong?"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "## 1. Need help?" not in result.markdown


def test_build_ticket_faq_markdown_uses_substantive_question_after_vague_opener() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "support_ticket",
        "pain_points": ["login reset"],
        "evidence": [{
            "text": "Need help? I cannot reset my password.",
            "source_id": "ticket-1",
            "source_type": "support_ticket",
        }],
    }])

    assert result.items[0]["question"] == "How do I reset my password?"
    assert result.items[0]["question_source"] == "customer_wording"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "## 1. Need help?" not in result.markdown


def test_build_ticket_faq_markdown_does_not_classify_generic_investigation_as_credit_report() -> None:
    result = build_ticket_faq_markdown([{
        "source_type": "support_ticket",
        "pain_points": ["reporting friction"],
        "evidence": [{
            "text": "Need help? I cannot export the investigation dashboard report.",
            "source_id": "ticket-1",
            "source_type": "support_ticket",
        }],
    }])

    assert result.items[0]["topic"] == "reporting friction"
    assert result.items[0]["question"] == "How do I export the investigation dashboard report?"
    assert "Open the reporting or analytics area" in result.markdown
    assert "credit bureau" not in result.markdown


def test_build_ticket_faq_markdown_does_not_treat_cfpb_report_as_saas_reporting() -> None:
    result = build_ticket_faq_markdown([{
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
    }])

    assert result.items[0]["topic"] == "opening an account"
    assert result.items[0]["question_source"] == "customer_wording"
    assert "Open the reporting or analytics area" not in result.markdown
    assert "Gather the application, account-opening notice" in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_credit_report_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "credit_reporting.csv",
        [{
            "Complaint ID": "3181474",
            "Product": "Credit reporting, credit repair services, or other personal consumer reports",
            "Issue": "Improper use of your report",
            "Consumer complaint narrative": (
                "The inquiries are a result of identity theft and should not remain on my report."
            ),
            "Company": "Example Bureau",
        }],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "credit report disputes"
    assert result.items[0]["question_source"] == "source_policy"
    assert result.output_checks == {
        "uses_user_vocabulary": True,
        "condensed": True,
        "has_action_items": True,
    }
    assert "Get your latest credit reports and mark the account" in result.markdown
    assert "Open the reporting or analytics area" not in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_debt_collection_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "debt_collection.csv",
        [{
            "Complaint ID": "3177182",
            "Product": "Debt collection",
            "Issue": "Communication tactics",
            "Consumer complaint narrative": (
                "The collector keeps calling and asking me to confirm my email address for a debt I do not owe."
            ),
            "Company": "Example Collector",
        }],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "debt collection disputes"
    assert result.output_checks["uses_user_vocabulary"] is True
    assert "Ask the collector in writing to identify the original creditor" in result.markdown
    assert "Open your profile, account settings, or login settings" not in result.markdown


def test_build_ticket_faq_markdown_uses_cfpb_product_context_for_mortgage_rows(tmp_path: Path) -> None:
    source = _write_source_csv(
        tmp_path,
        "mortgage.csv",
        [{
            "Complaint ID": "3178554",
            "Product": "Mortgage",
            "Issue": "Struggling to pay mortgage",
            "Consumer complaint narrative": (
                "The servicer posted a foreclosure notice even though the modification documents were submitted."
            ),
            "Company": "Example Servicer",
        }],
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
    }
    assert "Gather the mortgage statement, payment history, escrow record" in result.markdown
    assert "Open the bill, statement, payment history, or dispute record" not in result.markdown


def test_build_ticket_faq_markdown_does_not_treat_generic_loan_modification_as_mortgage(
    tmp_path: Path,
) -> None:
    source = _write_source_csv(
        tmp_path,
        "vehicle_loan.csv",
        [{
            "Complaint ID": "vehicle-1",
            "Product": "Vehicle loan or lease",
            "Issue": "Managing the loan or lease",
            "Consumer complaint narrative": (
                "I need help with a loan modification and payment dispute on my auto loan."
            ),
            "Company": "Example Auto Lender",
        }],
    )
    loaded = load_source_campaign_opportunities_from_file(source, file_format="csv")

    result = build_ticket_faq_markdown(loaded.opportunities, support_contact="https://example.com/support")

    assert result.items[0]["topic"] == "billing and payments"
    assert "mortgage servicer" not in result.markdown
    assert "Gather the mortgage statement" not in result.markdown
    assert "Open the bill, statement, payment history, or dispute record" in result.markdown


def test_build_ticket_faq_markdown_rejects_complaint_process_boilerplate_question() -> None:
    result = build_ticket_faq_markdown([{
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
    }])

    assert result.items[0]["topic"] == "mortgage servicing issues"
    assert result.items[0]["question"] == (
        "What should I do if my mortgage servicer will not fix a payment, payoff, foreclosure, or modification issue?"
    )
    assert result.items[0]["question_source"] == "source_policy"
    assert "## 1. What should I do if my mortgage servicer" in result.markdown
    assert "## 1. What should I do if my husband" not in result.markdown


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
    assert result.items[0]["frequency"] == 1
    assert result.items[0]["failure_risk_score"] == 1
    assert result.items[0]["opportunity_score"] == 2
    assert result.ticket_source_count == 1
    assert "A customer asked about reporting friction" in result.markdown


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


def test_build_ticket_faq_markdown_condenses_overflow_sources_instead_of_truncating() -> None:
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
    assert result.output_checks["condensed"] is True
    assert result.items[-1]["topic"] == "other support issues"
    assert result.items[-1]["source_ids"] == ("ticket-8", "ticket-9")


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
    assert result["diagnostics"]["ticket_counts"] == [2, 2]
    assert result["diagnostics"]["term_mapping_count"] == 0
    assert result["diagnostics"]["term_mappings"] == []
    assert result["diagnostics"]["items"][0] == {
        "rank": 1,
        "topic": "reporting friction",
        "question": "How do we export campaign attribution data before renewal?",
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
        "step_count": 3,
        "term_mapping_count": 0,
    }


def test_ticket_faq_cli_writes_vocabulary_gap_result_diagnostics(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,Attribution export,How do I export attribution data?,exports",
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
        "source_id_count": 1,
        "zero_result_source_count": 0,
        "failure_risk_score": 0,
        "failure_risk_signals": [],
        "opportunity_score": 1,
        "first_source_id": "ticket-1",
    }]
    assert result["diagnostics"]["items"][0]["term_mapping_count"] == 1


def test_ticket_faq_cli_accepts_custom_vocabulary_gap_rule(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
        "ticket-1,2026-05-01,SSO setup,How do I enable SSO for my team?,authentication",
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
    assert mappings[1]["opportunity_score"] == 1
    assert mappings[1]["zero_result_source_count"] == 0


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
    assert "uses_user_vocabulary" not in completed.stderr
    result = json.loads(result_output.read_text(encoding="utf-8"))
    assert result["status"] == "failed_output_checks"
    assert result["failed_output_checks"] == ["condensed"]
    assert result["diagnostics"]["rendered_ticket_source_count"] == 2
    assert result["diagnostics"]["unrepresented_ticket_sources"] == 0
    assert result["diagnostics"]["output_check_details"] == [
        {
            "check": "condensed",
            "passed": False,
            "why": (
                "The FAQ produced one item per ticket source, so the output was not condensed. "
                "ticket_source_count=2, generated=2."
            ),
        },
        {"check": "has_action_items", "passed": True},
        {"check": "uses_user_vocabulary", "passed": True},
    ]


def test_ticket_faq_cli_rejects_as_of_date_without_window(tmp_path: Path) -> None:
    source = _write_ticket_csv(
        tmp_path,
            "ticket-new,2026-05-01,Login email,How do I change my login email?,login",
    )
    completed = _run_ticket_faq_cli(source, "--as-of-date", "2026-05-20")

    assert completed.returncode == 1
    assert "--as-of-date requires --window-days" in completed.stderr
