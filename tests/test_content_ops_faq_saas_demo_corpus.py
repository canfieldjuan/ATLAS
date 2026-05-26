import csv
from pathlib import Path

from extracted_content_pipeline.campaign_source_adapters import (
    source_rows_to_campaign_opportunities,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = ROOT / "extracted_content_pipeline/examples/support_ticket_saas_demo_sources.csv"

EXPECTED_LABEL = "synthetic_b2b_saas_demo"
MIN_ROWS = 36
REQUIRED_PAIN_CATEGORIES = {
    "api and webhooks",
    "billing and plan management",
    "dashboard freshness",
    "data import",
    "integration sync",
    "permissions and seats",
    "reporting export",
    "sso setup",
    "workflow automation",
}
BLOCKED_CONSUMER_FINANCE_TERMS = {
    "bankruptcy",
    "cfpb",
    "credit report",
    "debt collection",
    "escrow",
    "foreclosure",
    "mortgage",
    "payday loan",
}


def _rows() -> list[dict[str, str]]:
    with DEMO_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _haystack(rows: list[dict[str, str]]) -> str:
    return "\n".join(" ".join(row.values()) for row in rows).lower()


def test_saas_demo_corpus_is_labeled_and_domain_clean() -> None:
    rows = _rows()

    assert len(rows) >= MIN_ROWS
    assert {row["Dataset Label"] for row in rows} == {EXPECTED_LABEL}
    assert {row["Source Type"] for row in rows} == {"support_ticket"}
    assert {row["Pain Category"] for row in rows} == REQUIRED_PAIN_CATEGORIES
    assert all(row["Ticket ID"].startswith("saas-demo-") for row in rows)
    assert all(row["Description"].endswith("?") for row in rows)

    corpus_text = _haystack(rows)
    leaked_terms = [
        term for term in sorted(BLOCKED_CONSUMER_FINANCE_TERMS)
        if term in corpus_text
    ]
    assert leaked_terms == []


def test_saas_demo_corpus_generates_valid_faq_output() -> None:
    rows = _rows()
    normalized = source_rows_to_campaign_opportunities(
        rows,
        target_mode="support_account",
    )

    assert normalized.warnings == ()
    assert len(normalized.opportunities) == len(rows)

    result = build_ticket_faq_markdown(
        normalized.opportunities,
        title="Synthetic B2B SaaS Support FAQ Demo",
        max_items=12,
        max_evidence_per_item=3,
        support_contact="https://example.com/support",
    )

    assert result.source_count == len(rows)
    assert result.ticket_source_count == len(rows)
    assert result.output_checks == {
        "condensed": True,
        "has_action_items": True,
        "uses_user_vocabulary": True,
    }
    assert result.items
    assert all(item["source_ids"] for item in result.items)

    rendered_topics = {item["topic"] for item in result.items}
    assert "reporting friction" in rendered_topics
    assert "billing and payments" in rendered_topics
    assert "manual follow-up" in rendered_topics

    markdown = result.markdown.lower()
    leaked_terms = [
        term for term in sorted(BLOCKED_CONSUMER_FINANCE_TERMS)
        if term in markdown
    ]
    assert leaked_terms == []
