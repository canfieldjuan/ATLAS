from __future__ import annotations

import pytest

from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_report_artifact,
)
from extracted_content_pipeline.ticket_faq_markdown import TicketFAQMarkdownResult

from atlas_brain.deflection_pdf_renderer import (
    PDF_QUESTION_DETAIL_LIMIT,
    PDF_RANKED_TABLE_LIMIT,
    _curate_markdown_for_pdf,
    _toc_entries,
    render_deflection_full_report_pdf,
)


def _large_markdown() -> str:
    ranked_rows = "\n".join(
        f"| {index} | Question {index} | {index + 1} |"
        for index in range(1, PDF_RANKED_TABLE_LIMIT + 4)
    )
    details = []
    for index in range(1, PDF_QUESTION_DETAIL_LIMIT + 4):
        details.extend([
            f"### {index}. Question {index}",
            "",
            "**Answer status:** no proven answer yet",
            "",
            "**Complete evidence:**",
            "",
            f"**Source IDs (full list):** ticket-{index}, ticket-extra-{index}",
            "",
            f"- `ticket-{index}` - raw quoted evidence for question {index}",
            "",
        ])
    return "\n".join([
        "# Support Ticket Deflection Report",
        "",
        "## Support Tax Confirmation",
        "",
        "Readable executive summary stays in the PDF.",
        "",
        "## Ranked Question Opportunities",
        "",
        "| Rank | Customer question | Tickets |",
        "|---:|---|---:|",
        ranked_rows,
        "",
        "## Question Details and Evidence",
        "",
        "Each ranked question appears once below.",
        "",
        *details,
    ])


def test_render_deflection_full_report_pdf_from_artifact_markdown() -> None:
    pdf_bytes = render_deflection_full_report_pdf(
        {
            "markdown": (
                "# Support Ticket Deflection Report\n\n"
                "## Support Tax Confirmation\n\n"
                "- Customers ask why duplicate-looking invoices appear \u2014 with "
                "\u201ccafe\u201d users.\n\n"
                "| Rank | Customer question | Tickets |\n"
                "|---:|---|---:|\n"
                "| 1 | Why did I get two invoices? | 4 |\n"
                "\n"
                "**Complete evidence:**\n\n"
                "**Source IDs (full list):** ticket-1\n\n"
                "- `ticket-1` - raw evidence\n"
            ),
        }
    )

    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_curated_pdf_markdown_caps_ranked_table_and_question_details() -> None:
    curated = _curate_markdown_for_pdf(_large_markdown())

    assert f"| {PDF_RANKED_TABLE_LIMIT} | Question {PDF_RANKED_TABLE_LIMIT} |" in curated
    assert f"| {PDF_RANKED_TABLE_LIMIT + 1} | Question {PDF_RANKED_TABLE_LIMIT + 1} |" not in curated
    assert "Ranked question table capped" in curated
    assert f"### {PDF_QUESTION_DETAIL_LIMIT}. Question {PDF_QUESTION_DETAIL_LIMIT}" in curated
    assert f"### {PDF_QUESTION_DETAIL_LIMIT + 1}. Question {PDF_QUESTION_DETAIL_LIMIT + 1}" not in curated
    assert "Question detail blocks capped" in curated
    assert f"ticket-{PDF_QUESTION_DETAIL_LIMIT + 1}" not in curated


def test_curated_pdf_markdown_replaces_complete_evidence_blocks_with_export_pointer() -> None:
    curated = _curate_markdown_for_pdf(_large_markdown())

    assert "**Complete evidence:**" in curated
    assert "complete evidence export JSON" in curated
    assert "**Source IDs (full list):**" not in curated
    assert "raw quoted evidence for question 1" not in curated
    assert "Readable executive summary stays in the PDF." in curated
    assert "Each ranked question appears once below." in curated


def test_curated_pdf_markdown_skips_heading_shaped_evidence_until_real_boundary() -> None:
    markdown = "\n".join([
        "# Support Ticket Deflection Report",
        "",
        "## Question Details and Evidence",
        "",
        "### 1. First question",
        "",
        "**Complete evidence:**",
        "",
        "**Source IDs (full list):** ticket-1",
        "",
        "- `ticket-1` - Failure details",
        "### Error details pasted from a customer email",
        "Raw stack trace should stay out of the curated PDF.",
        "",
        "### 2. Second question",
        "",
        "**Answer status:** no proven answer yet",
        "",
    ])

    curated = _curate_markdown_for_pdf(markdown)

    assert "complete evidence export JSON" in curated
    assert "ticket-1" not in curated
    assert "Error details pasted from a customer email" not in curated
    assert "Raw stack trace should stay out of the curated PDF." not in curated
    assert "### 2. Second question" in curated


def test_curated_pdf_consumes_real_deflection_report_markdown() -> None:
    artifact = build_deflection_report_artifact(
        TicketFAQMarkdownResult(
            markdown="# FAQ",
            source_count=2,
            ticket_source_count=2,
            output_checks={"condensed": True},
            items=(
                {
                    "question": "How do I export attribution reports?",
                    "customer_wording": "export reports",
                    "ticket_count": 2,
                    "opportunity_score": 7,
                    "answer": "Open Analytics, choose Attribution, then Download report.",
                    "answer_evidence_status": "resolution_evidence",
                    "steps": ("Open Analytics and choose Attribution.",),
                    "source_ids": ("ticket-export-1", "ticket-export-2"),
                    "evidence_quotes": (
                        "`ticket-export-1` - Export attribution\n"
                        "### Error details pasted from the ticket\n"
                        "This raw evidence line should not render in the curated PDF.",
                    ),
                },
                {
                    "question": "How do I enable SSO?",
                    "customer_wording": "enable SSO",
                    "ticket_count": 1,
                    "answer_evidence_status": "draft_needs_review",
                    "source_ids": ("ticket-sso-1",),
                    "evidence_quotes": ("`ticket-sso-1` - SSO setup",),
                },
            ),
        )
    )

    curated = _curate_markdown_for_pdf(artifact.markdown)
    entries = _toc_entries(curated)

    assert (2, "Ranked Question Opportunities") in entries
    assert (2, "Question Details and Evidence") in entries
    assert (3, "1. How do I export attribution reports?") in entries
    assert "complete evidence export JSON" in curated
    assert "**Source IDs (full list):**" not in curated
    assert "ticket-export-1" not in curated
    assert "Error details pasted from the ticket" not in curated
    assert "This raw evidence line should not render in the curated PDF." not in curated
    assert "Open Analytics, choose Attribution" in curated
    assert "How do I enable SSO?" in curated


def test_plain_toc_uses_curated_headings_only() -> None:
    curated = _curate_markdown_for_pdf(_large_markdown())
    entries = _toc_entries(curated)

    assert (2, "Support Tax Confirmation") in entries
    assert (2, "Ranked Question Opportunities") in entries
    assert (3, f"{PDF_QUESTION_DETAIL_LIMIT}. Question {PDF_QUESTION_DETAIL_LIMIT}") in entries
    assert (
        3,
        f"{PDF_QUESTION_DETAIL_LIMIT + 1}. Question {PDF_QUESTION_DETAIL_LIMIT + 1}",
    ) not in entries


def test_render_deflection_full_report_pdf_requires_markdown() -> None:
    with pytest.raises(ValueError, match="artifact markdown is required"):
        render_deflection_full_report_pdf({"summary": {"generated": 0}})
