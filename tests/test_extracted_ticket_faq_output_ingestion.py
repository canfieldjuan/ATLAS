from __future__ import annotations

from extracted_content_pipeline.campaign_source_adapters import (
    source_material_to_source_rows,
    source_rows_to_campaign_opportunities,
)
from extracted_content_pipeline.faq_output_ingestion import (
    FAQ_OUTPUT_SOURCE_TYPE,
    faq_output_to_source_rows,
    is_faq_output_bundle,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


def test_faq_output_rows_normalize_into_campaign_opportunities() -> None:
    faq_output = {
        "generated": 2,
        "markdown": "# FAQ",
        "saved_ids": ["faq-draft-1"],
        "items": [
            {
                "topic": "billing confusion",
                "question": "Why was I charged twice?",
                "question_source": "customer_wording",
                "summary": "Customers ask why duplicate-looking invoices appear.",
                "steps": (
                    "Check whether the second charge is a pending authorization.",
                    "Confirm the invoice date and subscription workspace.",
                ),
                "evidence_quotes": (
                    '`ticket-1` - Billing: "Why was I charged twice?"',
                    '`ticket-2` - Invoice: "I see two charges this month."',
                ),
                "source_ids": ("ticket-1", "ticket-2"),
                "answer_evidence_status": "resolution_evidence",
                "frequency": 2,
                "weighted_frequency": 2,
                "failure_risk_score": 1,
                "opportunity_score": 4,
                "ticket_count": 2,
                "evidence_count": 2,
                "resolution_source_count": 1,
            },
            {
                "topic": "export setup",
                "question": "How do I export the report?",
                "summary": "Customers ask where report exports live.",
                "steps": ("Open Reports.",),
                "source_ids": ("ticket-3",),
                "answer_evidence_status": "draft_needs_review",
            },
        ],
    }

    rows = faq_output_to_source_rows(faq_output)
    loaded = source_rows_to_campaign_opportunities(
        rows,
        target_mode="landing_page",
    )

    assert [row["source_type"] for row in rows] == [
        FAQ_OUTPUT_SOURCE_TYPE,
        FAQ_OUTPUT_SOURCE_TYPE,
    ]
    assert rows[0]["source_id"] == "faq-draft-1:item-1"
    assert rows[1]["source_id"] == "faq-draft-1:item-2"
    assert rows[0]["faq_draft_id"] == "faq-draft-1"
    assert rows[1]["faq_draft_id"] == "faq-draft-1"
    assert rows[0]["source_title"] == "Why was I charged twice?"
    assert rows[0]["faq_source_ticket_ids"] == ["ticket-1", "ticket-2"]
    assert rows[0]["faq_customer_language"] == [
        "Why was I charged twice?",
        '`ticket-1` - Billing: "Why was I charged twice?"',
        '`ticket-2` - Invoice: "I see two charges this month."',
    ]
    assert "FAQ question: Why was I charged twice?" in rows[0]["text"]
    assert "Customer wording:" in rows[0]["text"]
    warning_codes = [warning.code for warning in loaded.warnings]
    assert "missing_source_text" not in warning_codes
    first = loaded.opportunities[0]
    assert first["target_mode"] == "landing_page"
    assert first["source_type"] == FAQ_OUTPUT_SOURCE_TYPE
    assert first["source_title"] == "Why was I charged twice?"
    assert first["faq_answer_evidence_status"] == "resolution_evidence"
    assert first["pain_points"] == ["billing confusion"]
    assert first["opportunity_score"] == 4
    assert first["evidence"] == [{
        "text": rows[0]["text"],
        "source_id": "faq-draft-1:item-1",
        "source_type": FAQ_OUTPUT_SOURCE_TYPE,
        "source_title": "Why was I charged twice?",
    }]


def test_source_material_to_source_rows_accepts_ticket_faq_result_dict() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "source_title": "Invoice question",
                "source_id": "ticket-1",
                "evidence": [{
                    "text": "Why was I charged twice this month?",
                    "source_id": "ticket-1",
                    "source_type": "support_ticket",
                    "resolution_text": (
                        "Open Billing, review the invoice history, and compare "
                        "pending authorizations with settled charges."
                    ),
                }],
            }
        ]
    )

    rows = source_material_to_source_rows(result.as_dict())
    loaded = source_rows_to_campaign_opportunities(rows, target_mode="blog_post")

    assert len(rows) == 1
    assert rows[0]["source_type"] == FAQ_OUTPUT_SOURCE_TYPE
    assert rows[0]["faq_answer_evidence_status"] == "resolution_evidence"
    assert "Why was I charged twice this month?" in rows[0]["text"]
    warning_codes = [warning.code for warning in loaded.warnings]
    assert "missing_source_text" not in warning_codes
    assert loaded.opportunities[0]["source_type"] == FAQ_OUTPUT_SOURCE_TYPE
    assert loaded.opportunities[0]["target_mode"] == "blog_post"


def test_faq_output_adapter_ignores_non_faq_and_empty_items() -> None:
    assert is_faq_output_bundle({"generated": 0, "items": []})
    assert faq_output_to_source_rows({"generated": 0, "items": []}) == []
    assert faq_output_to_source_rows({"generated": 1, "items": [{}]}) == []
    assert not is_faq_output_bundle({"items": [{"question": "Not a FAQ result"}]})
    assert faq_output_to_source_rows({"items": [{"question": "Not a FAQ result"}]}) == []
