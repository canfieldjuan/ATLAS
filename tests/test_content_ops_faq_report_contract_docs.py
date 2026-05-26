import json
from pathlib import Path

from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs/frontend/content_ops_faq_report_contract.md"
EXAMPLE_PATH = ROOT / "docs/frontend/content_ops_faq_report_example.json"


def _producer_report_shape() -> tuple[set[str], set[str]]:
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "search-export-1",
                "source_type": "search_log",
                "text": "How do I export attribution report?",
                "zero_results": "true",
                "source_weight": "20",
            },
            {
                "source_id": "ticket-email-1",
                "source_type": "support_ticket",
                "source_title": "Email update",
                "text": "How do I change my email?",
                "resolution_text": (
                    "Open account settings, choose Profile, update the email "
                    "address, then confirm the verification email"
                ),
            },
        ],
        title="Support Ticket FAQ Report",
        max_items=2,
        max_evidence_per_item=2,
        support_contact="https://example.com/support",
        documentation_terms=("Download report",),
        vocabulary_gap_rules=(("export", "download report"),),
    )

    assert result.items
    return set(result.as_dict()), set(result.items[0])


def test_content_ops_faq_report_example_matches_documented_core_shape() -> None:
    payload = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    producer_payload_keys, producer_item_keys = _producer_report_shape()

    assert set(payload) == producer_payload_keys
    assert payload["generated"] == len(payload["items"])
    assert isinstance(payload["markdown"], str) and payload["markdown"].startswith("# ")
    assert payload["source_count"] >= payload["ticket_source_count"] >= 1
    assert payload["output_checks"] == {
        "condensed": True,
        "has_action_items": True,
        "uses_user_vocabulary": True,
    }

    required_item_keys = {
        "answer_evidence_status",
        "evidence_quotes",
        "failure_risk_signals",
        "opportunity_score",
        "question",
        "question_source",
        "source_ids",
        "steps",
        "summary",
        "term_mappings",
        "topic",
        "when_to_contact_support",
    }
    for item in payload["items"]:
        assert set(item) == producer_item_keys
        assert required_item_keys <= set(item)
        assert item["question_source"] in {"customer_wording", "source_policy"}
        assert item["answer_evidence_status"] in {
            "draft_needs_review",
            "resolution_evidence",
        }
        assert item["steps"]
        assert item["source_ids"]


def test_content_ops_faq_report_contract_links_example() -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")

    assert "content_ops_faq_report_example.json" in doc
    assert "account_id: string;" in doc
    assert EXAMPLE_PATH.exists()
