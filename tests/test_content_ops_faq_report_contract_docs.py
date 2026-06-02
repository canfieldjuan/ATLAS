import json
from pathlib import Path

from extracted_content_pipeline.faq_deflection_report import (
    build_deflection_snapshot,
    build_deflection_report_artifact,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs/frontend/content_ops_faq_report_contract.md"
EXAMPLE_PATH = ROOT / "docs/frontend/content_ops_faq_report_example.json"
DEFLECTION_EXAMPLE_PATH = (
    ROOT / "docs/frontend/content_ops_faq_deflection_report_example.json"
)
DEFLECTION_SNAPSHOT_EXAMPLE_PATH = (
    ROOT / "docs/frontend/content_ops_faq_deflection_snapshot_example.json"
)
DEFLECTION_CHECKOUT_CONTRACT_PATH = (
    ROOT / "docs/frontend/content_ops_faq_deflection_checkout_contract.md"
)


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


def _producer_deflection_report_payload() -> dict[str, object]:
    result = build_ticket_faq_markdown(
        [
            {
                "source_id": "ticket-export-1",
                "source_type": "support_ticket",
                "source_title": "Export attribution",
                "text": "How do I export attribution reports?",
                "resolution_text": (
                    "Open Analytics, choose Attribution, then click Download report"
                ),
            },
            {
                "source_id": "ticket-export-2",
                "source_type": "support_ticket",
                "source_title": "Report download",
                "text": "Where is the report download for attribution exports?",
                "resolution_text": (
                    "Open Analytics, choose Attribution, then click Download report"
                ),
            },
            {
                "source_id": "ticket-sso-1",
                "source_type": "support_ticket",
                "source_title": "SSO setup",
                "text": "How do I enable SSO for my team?",
            },
            {
                "source_id": "ticket-sso-2",
                "source_type": "support_ticket",
                "source_title": "Team login",
                "text": "Can I turn on SSO for all users?",
            },
        ],
        title="Support Ticket FAQ Source",
        max_items=2,
        max_evidence_per_item=1,
        support_contact="https://example.com/support",
        documentation_terms=("Download report", "Single sign-on setup"),
        vocabulary_gap_rules=(
            ("export", "Download report"),
            ("SSO", "Single sign-on setup"),
            ("report download", "Download report"),
        ),
    )

    return build_deflection_report_artifact(result).as_dict()


def test_content_ops_faq_report_example_matches_documented_core_shape() -> None:
    payload = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    producer_payload_keys, producer_item_keys = _producer_report_shape()
    encoded = json.dumps(payload, sort_keys=True)

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
    assert "Use the uploaded resolution evidence" not in encoded
    assert "Customers mention:" not in encoded
    assert "Confirm the answer matches" not in encoded


def test_content_ops_faq_deflection_example_matches_producer_shape() -> None:
    payload = json.loads(DEFLECTION_EXAMPLE_PATH.read_text(encoding="utf-8"))
    producer_payload = _producer_deflection_report_payload()
    encoded = json.dumps(payload, sort_keys=True)

    assert set(payload) == set(producer_payload)
    assert set(payload["summary"]) == set(producer_payload["summary"])
    assert set(payload["faq_result"]) == set(producer_payload["faq_result"])
    assert payload["summary"]["drafted_answer_count"] == 1
    assert payload["summary"]["no_proven_answer_count"] == 1
    assert payload["summary"]["generated"] == len(payload["faq_result"]["items"])
    assert payload["summary"]["output_checks"] == {
        "condensed": True,
        "has_action_items": True,
        "uses_user_vocabulary": True,
    }
    assert all(payload["faq_result"]["output_checks"].values())
    assert "## Drafted Answers With Proven Solutions" in payload["markdown"]
    assert "## No Proven Answer Yet" in payload["markdown"]
    assert "Use the uploaded resolution evidence" not in encoded
    assert "Customers mention:" not in encoded
    assert "Confirm the answer matches" not in encoded


def test_content_ops_faq_deflection_snapshot_example_matches_producer_shape() -> None:
    payload = json.loads(DEFLECTION_SNAPSHOT_EXAMPLE_PATH.read_text(encoding="utf-8"))
    producer_payload = build_deflection_snapshot(
        _producer_deflection_report_payload(),
        top_n=2,
    ).as_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload == producer_payload
    assert set(payload) == {"summary", "top_questions"}
    assert set(payload["summary"]) == {
        "generated",
        "drafted_answer_count",
        "no_proven_answer_count",
    }
    for question in payload["top_questions"]:
        assert set(question) == {
            "rank",
            "question",
            "weighted_frequency",
            "customer_wording",
        }
    assert "markdown" not in encoded
    assert "faq_result" not in encoded
    assert "steps" not in encoded
    assert "evidence" not in encoded
    assert "source_ids" not in encoded


def test_content_ops_faq_report_contract_links_example() -> None:
    doc = DOC_PATH.read_text(encoding="utf-8")

    assert "content_ops_faq_report_example.json" in doc
    assert "content_ops_faq_deflection_report_example.json" in doc
    assert "content_ops_faq_deflection_snapshot_example.json" in doc
    assert "content_ops_faq_deflection_checkout_contract.md" in doc
    assert "type DeflectionSnapshot" in doc
    assert "account_id: string;" in doc
    assert EXAMPLE_PATH.exists()
    assert DEFLECTION_EXAMPLE_PATH.exists()
    assert DEFLECTION_SNAPSHOT_EXAMPLE_PATH.exists()
    assert DEFLECTION_CHECKOUT_CONTRACT_PATH.exists()


def test_content_ops_faq_deflection_checkout_contract_pins_paid_handoff() -> None:
    doc = DEFLECTION_CHECKOUT_CONTRACT_PATH.read_text(encoding="utf-8")

    required_terms = {
        'source: "content_ops_deflection_report"',
        "account_id: string;",
        "request_id: string;",
        "GET /content-ops/deflection-reports/{request_id}/snapshot",
        "GET /content-ops/deflection-reports/{request_id}/artifact",
        "csv_file: File;",
        "do not expose raw support-ticket CSVs through a",
        "POST /content-ops/deflection-reports/{request_id}/paid",
        "amount_total >= 150000",
        'currency: "usd"',
        "The portfolio does not call an authed \"mark paid\" endpoint",
        "Stripe webhook -> ATLAS verifies -> ATLAS marks paid",
    }

    for term in required_terms:
        assert term in doc
