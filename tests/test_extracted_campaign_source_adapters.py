from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from extracted_content_pipeline.campaign_source_adapters import (
    load_source_campaign_opportunities_from_file,
    source_row_to_campaign_opportunity,
    source_rows_to_campaign_opportunities,
)


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/build_extracted_campaign_opportunities_from_sources.py"
EXAMPLE_SOURCE_ROWS = (
    ROOT / "extracted_content_pipeline/examples/campaign_source_rows.jsonl"
)


def test_source_row_maps_review_text_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "review_id": "review-1",
        "reviewer_company": "Acme Logistics",
        "vendor": "HubSpot",
        "review_text": "Pricing became hard to justify after renewal.",
        "pain_category": "pricing pressure",
        "contact_email": "buyer@example.com",
    })

    assert warnings == ()
    assert opportunity["id"] == "review-1"
    assert opportunity["target_id"] == "review-1"
    assert opportunity["source_id"] == "review-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor"] == "HubSpot"
    assert opportunity["pain_points"] == ["pricing pressure"]
    assert opportunity["evidence"] == [{
        "text": "Pricing became hard to justify after renewal.",
        "source_id": "review-1",
        "source_type": "review",
    }]


def test_source_rows_normalize_and_warn_for_missing_text() -> None:
    loaded = source_rows_to_campaign_opportunities(
        [
            {
                "id": "call-1",
                "company": "Acme",
                "vendor": "HubSpot",
                "transcript": "We are reviewing Salesforce before renewal.",
                "pain_points": ["renewal pressure"],
            },
            {"id": "empty-1", "company": "Beta", "vendor": "Zendesk"},
        ],
        target_mode="vendor_retention",
    )

    assert len(loaded.opportunities) == 1
    first = loaded.opportunities[0]
    assert first["target_id"] == "call-1"
    assert first["target_mode"] == "vendor_retention"
    assert first["evidence"][0]["source_type"] == "transcript"
    assert "missing_source_text" in [warning.code for warning in loaded.warnings]


def test_source_document_title_does_not_become_buyer_fields() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "id": "doc-1",
        "name": "Q1 churn transcript",
        "title": "Discovery call notes",
        "vendor": "HubSpot",
        "text": "The account is comparing HubSpot with Salesforce.",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "doc-1"
    assert opportunity["source_title"] == "Discovery call notes"
    assert opportunity["evidence"][0]["source_title"] == "Discovery call notes"
    assert "company_name" not in opportunity
    assert "contact_name" not in opportunity
    assert "contact_title" not in opportunity
    assert "name" not in opportunity
    assert "title" not in opportunity


def test_source_row_maps_support_ticket_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "subject": "Renewal pricing escalation",
        "message": "The renewal quote jumped and support has not explained the new tier.",
        "pain_category": "pricing pressure",
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ticket-1"
    assert opportunity["company_name"] == "Acme Logistics"
    assert opportunity["vendor"] == "HubSpot"
    assert opportunity["source_title"] == "Renewal pricing escalation"
    assert "subject" not in opportunity
    assert opportunity["evidence"] == [{
        "text": "The renewal quote jumped and support has not explained the new tier.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Renewal pricing escalation",
    }]


def test_source_row_prefers_body_over_ticket_message() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "body": "Use this longer exported body as the evidence text.",
        "message": "Do not use this short ticket message when body exists.",
    })

    assert warnings == ()
    assert opportunity["evidence"][0]["text"] == (
        "Use this longer exported body as the evidence text."
    )


def test_source_row_maps_ticket_comments_to_evidence_text() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-thread-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "subject": "Reporting export blocked",
        "comments": [
            {
                "author": "Customer",
                "body": "We cannot export attribution data before renewal.",
            },
            {
                "role": "Agent",
                "message": "The export requires a higher plan.",
            },
        ],
    })

    assert warnings == ()
    assert opportunity["target_id"] == "ticket-thread-1"
    assert opportunity["evidence"][0] == {
        "text": (
            "Customer: We cannot export attribution data before renewal.\n"
            "Agent: The export requires a higher plan."
        ),
        "source_id": "ticket-thread-1",
        "source_type": "support_ticket",
        "source_title": "Reporting export blocked",
    }


def test_source_row_prefers_scalar_text_over_thread_messages() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "ticket_id": "ticket-thread-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "message": "Use this exported ticket summary.",
        "comments": [{"body": "Do not use this comment when scalar text exists."}],
    })

    assert warnings == ()
    assert opportunity["evidence"][0]["text"] == "Use this exported ticket summary."


def test_source_row_maps_nps_feedback_to_campaign_opportunity() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "response_id": "nps-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "nps_score": 3,
        "feedback_text": "Renewal pricing is hard to defend against Salesforce.",
        "pain_category": "pricing pressure",
    })

    assert warnings == ()
    assert opportunity["id"] == "nps-1"
    assert opportunity["target_id"] == "nps-1"
    assert opportunity["nps_score"] == 3
    assert opportunity["source_type"] == "nps_response"
    assert opportunity["pain_points"] == ["pricing pressure"]
    assert opportunity["evidence"] == [{
        "text": "Renewal pricing is hard to defend against Salesforce.",
        "source_id": "nps-1",
        "source_type": "nps_response",
    }]


def test_source_row_prefers_nps_over_csat_when_both_scores_exist() -> None:
    opportunity, warnings = source_row_to_campaign_opportunity({
        "response_id": "survey-1",
        "company": "Acme Logistics",
        "vendor": "HubSpot",
        "nps_score": 5,
        "csat_score": 4,
        "feedback": "The team is still evaluating alternatives.",
    })

    assert warnings == ()
    assert opportunity["source_type"] == "nps_response"


def test_load_source_campaign_opportunities_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "id": "review-1",
                "company": "Acme",
                "vendor": "HubSpot",
                "review_text": "Pricing is a problem.",
            }),
            json.dumps({
                "id": "complaint-1",
                "company": "Beta",
                "vendor": "Zendesk",
                "complaint": "Support tickets are taking too long.",
            }),
        ]),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.source == str(path)
    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "complaint-1",
    ]
    assert loaded.opportunities[1]["evidence"][0]["source_type"] == "complaint"


def test_load_source_campaign_opportunities_from_csv(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,review_text,pain_category",
            "review-1,Acme,HubSpot,Pricing is a problem,pricing",
        ]),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "review-1"
    assert loaded.opportunities[0]["company_name"] == "Acme"
    assert loaded.opportunities[0]["pain_points"] == ["pricing"]
    assert loaded.opportunities[0]["evidence"] == [{
        "text": "Pricing is a problem",
        "source_id": "review-1",
        "source_type": "review",
    }]


def test_packaged_source_rows_example_loads() -> None:
    loaded = load_source_campaign_opportunities_from_file(EXAMPLE_SOURCE_ROWS)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-acme-1",
        "transcript-northstar-1",
        "ticket-riverbend-1",
    ]
    assert loaded.opportunities[0]["evidence"][0]["source_type"] == "review"
    assert loaded.opportunities[1]["evidence"][0]["source_type"] == "transcript"
    assert loaded.opportunities[2]["evidence"][0]["source_type"] == "support_ticket"
    assert loaded.warnings == ()


def test_load_source_campaign_opportunities_from_support_ticket_object(tmp_path: Path) -> None:
    path = tmp_path / "support_tickets.json"
    path.write_text(
        json.dumps({
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "company": "Acme",
                    "vendor": "HubSpot",
                    "subject": "Reporting export issue",
                    "description": "The team cannot export attribution data before renewal.",
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "ticket-1"
    assert loaded.opportunities[0]["source_type"] == "support_ticket"
    assert loaded.opportunities[0]["source_title"] == "Reporting export issue"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "The team cannot export attribution data before renewal.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Reporting export issue",
    }


def test_load_source_campaign_opportunities_from_survey_object(tmp_path: Path) -> None:
    path = tmp_path / "survey_responses.json"
    path.write_text(
        json.dumps({
            "survey_responses": [
                {
                    "survey_id": "survey-1",
                    "company": "Beta Retail",
                    "vendor": "Zendesk",
                    "csat_score": "2",
                    "open_ended_response": (
                        "Support responses take too long and the team is comparing Intercom."
                    ),
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["target_id"] == "survey-1"
    assert loaded.opportunities[0]["csat_score"] == "2"
    assert loaded.opportunities[0]["source_type"] == "csat_response"
    assert loaded.opportunities[0]["evidence"][0] == {
        "text": "Support responses take too long and the team is comparing Intercom.",
        "source_id": "survey-1",
        "source_type": "csat_response",
    }


def test_load_source_campaign_opportunities_from_multi_collection_bundle(tmp_path: Path) -> None:
    path = tmp_path / "customer_bundle.json"
    path.write_text(
        json.dumps({
            "account_id": "acct-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "reviews": [
                {
                    "review_id": "review-1",
                    "review_text": "Renewal pricing is hard to defend.",
                }
            ],
            "support_tickets": [
                {
                    "ticket_id": "ticket-1",
                    "subject": "Reporting exports blocked",
                    "description": "Exports require a higher tier before renewal.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "ticket-1",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Acme Logistics",
        "Acme Logistics",
    ]
    assert [row["vendor"] for row in loaded.opportunities] == ["HubSpot", "HubSpot"]
    assert loaded.opportunities[0]["account_id"] == "acct-1"
    assert loaded.opportunities[0]["evidence"][0]["source_type"] == "review"
    assert loaded.opportunities[1]["evidence"][0] == {
        "text": "Exports require a higher tier before renewal.",
        "source_id": "ticket-1",
        "source_type": "support_ticket",
        "source_title": "Reporting exports blocked",
    }


def test_load_source_campaign_opportunities_from_nested_sources_bundle(tmp_path: Path) -> None:
    path = tmp_path / "nested_sources.json"
    path.write_text(
        json.dumps({
            "company": "Beta Retail",
            "vendor": "Zendesk",
            "sources": {
                "reviews": [
                    {
                        "id": "review-1",
                        "review_text": "The team is comparing Intercom.",
                    }
                ],
                "surveys": [
                    {
                        "response_id": "survey-1",
                        "csat_score": 2,
                        "open_ended_response": "Support takes too long.",
                    }
                ],
            },
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert [row["target_id"] for row in loaded.opportunities] == [
        "review-1",
        "survey-1",
    ]
    assert [row["company_name"] for row in loaded.opportunities] == [
        "Beta Retail",
        "Beta Retail",
    ]
    assert loaded.opportunities[0]["source_type"] == "review"
    assert loaded.opportunities[1]["source_type"] == "csat_response"


def test_source_bundle_child_fields_override_parent_metadata(tmp_path: Path) -> None:
    path = tmp_path / "child_override.json"
    path.write_text(
        json.dumps({
            "company": "Parent Co",
            "vendor": "HubSpot",
            "reviews": [
                {
                    "id": "review-1",
                    "company": "Child Co",
                    "vendor": "Salesforce",
                    "review_text": "The team is switching platforms.",
                }
            ],
        }),
        encoding="utf-8",
    )

    loaded = load_source_campaign_opportunities_from_file(path)

    assert loaded.opportunities[0]["company_name"] == "Child Co"
    assert loaded.opportunities[0]["vendor"] == "Salesforce"


def test_source_adapter_cli_outputs_generation_payload(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps({
            "reviews": [
                {
                    "id": "review-1",
                    "company": "Acme",
                    "vendor": "HubSpot",
                    "review_text": "Pricing is a problem.",
                }
            ]
        }),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--target-mode",
            "vendor_retention",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["target_mode"] == "vendor_retention"
    assert payload["limit"] == 1
    assert payload["opportunities"][0]["target_id"] == "review-1"
    assert payload["opportunities"][0]["evidence"][0]["text"] == "Pricing is a problem."


def test_source_adapter_cli_outputs_csv_generation_payload(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,transcript,pain_points",
            "call-1,Acme,HubSpot,Renewal review is active,renewal pressure",
        ]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "csv",
            "--target-mode",
            "vendor_retention",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["opportunities"][0]["target_id"] == "call-1"
    assert payload["opportunities"][0]["evidence"][0]["source_type"] == "transcript"


def test_source_adapter_cli_rejects_non_positive_text_limit(tmp_path: Path) -> None:
    path = tmp_path / "sources.json"
    path.write_text("[]", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--max-text-chars",
            "0",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "--max-text-chars must be positive" in completed.stderr
