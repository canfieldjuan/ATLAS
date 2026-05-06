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
