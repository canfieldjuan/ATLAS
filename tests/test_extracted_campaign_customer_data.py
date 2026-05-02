from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from extracted_content_pipeline.campaign_customer_data import (
    FileIntelligenceRepository,
    load_campaign_opportunities_from_file,
    normalize_campaign_opportunity_rows,
)
from extracted_content_pipeline.campaign_ports import TenantScope


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/run_extracted_campaign_generation_example.py"


def test_load_campaign_opportunities_from_json_payload_preserves_custom_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(
        json.dumps({
            "opportunities": [
                {
                    "id": "opp-1",
                    "company": "Acme",
                    "vendor": "HubSpot",
                    "email": "buyer@example.com",
                    "title": "VP Revenue",
                    "pain_category": "pricing",
                    "competitor": "Salesforce, Zoho",
                    "custom_segment": "enterprise",
                }
            ]
        }),
        encoding="utf-8",
    )

    loaded = load_campaign_opportunities_from_file(
        path,
        target_mode="vendor_retention",
    )

    assert loaded.source == str(path)
    assert loaded.warnings == ()
    row = loaded.opportunities[0]
    assert row["target_id"] == "opp-1"
    assert row["company_name"] == "Acme"
    assert row["vendor_name"] == "HubSpot"
    assert row["contact_email"] == "buyer@example.com"
    assert row["contact_title"] == "VP Revenue"
    assert row["pain_points"] == ["pricing"]
    assert row["competitors"] == ["Salesforce", "Zoho"]
    assert row["custom_segment"] == "enterprise"
    assert row["target_mode"] == "vendor_retention"


def test_load_campaign_opportunities_from_csv_coerces_json_cells_and_warns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "opportunities.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,email,pain_category,evidence",
            'opp-1,Acme,HubSpot,buyer@example.com,pricing,"[{""quote"": ""Too expensive""}]"',
            "opp-2,BetaCRM,,ops@example.com,,",
        ]),
        encoding="utf-8",
    )

    loaded = load_campaign_opportunities_from_file(path)

    assert len(loaded.opportunities) == 2
    assert loaded.opportunities[0]["evidence"] == [{"quote": "Too expensive"}]
    assert loaded.opportunities[1]["company_name"] == "BetaCRM"
    assert [warning.code for warning in loaded.warnings] == ["missing_vendor_name"]
    assert loaded.warnings[0].row_index == 2


def test_normalize_campaign_opportunity_rows_skips_non_object_rows() -> None:
    loaded = normalize_campaign_opportunity_rows(
        [
            {"id": "opp-1", "company": "Acme", "vendor": "HubSpot"},
            "bad-row",
        ],
        target_mode="vendor_retention",
    )

    assert len(loaded.opportunities) == 1
    assert loaded.opportunities[0]["target_id"] == "opp-1"
    assert loaded.warnings[-1].code == "row_not_object"
    assert loaded.warnings[-1].row_index == 2


@pytest.mark.asyncio
async def test_file_intelligence_repository_filters_and_limits_rows(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(
        json.dumps([
            {"id": "opp-1", "company": "Acme", "vendor": "HubSpot"},
            {"id": "opp-2", "company": "Beta", "vendor": "Zendesk"},
            {"id": "opp-3", "company": "Gamma", "vendor": "HubSpot"},
        ]),
        encoding="utf-8",
    )
    repo = FileIntelligenceRepository.from_file(path)

    rows = await repo.read_campaign_opportunities(
        scope=TenantScope(account_id="acct-1"),
        target_mode="vendor_retention",
        limit=1,
        filters={"vendor_name": "HubSpot"},
    )

    assert [row["target_id"] for row in rows] == ["opp-1"]
    assert rows[0]["target_mode"] == "vendor_retention"


def test_campaign_generation_cli_accepts_csv_customer_export() -> None:
    path = ROOT / "extracted_content_pipeline/examples/campaign_generation_payload.csv"

    completed = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--format",
            "csv",
            "--limit",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert result["result"]["generated"] == 1
    assert result["source"].endswith("campaign_generation_payload.csv")
    assert result["drafts"][0]["target_id"] == "opp-acme-hubspot"
    assert result["drafts"][0]["metadata"]["source_opportunity"]["custom_segment"] == (
        "enterprise logistics"
    )


def test_campaign_generation_cli_surfaces_customer_data_warnings(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.csv"
    path.write_text(
        "\n".join([
            "id,company,vendor,email,pain_category",
            "opp-1,Acme,,buyer@example.com,pricing",
        ]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, str(CLI), str(path), "--format", "csv"],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)

    assert result["result"]["generated"] == 1
    assert result["opportunity_warnings"][0]["code"] == "missing_vendor_name"
    assert result["opportunity_warnings"][0]["row_index"] == 1
