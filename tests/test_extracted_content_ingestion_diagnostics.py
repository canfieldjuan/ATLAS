from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from extracted_content_pipeline.ingestion_diagnostics import inspect_ingestion_file


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts/inspect_extracted_content_ingestion.py"


def test_inspect_opportunity_json_reports_ready_rows(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": "opp-1",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "email": "buyer@example.com",
            "evidence": [{"text": "Renewal pricing is under review."}],
        }
    ]))

    report = inspect_ingestion_file(path, sample_limit=1)
    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["mode"] == "opportunities"
    assert payload["opportunity_count"] == 1
    assert payload["warning_count"] == 0
    assert payload["missing_field_counts"] == {}
    assert payload["samples"][0]["target_id"] == "opp-1"


def test_inspect_source_rows_counts_source_types_and_warnings(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "ticket_id": "ticket-1",
                "company": "Acme Logistics",
                "vendor": "HubSpot",
                "message": "Renewal quote jumped before export work finished.",
            }),
            json.dumps({
                "id": "empty-1",
                "company": "Beta",
                "vendor": "Zendesk",
            }),
        ])
    )

    report = inspect_ingestion_file(
        path,
        source_rows=True,
        source_format="jsonl",
        sample_limit=2,
    )
    payload = report.as_dict()

    assert payload["ok"] is True
    assert payload["mode"] == "source_rows"
    assert payload["opportunity_count"] == 1
    assert payload["warning_counts"] == {
        "missing_contact_email": 1,
        "missing_source_text": 1,
    }
    assert payload["source_type_counts"] == {"support_ticket": 1}
    assert payload["samples"][0]["evidence"][0]["source_type"] == "support_ticket"


def test_inspect_reports_missing_generation_fields(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": "opp-1",
            "company": "Acme Logistics",
        }
    ]))

    payload = inspect_ingestion_file(path).as_dict()

    assert payload["ok"] is True
    assert payload["missing_field_counts"] == {
        "evidence": 1,
        "vendor_name": 1,
    }


def test_inspect_sample_limit_bounds_output(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text(json.dumps([
        {
            "id": f"opp-{index}",
            "company": "Acme Logistics",
            "vendor": "HubSpot",
            "evidence": [{"text": "Renewal pricing is under review."}],
        }
        for index in range(3)
    ]))

    payload = inspect_ingestion_file(path, sample_limit=2).as_dict()

    assert payload["opportunity_count"] == 3
    assert [sample["target_id"] for sample in payload["samples"]] == ["opp-0", "opp-1"]


def test_inspection_cli_emits_valid_json_for_source_rows(tmp_path: Path) -> None:
    path = tmp_path / "sources.jsonl"
    path.write_text(json.dumps({
        "review_id": "review-1",
        "reviewer_company": "Acme Logistics",
        "vendor": "HubSpot",
        "review_text": "Pricing is hard to justify.",
    }))

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--source-rows",
            "--source-format",
            "jsonl",
            "--sample-limit",
            "1",
            "--json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["ok"] is True
    assert payload["source_type_counts"] == {"review": 1}
    assert len(payload["samples"]) == 1


def test_inspection_cli_rejects_negative_sample_limit(tmp_path: Path) -> None:
    path = tmp_path / "opportunities.json"
    path.write_text("[]")

    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(path),
            "--sample-limit",
            "-1",
            "--json",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert proc.returncode != 0
    assert "--sample-limit must be non-negative" in proc.stderr
