from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_deflection_signal_spike.py"
ZENDESK_THREAD_SAMPLE = ROOT / "tests/fixtures/zendesk_full_thread_seed_sample.json"
SPEC = importlib.util.spec_from_file_location("probe_deflection_signal_spike", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_signal_spike_reports_s1_availability_without_raw_source_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tickets.csv"
    rows = [
        {
            "Ticket ID": "SUP-123456",
            "Subject": "Requester name is Jane Client",
            "Description": "Jane Client asked from jane.client@example.com about exports.",
            "Resolution": "Open Reports, choose Export, then select CSV.",
            "Ticket Status": "Closed",
            "Customer Satisfaction Rating": "good",
            "Product": "Analytics",
            "Issue": "Exports",
            "support_cost": "27.50",
        },
        {
            "Ticket ID": "SUP-999999",
            "Subject": "Session token is ABCD1234EFGH",
            "Description": "Where do we reset account 1234567?",
            "Ticket Status": "Reopened",
            "Customer Satisfaction Rating": "2",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = MODULE.probe_signal_availability(path, source_format="csv")
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["schema_version"] == MODULE.SCHEMA_VERSION
    assert summary["rows"]["included_row_count"] == 2
    assert summary["signals"]["support_resolution_evidence"] == {
        "readiness": "ready",
        "reason": "resolution evidence is present",
        "rows_with_resolution_evidence": 1,
        "coverage": 0.5,
    }
    assert summary["signals"]["ticket_status"]["readiness"] == "ready"
    assert summary["signals"]["ticket_status"]["summary"] == {
        "reopened": 1,
        "resolved": 1,
    }
    assert summary["signals"]["csat"]["readiness"] == "partial"
    assert summary["signals"]["csat"]["basis"] == "mixed"
    assert summary["signals"]["cost_basis"]["readiness"] == "partial"
    assert summary["signals"]["cost_basis"]["basis"] == "source_fields_present"
    assert summary["signals"]["owner_lane_fix_type"]["readiness"] == "partial"
    assert summary["signals"]["owner_lane_fix_type"]["unknown_fallback_required"] is True
    assert summary["signals"]["snippet_projection_safety"]["readiness"] == "partial"
    assert summary["signals"]["snippet_projection_safety"]["text_fields_changed_by_scrub"] >= 2
    assert summary["safety"] == {
        "summary_only": True,
        "raw_source_values_recorded": False,
        "source_ids_recorded": False,
        "snippets_recorded": False,
    }

    assert "SUP-123456" not in rendered
    assert "Jane Client" not in rendered
    assert "jane.client@example.com" not in rendered
    assert "ABCD1234EFGH" not in rendered


def test_signal_spike_require_s1_ready_fails_on_insufficient_data(
    tmp_path: Path,
    capsys,
) -> None:
    path = tmp_path / "tickets.json"
    path.write_text(
        json.dumps([
            {
                "ticket_id": "ticket-1",
                "subject": "How do I reset my password?",
                "description": "I cannot find the reset page.",
            }
        ]),
        encoding="utf-8",
    )

    status = MODULE.main([str(path), "--require-s1-ready"])

    assert status == 1
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["signals"]["support_resolution_evidence"]["readiness"] == (
        "insufficient_data"
    )
    assert output["signals"]["csat"]["readiness"] == "insufficient_data"
    assert "support_resolution_evidence" in captured.err
    assert "csat_prioritization" in captured.err
    assert "ticket-1" not in captured.out
    assert "reset page" not in captured.out


def test_signal_spike_uses_zendesk_thread_import_without_warning_leaks() -> None:
    summary = MODULE.probe_signal_availability(
        ZENDESK_THREAD_SAMPLE,
        source_format="json",
        zendesk_thread="auto",
    )
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["input"]["zendesk_thread_import"] is True
    assert summary["rows"]["included_row_count"] == 4
    assert summary["signals"]["ticket_status"]["summary"] == {
        "open": 3,
        "resolved": 1,
    }
    assert summary["signals"]["csat"]["basis"] == "textual"
    assert summary["signals"]["support_resolution_evidence"][
        "rows_with_resolution_evidence"
    ] == 2
    assert "finetunelab.zendesk.com" not in rendered
    assert "duplicate charge" not in rendered


def test_signal_spike_keeps_generic_json_rows_out_of_zendesk_import(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generic_ticket_rows.json"
    path.write_text(
        json.dumps([
            {
                "ticket_id": "GENERIC-424242",
                "subject": "Export CSV",
                "description": "Need the CSV export from https://example.test/ticket",
                "resolution_text": "Open Reports and choose Export.",
                "status": "closed",
                "satisfaction_rating": "good",
            }
        ]),
        encoding="utf-8",
    )

    summary = MODULE.probe_signal_availability(path, source_format="json")
    rendered = json.dumps(summary, sort_keys=True)

    assert summary["input"]["zendesk_thread_import"] is False
    assert summary["signals"]["support_resolution_evidence"][
        "rows_with_resolution_evidence"
    ] == 1
    assert summary["signals"]["csat"]["basis"] == "textual"
    assert "GENERIC-424242" not in rendered
    assert "example.test" not in rendered
    assert "Open Reports" not in rendered
