from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_content_ops_faq_output_proof.py"
SPEC = importlib.util.spec_from_file_location(
    "smoke_content_ops_faq_output_proof",
    SCRIPT,
)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_faq_output_proof_writes_artifacts_and_passes(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"

    exit_code = MODULE.main(["--artifact-dir", str(artifact_dir)])

    assert exit_code == 0
    summary = json.loads((artifact_dir / "proof_summary.json").read_text())
    result = json.loads((artifact_dir / "faq_result.json").read_text())
    full_result = json.loads((artifact_dir / "faq_full_result.json").read_text())
    markdown = (artifact_dir / "faq_output.md").read_text()
    assert summary["ok"] is True
    assert summary["failures"] == []
    assert summary["proof"]["status"] == "ok"
    assert summary["proof"]["generated"] >= 3
    assert len(full_result["items"]) == summary["proof"]["generated"]
    assert summary["proof"]["support_contact_present"] is True
    assert summary["proof"]["source_ids_present"] is True
    assert set(summary["proof"]["topics"]) >= {
        "reporting friction",
        "integration setup",
        "billing and payments",
    }
    assert summary["proof"]["min_source_id_count"] >= 1
    assert summary["proof"]["min_step_count"] >= 2
    assert summary["proof"]["ingestion_bridge"]["adapted_source_row_count"] >= (
        summary["proof"]["generated"]
    )
    assert summary["proof"]["ingestion_bridge"]["adapted_source_types"] == [
        "faq_output"
    ]
    assert summary["proof"]["ingestion_bridge"]["resolution_text_row_count"] >= 1
    assert (
        summary["proof"]["ingestion_bridge"][
            "support_ticket_resolution_evidence_present"
        ]
        is True
    )
    assert (
        summary["proof"]["ingestion_bridge"][
            "support_ticket_resolution_evidence_count"
        ]
        >= 1
    )
    assert summary["proof"]["vocabulary_gaps"]["term_mapping_count"] >= 2
    assert {"export", "SSO"} <= set(
        summary["proof"]["vocabulary_gaps"]["top_customer_terms"]
    )
    assert result["output_checks"] == {
        "condensed": True,
        "has_action_items": True,
        "uses_user_vocabulary": True,
    }
    assert "ticket-export-1" in markdown
    assert "search-export-1" in markdown
    assert "support@example.com" in markdown


def test_faq_output_proof_rejects_custom_source_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        MODULE.main([
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--source",
            str(tmp_path / "custom.csv"),
        ])

    assert exc.value.code == 2


def test_faq_output_proof_reports_missing_result_json() -> None:
    failures = MODULE._proof_failures(
        returncode=0,
        result_payload=None,
        proof={},
        markdown="",
        support_contact="support@example.com",
    )

    assert failures == [
        {"check": "result_json_present", "detail": "missing_or_invalid"}
    ]


def test_faq_output_proof_reports_failed_predicates() -> None:
    failures = MODULE._proof_failures(
        returncode=1,
        result_payload={
            "status": "failed_output_checks",
            "failed_output_checks": ["condensed"],
        },
        proof={
            "status": "failed_output_checks",
            "generated": 1,
            "failed_output_checks": ["condensed"],
            "output_checks": {
                "uses_user_vocabulary": True,
                "condensed": False,
                "has_action_items": False,
            },
            "topics": ["reporting friction"],
            "min_source_id_count": 0,
            "min_step_count": 1,
            "ingestion_bridge": {
                "adapted_source_row_count": 0,
                "adapted_source_types": [],
                "resolution_text_row_count": 0,
                "support_ticket_resolution_evidence_present": False,
                "support_ticket_resolution_evidence_count": 0,
            },
            "support_contact_present": False,
            "source_ids_present": False,
            "vocabulary_gaps": {
                "term_mapping_count": 1,
                "top_customer_terms": ["export"],
            },
        },
        markdown="## Reporting friction",
        support_contact="support@example.com",
    )

    checks = {failure["check"] for failure in failures}
    assert {
        "cli_exit_zero",
        "result_status_ok",
        "output_checks_pass",
        "output_check_condensed",
        "output_check_has_action_items",
        "topic_present",
        "generated_items",
        "source_id_coverage",
        "action_step_coverage",
        "faq_output_adapter_row_coverage",
        "faq_output_adapter_source_type",
        "faq_output_resolution_text_bridge",
        "support_ticket_resolution_evidence_present",
        "support_ticket_resolution_evidence_count",
        "support_contact_present",
        "source_ids_rendered",
        "vocabulary_gap_mappings",
        "vocabulary_gap_terms",
        "markdown_action_section",
    } <= checks
