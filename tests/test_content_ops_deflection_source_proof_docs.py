import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md"
CFPB_PROOF = ROOT / "docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md"
ZENDESK_PROOF = ROOT / "docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md"
ZENDESK_PRODUCT_PROOF = ROOT / "docs/extraction/validation/deflection_zendesk_product_proof_2026-06-14.md"
ZENDESK_PRODUCT_SUMMARY = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/summary.json"
)
ZENDESK_PRODUCT_EXCERPT = (
    ROOT
    / "docs/extraction/validation/fixtures/deflection_zendesk_product_proof_20260614/report_excerpt.md"
)


def _doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _flat_doc(path: Path) -> str:
    return " ".join(_doc(path).split())


def test_runbook_separates_cfpb_stress_from_zendesk_product_proof() -> None:
    text = _flat_doc(RUNBOOK)

    assert re.search(r"CFPB full-volume exports.*stress/scale evidence", text)
    assert re.search(r"not Zendesk-like ticket calibration", text)
    assert re.search(r"Zendesk-shaped full-thread exports.*product/integration evidence", text)
    assert re.search(r"Small seeded Zendesk fixtures.*do not prove full-volume", text)


def test_cfpb_live_proof_rejects_product_quality_calibration_claim() -> None:
    text = _flat_doc(CFPB_PROOF)

    assert re.search(r"CFPB stress/scale evidence", text)
    assert re.search(r"not a Zendesk-like ticket calibration", text)
    assert re.search(r"not a buyer-readiness standard", text)
    assert re.search(r"25,000 repeat-ticket minimum.*Zendesk product-quality gate", text)


def test_zendesk_full_thread_proof_states_small_fixture_boundary() -> None:
    text = _flat_doc(ZENDESK_PROOF)

    assert re.search(r"Zendesk-shaped product/integration evidence", text)
    assert re.search(r"buyer.*actual support workflow", text)
    assert re.search(r"not a full-volume stress proof", text)
    assert re.search(r"product-quality shape and full-volume size", text)


def test_zendesk_product_proof_records_live_api_boundaries_and_sanitized_artifacts() -> None:
    text = _flat_doc(ZENDESK_PRODUCT_PROOF)
    summary = json.loads(ZENDESK_PRODUCT_SUMMARY.read_text(encoding="utf-8"))

    assert re.search(r"replaces the four-row Zendesk fixture", text)
    assert re.search(r"Zendesk product/integration evidence", text)
    assert re.search(r"This is not a full-volume stress proof", text)
    assert re.search(r"Raw Zendesk export: not committed", text)
    assert re.search(r"hosted Atlas export route.*was not available", text)
    assert re.search(r"portfolio wrapper prerequisites were also missing", text)
    assert re.search(r"#1567.*canonical source for Zendesk cursor-endpoint behavior", text)
    assert "per_page" not in text

    assert summary["source"] == "zendesk_trial_api_live_export"
    assert summary["raw_export_committed"] is False
    assert summary["ticket_count"] >= 20
    assert summary["comment_count"] >= summary["ticket_count"]
    assert summary["private_comment_count"] > 0
    assert summary["normalization_warning_count"] == 0
    assert summary["publishable_answer_count"] > 0
    assert summary["package_metadata"]["support_ticket_resolution_evidence_present"] is True
    assert summary["package_metadata"]["ticket_status_present"] is True
    assert summary["package_metadata"]["csat_present"] is True
    assert summary["private_note_leak_checks"]["private_marker_present_in_markdown"] is False
    assert summary["private_note_leak_checks"]["auto_ack_present_in_markdown"] is False


def test_zendesk_product_proof_names_output_quality_boundaries() -> None:
    text = _flat_doc(ZENDESK_PRODUCT_PROOF)
    excerpt = _doc(ZENDESK_PRODUCT_EXCERPT)
    summary = json.loads(ZENDESK_PRODUCT_SUMMARY.read_text(encoding="utf-8"))

    assert "[Atlas seed" in excerpt
    assert "What should I do about atla?" in excerpt
    atla_questions = [
        q for q in summary["top_questions"]
        if q["question"] == "What should I do about atla?"
    ]
    assert [q["rank"] for q in atla_questions] == [4, 6, 8]

    assert re.search(r"Output-Quality Boundary", text)
    assert re.search(r"Synthetic subject prefixes leak", text)
    assert re.search(r"degraded question label `What should I do about atla\?`", text)
    assert re.search(r"not buyer-ready as-is", text)
