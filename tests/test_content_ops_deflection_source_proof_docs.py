import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md"
CFPB_PROOF = ROOT / "docs/extraction/validation/deflection_full_volume_live_proof_2026-06-14.md"
ZENDESK_PROOF = ROOT / "docs/extraction/validation/deflection_zendesk_full_thread_proof_2026-06-13.md"


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
