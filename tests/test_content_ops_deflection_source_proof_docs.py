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

    assert "CFPB full-volume exports are stress/scale evidence." in text
    assert "They are not Zendesk-like ticket calibration" in text
    assert "Zendesk-shaped full-thread exports are product/integration evidence." in text
    assert "Small seeded Zendesk fixtures do not prove full-volume performance" in text


def test_cfpb_live_proof_rejects_product_quality_calibration_claim() -> None:
    text = _flat_doc(CFPB_PROOF)

    assert "this is CFPB stress/scale evidence" in text
    assert "not a Zendesk-like ticket calibration source" in text
    assert "not a buyer-readiness standard for product-quality support output" in text
    assert "Do not treat the 25,000 repeat-ticket minimum as a Zendesk product-quality gate." in text


def test_zendesk_full_thread_proof_states_small_fixture_boundary() -> None:
    text = _flat_doc(ZENDESK_PROOF)

    assert "this is Zendesk-shaped product/integration evidence" in text
    assert "pointed at the buyer's actual support workflow" in text
    assert "not a full-volume stress proof by itself" in text
    assert "product-quality shape and full-volume size" in text
