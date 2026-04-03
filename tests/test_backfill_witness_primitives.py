import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "backfill_witness_primitives.py"
_SPEC = importlib.util.spec_from_file_location("backfill_witness_primitives", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_candidate_clause = _MODULE._candidate_clause


def test_candidate_clause_targets_stale_default_productivity_org_pressure_timeline_and_budget():
    clause = _candidate_clause()

    assert "productivity_delta_claim" in clause
    assert "org_pressure_type" in clause
    assert "decision_timeline" in clause
    assert "timeline,contract_end" in clause
    assert "timeline,evaluation_deadline" in clause
    assert "budget_signals,annual_spend_estimate" in clause
    assert "budget_signals,price_per_seat" in clause
    assert "budget_signals,seat_count" in clause
    assert "budget_signals,price_increase_mentioned" in clause
    assert "ayear" in clause
    assert "save time" in clause
    assert "streamlin" in clause
    assert "manual data entry" in clause
    assert "all-in-one" in clause
    assert "source of truth" in clause
    assert "approved software" in clause
    assert "next quarter" in clause
    assert "at renewal" in clause
    assert "auto[- ]?renew" in clause
    assert "current contract" in clause
    assert "final month" in clause
    assert "quoted" in clause
    assert "per user" in clause
    assert "a year" in clause
    assert "monthly" in clause
    assert "/\\s*mo" in clause
    assert "reviewer_context,company_name" in clause
    assert "reviewer_company" in clause
    assert "reviewer_context,role_level" in clause
    assert "reviewer_context,decision_maker" in clause
    assert "reviewer_title" in clause
    assert "repeat churn signal" in clause
    assert "buyer_authority,has_budget_authority" in clause
    assert "contract_renewal_mentioned" in clause
