import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "backfill_witness_primitives.py"
_SPEC = importlib.util.spec_from_file_location("backfill_witness_primitives", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_candidate_clause = _MODULE._candidate_clause


def test_candidate_clause_targets_stale_default_productivity_org_pressure_and_timeline():
    clause = _candidate_clause()

    assert "productivity_delta_claim" in clause
    assert "org_pressure_type" in clause
    assert "decision_timeline" in clause
    assert "timeline,contract_end" in clause
    assert "timeline,evaluation_deadline" in clause
    assert "save time" in clause
    assert "approved software" in clause
    assert "next quarter" in clause
