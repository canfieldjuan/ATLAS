"""Acceptance tests for the EvidenceClaim contract.

Loads every JSON fixture under ``tests/fixtures/evidence_claims`` and
validates each (claim_type, witness, target_entity) tuple against the
expected status / rejection_reason. This is the contract that pins the
v1 deterministic validator before any consumer migrates onto it.

See docs/progress/evidence_claim_contract_plan_2026-04-25.md and
``atlas_brain/services/b2b/evidence_claim.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from atlas_brain.services.b2b.evidence_claim import (
    ClaimType,
    ClaimValidationStatus,
    validate_claim,
)


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "evidence_claims"


def _load_all() -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for path in sorted(_FIXTURE_DIR.glob("*.json")):
        with path.open() as fh:
            data = json.load(fh)
        data["__path"] = str(path)
        fixtures.append(data)
    return fixtures


def _flatten_cases() -> list[tuple[str, str, dict, dict, dict, dict]]:
    """Yield (fixture_name, claim_type_str, witness, expected, review, fixture)
    for every claim assertion in every fixture."""
    cases: list[tuple[str, str, dict, dict, dict, dict]] = []
    for fixture in _load_all():
        name = fixture["name"]
        witness = fixture["witness"]
        review = fixture.get("review", {})
        for claim_type_str, expected in fixture.get("expected", {}).items():
            cases.append((name, claim_type_str, witness, expected, review, fixture))
    return cases


_CASES = _flatten_cases()


def _secondary_target_for(claim_type: ClaimType, witness: dict[str, Any]) -> str | None:
    if claim_type in {
        ClaimType.DISPLACEMENT_PROOF_TO_COMPETITOR,
        ClaimType.DISPLACEMENT_PROOF_FROM_COMPETITOR,
        ClaimType.FEATURE_GAP_CLAIM,
    }:
        comp = witness.get("competitor")
        return str(comp) if comp else None
    return None


def _ids(case: tuple[str, str, dict, dict, dict, dict]) -> str:
    return f"{case[0]}::{case[1]}"


@pytest.mark.parametrize("case", _CASES, ids=_ids)
def test_evidence_claim_fixture(case):
    name, claim_type_str, witness, expected, review, _fixture = case
    claim_type = ClaimType(claim_type_str)
    secondary_target = _secondary_target_for(claim_type, witness)
    result = validate_claim(
        claim_type=claim_type,
        witness=witness,
        target_entity=witness.get("vendor_name") or review.get("vendor_name", ""),
        secondary_target=secondary_target,
        source_review=review,
    )
    assert result.status.value == expected["status"], (
        f"{name}::{claim_type_str} expected status={expected['status']} "
        f"got status={result.status.value} reason={result.rejection_reason!r}"
    )
    assert result.rejection_reason == expected.get("rejection_reason"), (
        f"{name}::{claim_type_str} expected reason={expected.get('rejection_reason')!r} "
        f"got reason={result.rejection_reason!r}"
    )
