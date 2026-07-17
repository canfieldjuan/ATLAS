"""Tests for the deterministic Content Factory copy-verification gate (Phase 4.1).

Both sides of the guard: clean copy passes; each forbidden-claim category and each PII
shape fails; negated claims pass (parity with the source tool); and the produced verdict
actually gates promotion through the #2116 EditorialAudit contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.content_factory import EditorialAudit
from atlas_brain.services.content_factory_copy_verification import verify_copy

CLEAN = (
    "Support and CX leaders: the Resolution Audit ranks your most repeated tickets and "
    "drafts owner-routed next steps so the right team can review the fix."
)


def test_clean_copy_passes_with_no_hits():
    r = verify_copy(CLEAN)
    assert r.verdict == "pass"
    assert r.hits == []


@pytest.mark.parametrize(
    "text,code_fragment",
    [
        ("We deliver guaranteed savings on every plan.", "guaranteed-savings"),
        ("Our tool guarantees savings for your team.", "guarantees-savings"),
        ("Expect 40% deflection from day one.", "fixed-deflection-percent"),
        # Both forms hit after the operator-authorized %-boundary fix.
        ("We cut ticket volume by 30 percent in a month.", "fixed-ticket-volume-reduction"),
        ("We cut ticket volume by 30% in a month.", "fixed-ticket-volume-reduction"),
        ("Customers see 25% fewer tickets.", "fixed-fewer-tickets"),
        ("Enjoy automatic ticket answering out of the box.", "automatic-ticket-answering"),
        ("Your help center is auto-published nightly.", "auto-published"),
        ("It answers tickets automatically for you.", "answers-tickets-automatically"),
        ("This lets you replace agents entirely.", "replace-agents"),
        ("You can avoid a support hire this year.", "avoid-support-hire"),
    ],
)
def test_forbidden_claims_fail(text, code_fragment):
    r = verify_copy(text)
    assert r.verdict == "fail"
    assert any(code_fragment in h for h in r.hits), r.hits


@pytest.mark.parametrize(
    "text",
    [
        "We make no guaranteed savings claims.",
        "This does not promise guaranteed savings.",
        "We never promise a fixed 40% deflection.",
    ],
)
def test_negated_claims_pass(text):
    # Parity with the source tool: a negated claim is not a hit.
    assert verify_copy(text).verdict == "pass"


@pytest.mark.parametrize(
    "text,code",
    [
        ("Reach me at jane.doe@example.com for details.", "email"),
        ("Call the owner at (618) 555-1234 today.", "phone"),
        ("Direct line: 618-555-9876.", "phone"),
    ],
)
def test_raw_contact_pii_fails(text, code):
    r = verify_copy(text)
    assert r.verdict == "fail"
    assert any(h.startswith(code + ":") for h in r.hits), r.hits


def test_multiple_hits_all_recorded():
    r = verify_copy("Guaranteed savings! Email us at x@y.com.")
    assert r.verdict == "fail"
    assert len(r.hits) >= 2


def test_non_string_rejected():
    with pytest.raises(TypeError):
        verify_copy(None)


# --- the verdict actually gates promotion (wires to the #2116 contract) ---


def _audit(text, recommendation):
    return {
        "schema": "editorial_audit.v1",
        "project_id": "resolution-audit",
        "recommendation": recommendation,
        "copy_verification": verify_copy(text).model_dump(),
    }


def test_clean_copy_can_be_promoted():
    audit = EditorialAudit.model_validate(_audit(CLEAN, "promote"))
    assert audit.recommendation == "promote"
    assert audit.copy_verification.verdict == "pass"


def test_forbidden_copy_cannot_be_promoted():
    # verify_copy -> fail -> EditorialAudit's promote-requires-pass guard rejects it.
    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(_audit("Guaranteed savings for all.", "promote"))


def test_forbidden_copy_may_still_recommend_revise():
    audit = EditorialAudit.model_validate(_audit("Guaranteed savings for all.", "revise"))
    assert audit.recommendation == "revise"
    assert audit.copy_verification.verdict == "fail"
