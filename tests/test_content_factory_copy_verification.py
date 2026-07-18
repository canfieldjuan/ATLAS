"""Tests for the deterministic Content Factory copy-verification gate (Phase 4.1).

Both sides of the guard: clean/legitimate copy passes (false-positive guard); each
promote-blocking category's common wordings fail (incl. the variants Codex flagged);
negation is scoped to the claim (unrelated earlier negations no longer suppress a real
hit, and the gate errs toward flagging when a negation's scope is ambiguous -- fail
closed); PII is blocked but redacted out of the persisted hits; and the verdict gates
promotion through the #2116 EditorialAudit contract.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlas_brain.schemas.content_factory import EditorialAudit
from atlas_brain.services.content_factory_copy_verification import verify_copy


# --- clean / legitimate copy must PASS (false-positive guard) ---


@pytest.mark.parametrize(
    "text",
    [
        "Support and CX leaders: the Resolution Audit ranks your most repeated tickets "
        "and drafts owner-routed next steps so the right team can review the fix.",
        "We help teams understand where their support workload concentrates.",
        "Our guarantee is honest, evidence-backed reporting.",  # 'guarantee' but not savings
        "We guarantee honest reporting. Savings summaries are reviewed by your team.",  # cross-sentence
        "Auto-save keeps your draft safe as you write.",  # 'auto-' but not auto-publish
        "Replace your old spreadsheet with a single source of truth.",  # replace, not agents
        "Replace your agents' spreadsheet with a dashboard.",  # possessive: object is spreadsheet
        "Avoid distracting your agents with repetitive triage.",  # avoid, not a hire
        "Deflection improved noticeably this quarter.",  # deflection, no fixed %
        "Reduce back-and-forth by triaging tickets faster.",  # reduce tickets, no %
        "Avoid a stockout with better forecasting.",  # avoid, not a hire/agent
    ],
)
def test_legitimate_copy_passes(text):
    assert verify_copy(text).verdict == "pass", verify_copy(text).hits


# --- forbidden claims (incl. the variants Codex flagged) must FAIL ---


@pytest.mark.parametrize(
    "text,code_fragment",
    [
        # guaranteed savings + modifiers
        ("We deliver guaranteed savings on every plan.", "guaranteed-savings"),
        ("Enjoy guaranteed cost savings from month one.", "guaranteed-savings"),
        ("You get guaranteed monthly savings.", "guaranteed-savings"),
        ("Our tool guarantees savings for your team.", "guaranteed-savings"),
        ("Guaranteed 30% savings from day one.", "guaranteed-savings"),
        ("We guarantee 30% savings.", "guaranteed-savings"),
        ("Atlas not only guarantees savings for teams.", "guaranteed-savings"),  # emphatic, not negation
        # deflection %, both forms
        ("Expect 40% deflection from day one.", "fixed-deflection-percent"),
        ("Expect 40 percent deflection from day one.", "fixed-deflection-percent"),
        # ticket reductions, direct + volume + fewer, both % forms
        ("We cut ticket volume by 30% in a month.", "fixed-ticket-reduction"),
        ("We reduce tickets by 30%.", "fixed-ticket-reduction"),
        ("We reduce support tickets by 20 percent.", "fixed-ticket-reduction"),
        ("Customers see 25% fewer tickets.", "fixed-fewer-tickets"),
        ("Customers see 25% fewer support tickets.", "fixed-fewer-tickets"),
        # automation
        ("Enjoy automatic ticket answering out of the box.", "automatic-ticket-answering"),
        ("Your help center is auto-published nightly.", "auto-publish"),
        ("We auto-publish your help center nightly.", "auto-publish"),
        ("Atlas auto-publishes your help center.", "auto-publish"),
        ("It answers tickets automatically for you.", "answers-tickets-automatically"),
        # replacing agents / avoided hire
        ("This lets you replace agents entirely.", "replace-agents"),
        ("Replace your agents entirely.", "replace-agents"),
        ("You can avoid a support hire this year.", "avoid-support-hire"),
        ("Avoid hiring another support agent this year.", "avoid-support-hire"),
    ],
)
def test_forbidden_claims_fail(text, code_fragment):
    r = verify_copy(text)
    assert r.verdict == "fail", text
    assert any(code_fragment in h for h in r.hits), r.hits


# --- negation scoping ---


@pytest.mark.parametrize(
    "text",
    [
        "We make no guaranteed savings claims.",
        "This does not promise guaranteed savings.",
        "We will not guarantee savings.",
        "We don't guarantee savings.",
        "We cannot guarantee savings.",
    ],
)
def test_direct_negation_passes(text):
    # A negation immediately governing the claim suppresses the hit.
    assert verify_copy(text).verdict == "pass", verify_copy(text).hits


@pytest.mark.parametrize(
    "text",
    [
        "No setup fee, and guaranteed savings are included.",
        "This does not delay launch and guarantees savings.",
    ],
)
def test_unrelated_earlier_negation_does_not_suppress(text):
    # The P1 fix: an unrelated negation earlier in the sentence must NOT hide a real
    # claim (the source tool scanned the whole prefix and leaked these).
    assert verify_copy(text).verdict == "fail", verify_copy(text).hits


def test_far_scope_negation_flags_conservatively():
    # When a negation is too far from the claim to attribute cheaply, the gate errs
    # toward flagging (fail closed) -- a false positive only routes to human review,
    # whereas a false negative would ship an overclaim.
    assert verify_copy("We never promise a fixed 40% deflection.").verdict == "fail"


# --- PII: blocked, but redacted out of the persisted hits ---


@pytest.mark.parametrize(
    "text,code,secret",
    [
        ("Reach me at jane.doe@example.com for details.", "email", "jane.doe@example.com"),
        ("Call the owner at (618) 555-1234 today.", "phone", "555-1234"),
        ("Direct line: 618-555-9876.", "phone", "555-9876"),
    ],
)
def test_raw_contact_pii_fails_and_is_redacted(text, code, secret):
    r = verify_copy(text)
    assert r.verdict == "fail"
    assert f"{code}: <redacted>" in r.hits
    # the actual PII must never appear in the persisted verdict
    assert all(secret not in h for h in r.hits), r.hits


def test_multiple_hits_all_recorded():
    r = verify_copy("Guaranteed savings! Email us at x@y.com.")
    assert r.verdict == "fail"
    assert len(r.hits) >= 2
    assert all("x@y.com" not in h for h in r.hits)


def test_pii_inside_claim_evidence_is_redacted():
    # PII that falls INSIDE a matched claim phrase must not persist via the claim hit.
    r = verify_copy("Guaranteed 618-555-9876 savings.")
    assert r.verdict == "fail"
    assert all("618-555-9876" not in h for h in r.hits), r.hits


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


CLEAN = (
    "Support and CX leaders: the Resolution Audit ranks your most repeated tickets and "
    "drafts owner-routed next steps so the right team can review the fix."
)


def test_clean_copy_can_be_promoted():
    audit = EditorialAudit.model_validate(_audit(CLEAN, "promote"))
    assert audit.recommendation == "promote"
    assert audit.copy_verification.verdict == "pass"


def test_forbidden_copy_cannot_be_promoted():
    with pytest.raises(ValidationError):
        EditorialAudit.model_validate(_audit("Guaranteed savings for all.", "promote"))


def test_forbidden_copy_may_still_recommend_revise():
    audit = EditorialAudit.model_validate(_audit("Guaranteed savings for all.", "revise"))
    assert audit.recommendation == "revise"
    assert audit.copy_verification.verdict == "fail"
