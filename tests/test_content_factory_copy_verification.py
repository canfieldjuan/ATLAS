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

from atlas_brain.schemas.content_factory import EditorialAuditV2
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
        "schema": "editorial_audit.v2",
        "project_id": "resolution-audit",
        "recommendation": recommendation,
        "copy_verification": verify_copy(text).model_dump(),
    }


CLEAN = (
    "Support and CX leaders: the Resolution Audit ranks your most repeated tickets and "
    "drafts owner-routed next steps so the right team can review the fix."
)


def test_clean_copy_can_be_promoted():
    audit = EditorialAuditV2.model_validate(_audit(CLEAN, "promote"))
    assert audit.recommendation == "promote"
    assert audit.copy_verification.verdict == "pass"


def test_forbidden_copy_cannot_be_promoted():
    with pytest.raises(ValidationError):
        EditorialAuditV2.model_validate(_audit("Guaranteed savings for all.", "promote"))


def test_forbidden_copy_may_still_recommend_revise():
    audit = EditorialAuditV2.model_validate(_audit("Guaranteed savings for all.", "revise"))
    assert audit.recommendation == "revise"
    assert audit.copy_verification.verdict == "fail"


# --- advisory warning layer (#2136 item 2): non-blocking reviewer checklist ---


def test_advisory_flags_unqualified_answer_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We draft the answer for every repeated ticket.")
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_advisory_passes_qualified_answer_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We draft the answer only when that evidence exists in your tickets."
    )
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_advisory_flags_unqualified_ownership_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Billing owns the refund backlog.")
    assert any(w.startswith("unqualified-ownership-claim:") for w in warnings)


def test_advisory_passes_qualified_ownership_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Billing probably owns the refund backlog.")
    assert not any(w.startswith("unqualified-ownership-claim:") for w in warnings)


def test_advisory_flags_report_shape_without_owner_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("The Resolution Audit snapshot ranks repeated tickets.")
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_advisory_passes_report_shape_with_owner_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The Resolution Audit snapshot ranks repeated tickets and names the owner "
        "lane that needs to review each fix."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_advisory_cta_reminder_is_always_present():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    assert advisory_warnings("Anything at all.")[-1].startswith("reminder:")


def test_advisory_warnings_carry_no_free_text():
    """Round-4 class fix: warnings persist only code + sentence number +
    matched keyword, so PII can never reach the artifact -- no redaction
    completeness argument required."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "Our answer desk replies from bob@example.com within a day."
    )
    joined = " ".join(warnings)
    assert "bob@example" not in joined
    assert any(
        w.startswith("unqualified-answer-claim: sentence 1 (") for w in warnings
    )


def test_advisory_warnings_do_not_block_promotion():
    """Warnings are a checklist, not a gate: a passing verdict may promote
    regardless of how many advisory warnings ride along."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    body = "We draft the answer for every ticket."
    audit = dict(_audit(body, "promote"))
    audit["advisory_warnings"] = advisory_warnings(body)
    validated = EditorialAuditV2.model_validate(audit)
    assert validated.recommendation == "promote"
    assert len(validated.advisory_warnings) >= 2  # claim warning + reminder


def test_advisory_warnings_default_empty_on_v2():
    audit = EditorialAuditV2.model_validate(_audit(CLEAN, "revise"))
    assert audit.advisory_warnings == []


# --- round-1 review fixes: advisory precision + international PII ---


def test_advisory_topic_noun_does_not_suppress_routing_warning():
    """Bare topic nouns (billing/product/team) are not owner routing."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("The snapshot ranks repeated billing questions.")
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_advisory_product_name_is_not_an_answer_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The Resolution Audit ranks repeated tickets and names the owner lane."
    )
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)
    warnings = advisory_warnings("The Resolution Snapshot is ready.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_advisory_qualifier_in_one_clause_does_not_hide_another_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We draft one answer when evidence exists, but we draft every other "
        "answer regardless."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)
    warnings = advisory_warnings(
        "Billing probably owns refunds, but Product owns every escalation."
    )
    assert any(w.startswith("unqualified-ownership-claim:") for w in warnings)


@pytest.mark.parametrize(
    "sentence",
    [
        "Our answer desk is at +44 20 7946 0958.",
        "Our answer desk is at 020 7946 0958.",
        "Our answer desk is at 020/7946/0958.",
        "Answers are available at 020 - 7946 - 0958.",
        "Our answer desk is at 618-555-9876.",
    ],
)
def test_advisory_never_persists_phone_fragments(sentence):
    """Any separator style, any country format: with evidence-free warnings
    the number cannot appear because no draft text is persisted at all."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    joined = " ".join(advisory_warnings(sentence))
    for fragment in ("7946", "0958", "5559876", "555-9876"):
        assert fragment not in joined


def test_gate_scope_unchanged_for_international_phone():
    """Round-2 revert: the slice contract freezes verdict semantics, so the
    international matcher is redaction-only; widening the gate's PII block is
    a separate operator decision (standing catalogue policy)."""
    assert verify_copy("Call us at +44 20 7946 0958 today.").verdict == "pass"





def test_gate_ignores_plus_versions_and_short_numbers():
    assert verify_copy("Supports version +2.5 and 2026+ planning.").verdict == "pass"


def test_advisory_bare_draft_or_ranked_is_not_report_shape():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for text in ("This draft is ready for review.", "Ranked choice voting is supported."):
        warnings = advisory_warnings(text)
        assert not any(w.startswith("owner-routing-coverage:") for w in warnings), text


# --- round-3 review fixes ---


def test_gate_hit_evidence_still_masks_digit_runs():
    """The gate's claim-hit evidence (which DOES record matched phrases)
    keeps the digit-run backstop."""
    result = verify_copy("Guaranteed 61855598761 savings for all.")
    assert result.verdict == "fail"
    joined = " ".join(result.hits)
    assert "5559876" not in joined


def test_advisory_negated_routing_does_not_suppress_warning():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for text in (
        "The report ranks issues, but no one is assigned to them.",
        "The report ranks issues; they are not routed to Billing.",
    ):
        warnings = advisory_warnings(text)
        assert any(
            w.startswith("owner-routing-coverage:") for w in warnings
        ), text


def test_advisory_affirmative_routing_still_suppresses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues, and each one is routed to the owning team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_editorial_audit_v1_api_unchanged():
    """Pre-#2181 consumers keep working: EditorialAudit still validates a
    v1 payload."""
    from atlas_brain.schemas.content_factory import EditorialAudit

    audit = EditorialAudit.model_validate(
        {"schema": "editorial_audit.v1", "project_id": "p"}
    )
    assert audit.recommendation == "revise"


# --- round-5 review fixes ---


def test_qualifier_association_survives_any_separator():
    """Fail-closed association: one qualifier excuses ONE claim, so no
    separator style (em dash, slash, parens, or future ones) can hide a
    second claim behind a neighboring qualified one."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for text in (
        "We draft one answer when evidence exists — we draft every other answer regardless.",
        "We draft one answer when evidence exists / another answer regardless.",
        "We draft one answer (when evidence exists) and another answer regardless.",
    ):
        warnings = advisory_warnings(text)
        assert any(
            w.startswith("unqualified-answer-claim:") for w in warnings
        ), text


def test_each_claim_with_its_own_qualifier_stays_silent():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We draft an answer when evidence exists, and a resolution only if "
        "the tickets contain proof."
    )
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_inanimate_owns_does_not_suppress_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Caching owns the latency."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_owner_team_owns_still_suppresses_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. The billing team owns each fix."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_sentence_locators_count_real_sentences():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Version 2.1 provides an answer.")
    assert any("sentence 1" in w for w in warnings if w.startswith("unqualified-answer-claim"))
    warnings = advisory_warnings("Really?! We draft an answer.")
    assert any("sentence 2" in w for w in warnings if w.startswith("unqualified-answer-claim"))


def test_schema_rejects_free_text_advisory_warnings():
    """Choke point: a DIRECT writer cannot persist free-text warnings — the
    v2 contract only admits the bounded deterministic grammar."""
    from pydantic import ValidationError

    from atlas_brain.schemas.content_factory import EditorialAuditV2

    with pytest.raises(ValidationError):
        EditorialAuditV2.model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "p",
                "advisory_warnings": [
                    "Contact bob@example.com or +44 20 7946 0958"
                ],
            }
        )


def test_schema_accepts_deterministic_warning_grammar():
    from atlas_brain.schemas.content_factory import (
        ADVISORY_CTA_REMINDER,
        ADVISORY_OWNER_ROUTING_WARNING,
        EditorialAuditV2,
    )

    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "advisory_warnings": [
                "unqualified-answer-claim: sentence 3 ('answer')",
                ADVISORY_OWNER_ROUTING_WARNING,
                ADVISORY_CTA_REMINDER,
            ],
        }
    )
    assert len(audit.advisory_warnings) == 3


def test_generated_warnings_always_satisfy_schema_grammar():
    """Producer/contract lockstep: everything advisory_warnings emits must
    validate at the schema choke point."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    generated = advisory_warnings(
        "The Resolution Audit snapshot ranks repeated tickets. We draft the "
        "answer for every ticket. Billing owns refunds. Really?! Version 2.1 "
        "provides an answer at bob@example.com or 020/7946/0958."
    )
    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "advisory_warnings": generated,
        }
    )
    assert audit.advisory_warnings == generated
