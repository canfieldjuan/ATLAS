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
    # edited_body_markdown carries the SAME text the verdict and any advisory
    # warnings were computed from. Omitting it let a fixture pair locators with
    # an empty audited body -- the inconsistency #2189's locator binding
    # exists to make unrepresentable.
    return {
        "schema": "editorial_audit.v2",
        "project_id": "resolution-audit",
        "recommendation": recommendation,
        "edited_body_markdown": text,
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
    assert "unqualified-answer-claim: sentence 1" in warnings


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


def test_object_position_anaphora_does_not_bind_routing():
    """Round 13 reversed the round-12 direction: only SUBJECT-position
    anaphora binds a later routing clause to the report ("Each is
    assigned..."); an anaphor buried in the object ("owns each fix") does
    not -- fail-closed, the warning fires."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. The billing team owns each fix."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_report_item_subject_binds_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. These issues are owned by the billing team."
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
            # A locator names a sentence of THIS body, so the body must have
            # at least three (#2189).
            "edited_body_markdown": "One. Two. Three.",
            "advisory_warnings": [
                "unqualified-answer-claim: sentence 3",
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

    body = (
        "The Resolution Audit snapshot ranks repeated tickets. We draft the "
        "answer for every ticket. Billing owns refunds. Really?! Version 2.1 "
        "provides an answer at bob@example.com or 020/7946/0958."
    )
    generated = advisory_warnings(body)
    # The audit must carry the SAME body: a locator is only meaningful against
    # the text it was computed from (#2189), so lockstep is now the stronger
    # property "warnings from B validate on an audit whose body is B".
    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "edited_body_markdown": body,
            "advisory_warnings": generated,
        }
    )
    assert audit.advisory_warnings == generated


# --- round-6 review fixes ---


def test_unrelated_qualifier_in_other_clause_does_not_excuse():
    """Clause-scoped association: a qualifier in one clause cannot excuse a
    claim in another, even as the sentence's only claim."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "When evidence exists, support triages tickets, but we draft an "
        "answer regardless."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_bare_routing_noun_does_not_suppress():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("The report ranks issues. Routing takes time.")
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_sentence_locators_ignore_domain_and_abbreviation_periods():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Contact bob@example.com about an answer.")
    assert any(
        "sentence 1" in w for w in warnings if w.startswith("unqualified-answer-claim")
    )
    warnings = advisory_warnings("E.g. we draft an answer.")
    assert any(
        "sentence 1" in w for w in warnings if w.startswith("unqualified-answer-claim")
    )


def test_multiline_claim_keyword_stays_schema_valid():
    """Round-6 BLOCKER regression: a wrapped draft must not make the producer
    emit a locator its own v2 grammar rejects."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    body = "Billing\nreally owns refunds."
    generated = advisory_warnings(body)
    assert any(w.startswith("unqualified-ownership-claim:") for w in generated)
    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "edited_body_markdown": body,
            "advisory_warnings": generated,
        }
    )
    assert audit.advisory_warnings == generated


# --- round-7 review fixes ---


def test_qualifier_after_dash_does_not_consume_earlier_claim():
    """Clause granularity: a dash-separated qualifier governs its own
    proposition, not the claim before the dash."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We draft every answer regardless — when evidence exists support "
        "reviews tickets."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_assignment_to_metadata_is_not_owner_coverage():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is assigned to a severity."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_assignment_to_team_still_suppresses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is assigned to the billing team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_terminator_plus_newline_is_one_boundary():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Intro.\nWe draft an answer.")
    assert any(
        "sentence 2" in w for w in warnings if w.startswith("unqualified-answer-claim")
    )


# --- round-8 review fixes ---


def test_fronted_qualifier_excuses_following_claim():
    """Regression for the round-7 tightening: the ordinary fronted form
    stays silent."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("When evidence exists, we draft answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_fronted_qualifier_does_not_reach_later_clauses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "When evidence exists, support triages tickets, but we draft an "
        "answer regardless."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_owned_by_metadata_is_not_owner_coverage():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is probably owned by severity."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_owned_by_team_still_suppresses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is owned by the support team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_soft_wrapped_line_stays_in_sentence():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("This is a long\nwrapped answer for customers.")
    assert any(
        "sentence 1" in w for w in warnings if w.startswith("unqualified-answer-claim")
    )


@pytest.mark.parametrize(
    "text",
    [
        "We do not draft answers.",
        "No answers are generated.",
        "Refunds are never owned by Billing.",
    ],
)
def test_negated_claims_are_denials_not_warnings(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert not any(
        w.startswith(("unqualified-answer-claim:", "unqualified-ownership-claim:"))
        for w in warnings
    ), text


def test_positive_claims_still_warn_after_polarity_check():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    assert any(
        w.startswith("unqualified-answer-claim:")
        for w in advisory_warnings("We draft answers.")
    )
    assert any(
        w.startswith("unqualified-ownership-claim:")
        for w in advisory_warnings("Billing owns refunds.")
    )


# --- round-9 review fixes ---


def test_qualifier_from_prior_sentence_does_not_carry():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "Support reviews tickets when evidence exists. We draft answers regardless."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_emphatic_not_only_still_registers_claims():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    assert any(
        w.startswith("unqualified-answer-claim:")
        for w in advisory_warnings("We not only draft answers for every ticket.")
    )
    assert any(
        w.startswith("unqualified-ownership-claim:")
        for w in advisory_warnings("Billing not only owns refunds.")
    )


def test_soft_wrap_before_proper_noun_stays_in_sentence():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "This sentence wraps before the\nBilling answer for customers."
    )
    assert any(
        "sentence 1" in w for w in warnings if w.startswith("unqualified-answer-claim")
    )


def test_routing_by_metadata_is_not_owner_coverage():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues and routes each issue by severity."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_routing_each_issue_to_team_still_suppresses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues and routes each issue to the owning team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


# --- round-10 review fixes ---


def test_gate_hits_mask_multi_separator_digit_runs():
    result = verify_copy("Guaranteed 020--7946--0958 savings for all.")
    assert result.verdict == "fail"
    assert "7946" not in " ".join(result.hits)


@pytest.mark.parametrize(
    "text",
    [
        "Billing never owns refunds.",
        "Billing does not own refunds.",
        "Billing cannot own refunds.",
    ],
)
def test_subject_first_denials_are_not_ownership_claims(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    assert not any(
        w.startswith("unqualified-ownership-claim:")
        for w in advisory_warnings(text)
    ), text


def test_unrelated_absence_after_target_keeps_routing_affirmative():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is assigned to billing with no due date."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_negated_routing_target_still_warns():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Nothing is ever assigned to the billing team."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_v2_schema_version_is_pinned():
    from pydantic import ValidationError

    from atlas_brain.schemas.content_factory import EditorialAuditV2

    with pytest.raises(ValidationError):
        EditorialAuditV2.model_validate(
            {"schema": "editorial_audit.v2", "project_id": "p", "schema_version": 1}
        )


# --- round-11 review fixes ---


def test_gate_hits_mask_multiline_digit_runs():
    result = verify_copy("Guaranteed 020--\n7946--\n0958 savings.")
    assert result.verdict == "fail"
    assert "7946" not in " ".join(result.hits)


def test_owner_lane_unknown_is_not_coverage():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("The report ranks issues. The owner lane is unknown.")
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_trailing_modifier_still_keeps_routing_affirmative():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is assigned to billing with no due date."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_unrelated_prefix_negation_does_not_deny_claim():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("With no delay we draft answers.")
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_genuine_denials_still_recognized_after_binding():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for text in ("We do not draft answers.", "No answers are generated.",
                 "Billing never owns refunds."):
        assert not any(
            w.startswith(("unqualified-answer-claim:", "unqualified-ownership-claim:"))
            for w in advisory_warnings(text)
        ), text


# --- round-12 review fixes + engine invariants ---


def test_gate_hits_never_contain_digits_theorem():
    """THEOREM: no digit character survives into persisted hit evidence,
    regardless of separator style (word chars, underscores, newlines...)."""
    for evil in (
        "Guaranteed 020__7946__0958 savings.",
        "Guaranteed 020--\n7946--\n0958 savings.",
        "Guaranteed 020a7946a0958 savings today.",
    ):
        result = verify_copy(evil)
        joined = " ".join(result.hits)
        assert not any(ch.isdigit() for ch in joined), (evil, joined)


def test_denial_with_modifiers_recognized():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We do not draft any customer-facing answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_owner_lane_label_absence_warns():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for text in (
        "The report ranks issues. Owner lane: TBD.",
        "The report ranks issues. Owner lane — unassigned.",
    ):
        warnings = advisory_warnings(text)
        assert any(w.startswith("owner-routing-coverage:") for w in warnings), text


def test_report_shape_with_modifier_detected():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("The report clearly ranks issues by severity.")
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_negated_report_shape_is_silent():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("This is not a report that ranks issues.")
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_unrelated_ownership_does_not_cover_report():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues by severity. Billing probably owns invoice collection."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_anaphoric_routing_still_covers_report():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each is assigned to the billing team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_locators_cannot_carry_names():
    """Round-12 BLOCKER: warnings are code + sentence number only — a name
    inside a matched relation is unrepresentable."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Support lead Alice owns refunds.")
    joined = " ".join(warnings)
    assert "Alice" not in joined
    assert any(w == "unqualified-ownership-claim: sentence 1" for w in warnings)


# --- generative invariants (grammar-derived probes) ---

_CLAIM_TEMPLATES = [
    "We draft {noun} for every ticket.",
    "Our team provides {noun} daily.",
    "{noun} are delivered to customers.",
]
_DENIAL_TEMPLATES = [
    "We do not draft {noun}.",
    "We never provide {noun}.",
    "No {noun} are generated.",
    "We cannot draft {noun}.",
]
_NOUNS = ["answers", "resolutions", "drafted answers"]


def test_invariant_generated_denials_never_warn():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for template in _DENIAL_TEMPLATES:
        for noun in _NOUNS:
            text = template.format(noun=noun)
            assert not any(
                w.startswith("unqualified-answer-claim:")
                for w in advisory_warnings(text)
            ), text


def test_invariant_generated_bare_claims_always_warn():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    for template in _CLAIM_TEMPLATES:
        for noun in _NOUNS:
            text = template.format(noun=noun)
            assert any(
                w.startswith("unqualified-answer-claim:")
                for w in advisory_warnings(text)
            ), text


def test_invariant_all_outputs_satisfy_schema_grammar():
    """Every warning the producer can emit — across claim, denial, PII,
    routing, and pathological inputs — validates at the schema choke point."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    corpus = [t.format(noun=n) for t in _CLAIM_TEMPLATES + _DENIAL_TEMPLATES for n in _NOUNS]
    corpus += [
        "Support lead Alice owns refunds at bob@example.com or 020__7946__0958.",
        "The report ranks issues. Owner lane: TBD.",
        "Billing\nreally owns refunds.",
        "Intro.\n\nWe draft answers. Really?! Version 2.1 provides an answer.",
    ]
    for text in corpus:
        generated = advisory_warnings(text)
        audit = EditorialAuditV2.model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "p",
                "edited_body_markdown": text,
                "advisory_warnings": generated,
            }
        )
        assert audit.advisory_warnings == generated
        joined = " ".join(generated)
        assert "@" not in joined and "Alice" not in joined


# --- round-13 review fixes ---


def test_gate_masks_unicode_digits():
    """Digit theorem covers every decimal script, not just ASCII."""
    result = verify_copy("Guaranteed ٠٢٠__٧٩٤٦ savings.")
    joined = " ".join(result.hits)
    assert not any(ch.isdigit() for ch in joined), joined


@pytest.mark.parametrize(
    "text",
    [
        "The report does not rank issues.",
        "The audit never lists problems.",
        "The snapshot cannot show trends.",
    ],
)
def test_negated_shape_verb_is_not_report_shape(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings), text


def test_in_match_routing_negation_does_not_cover():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues, but Billing never owns them."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_object_anaphora_in_unrelated_ownership_does_not_bind():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Billing owns invoices for each customer."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


# --- round-14 review fixes ---


def test_gate_masks_all_unicode_digit_categories():
    """Category-complete: circled/superscript digits mask too (the predicate
    is str.isdigit/isnumeric itself, not a regex class)."""
    result = verify_copy("Guaranteed ⓪②⓪__⑦⑨④⑥ savings.")
    joined = " ".join(result.hits)
    assert not any(ch.isdigit() or ch.isnumeric() for ch in joined if ch != "#"), joined


def test_same_sentence_unrelated_ownership_does_not_cover():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues, while Billing probably owns invoices."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_coordinated_verb_phrase_still_covers():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues and routes each issue to the owning team."
    )
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_subject_determiner_denial_recognized():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("No support agent drafts answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_mid_clause_determiner_stays_narrow():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("With no delay we draft answers.")
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_negation_stops_at_causal_boundary():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We do not draft answers because Billing owns refunds."
    )
    assert any(w.startswith("unqualified-ownership-claim:") for w in warnings)


def test_locator_grammar_boundary_probe():
    from atlas_brain.schemas.content_factory import EditorialAuditV2
    from pydantic import ValidationError

    # #2189 changed this boundary deliberately. It used to assert that
    # "sentence 1000000" was admissible against NO body -- grammar alone
    # validated the digit shape, which is what let a raw phone number pass as
    # a locator. The boundary is now the audited body's own length.
    body = "One. Two. Three."

    def _audit_with(locator):
        return EditorialAuditV2.model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "p",
                "edited_body_markdown": body,
                "advisory_warnings": [f"unqualified-answer-claim: sentence {locator}"],
            }
        )

    # In range: the last sentence the body actually has.
    assert _audit_with(3).advisory_warnings

    # Out of range on both sides -- one past the body, and the grammar's own
    # zero floor.
    with pytest.raises(ValidationError):
        _audit_with(1000000)
    with pytest.raises(ValidationError):
        _audit_with(0)


# --- round-15 review fixes ---


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Billing drafts answers for customers.",
        "J. Smith drafts answers for customers.",
        "Acme Inc. drafts answers for customers.",
    ],
)
def test_abbreviation_periods_stay_in_sentence(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert any(
        w == "unqualified-answer-claim: sentence 1" for w in warnings
    ), (text, warnings)


def test_coordinated_adverbs_keep_report_shape():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report clearly and consistently ranks issues by severity."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_coordinated_objects_stay_in_denial_scope():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We do not draft answers or resolutions.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_proposition_coordination_still_splits():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "We draft one answer when evidence exists and another answer regardless."
    )
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


@pytest.mark.parametrize(
    "text",
    [
        "Only when evidence exists, we draft answers.",
        "Only if the tickets contain proof, we draft resolutions.",
    ],
)
def test_focus_modifier_fronted_qualifier_governs(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert not any(
        w.startswith("unqualified-answer-claim:") for w in warnings
    ), (text, warnings)


# --- round-16 review fixes ---


def test_owner_lane_topic_mention_is_not_routing():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues. Each owner lane is reviewed monthly."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


@pytest.mark.parametrize(
    "text",
    [
        "Without evidence we draft answers.",
        "Without delay we draft answers.",
    ],
)
def test_without_complement_keeps_claim(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings), text


def test_without_gerund_denies_action():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We close tickets without drafting answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_coordinator_heavy_draft_stays_fast():
    import time

    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    draft = ("We review tickets and issues and reports and answers daily. " * 600)
    started = time.monotonic()
    advisory_warnings(draft)
    assert time.monotonic() - started < 2.0


# --- round-17 review fixes ---


def test_coordinated_routing_of_non_report_items_does_not_cover():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report ranks issues and routes each invoice to Billing."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


def test_terminal_abbreviation_period_splits_sentence():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We partner with Acme Inc. We draft answers.")
    assert any(
        w == "unqualified-answer-claim: sentence 2" for w in warnings
    ), warnings


def test_internal_abbreviation_still_protected():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("Dr. Billing drafts answers for customers.")
    assert any(w == "unqualified-answer-claim: sentence 1" for w in warnings)


def test_smart_apostrophe_contraction_is_denial():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("We wouldn’t draft answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_negated_qualifier_complement_does_not_excuse():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("If the tickets contain no proof, we draft answers.")
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_affirmative_qualifier_still_excuses():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings("If the tickets contain proof, we draft answers.")
    assert not any(w.startswith("unqualified-answer-claim:") for w in warnings)


def test_same_sentence_routing_scan_is_linear():
    import time

    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    draft = (
        "The report ranks issues, "
        + "Billing owns invoices, " * 2000
        + "daily."
    )
    started = time.monotonic()
    advisory_warnings(draft)
    assert time.monotonic() - started < 2.0


# --- round-18 review fixes ---


@pytest.mark.parametrize(
    "text",
    [
        "If the tickets contain absolutely no proof, we draft answers.",
        "If the tickets contain definitely zero evidence, we draft answers.",
    ],
)
def test_modified_negated_qualifier_does_not_excuse(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert any(w.startswith("unqualified-answer-claim:") for w in warnings), text


def test_same_clause_routing_of_non_items_does_not_cover():
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(
        "The report routes each invoice to Billing and ranks issues."
    )
    assert any(w.startswith("owner-routing-coverage:") for w in warnings)


@pytest.mark.parametrize(
    "text",
    [
        "The Resolution Audit is not provided.",
        "The Resolution Snapshot is never generated.",
        "The action queue is not included.",
    ],
)
def test_denied_product_terms_are_not_report_shape(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    warnings = advisory_warnings(text)
    assert not any(w.startswith("owner-routing-coverage:") for w in warnings), text


def test_repeated_ownership_clause_is_linear():
    import time

    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    draft = "The report ranks issues, " + "Billing owns invoices and " * 4000 + "more."
    started = time.monotonic()
    advisory_warnings(draft)
    assert time.monotonic() - started < 2.5


# --- #2189: casing-independent polarity + body-bound locators -------------

_QUALIFIER_NEGATIONS = ["no", "not", "never", "nothing", "none", "zero"]


@pytest.mark.parametrize("negation", _QUALIFIER_NEGATIONS)
@pytest.mark.parametrize("caser", [str.lower, str.upper, str.title])
def test_negated_qualifier_complement_is_casing_independent(negation, caser):
    """#2189: the qualifier detector was case-insensitive but its complement
    polarity check was not, so `NO proof` read as a qualification and
    suppressed the claim warning. Polarity is not a property of capitalization.
    """
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    text = f"If the tickets contain {caser(negation)} proof, we draft answers."
    assert any(
        w.startswith("unqualified-answer-claim:") for w in advisory_warnings(text)
    ), text


def test_qualified_claim_still_suppressed_across_casings():
    """The other direction: a genuine qualifier must still qualify, in any
    casing, or the fix would just make everything warn."""
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    # Forms the engine genuinely qualifies today, exercised across casings.
    # (The opener/complement pairing is asymmetric on current main -- see the
    # plan's Deferred note -- so this asserts the CASING invariant, which is
    # what #2189 is about, not the pairing.)
    for text in [
        "If the tickets contain proof, we draft answers.",
        "IF THE TICKETS CONTAIN PROOF, we draft answers.",
        "When evidence exists, we draft answers.",
        "WHEN EVIDENCE EXISTS, we draft answers.",
    ]:
        assert not any(
            w.startswith("unqualified-answer-claim:") for w in advisory_warnings(text)
        ), text


def test_locator_cannot_exceed_the_audited_body():
    """#2189 P1: the grammar validated the digit SHAPE only, so a direct writer
    could persist a raw phone number wearing a locator's clothes."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2
    from pydantic import ValidationError

    def _audit_with(body, locator):
        return EditorialAuditV2.model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "p",
                "edited_body_markdown": body,
                "advisory_warnings": [f"unqualified-answer-claim: sentence {locator}"],
            }
        )

    # The reported case: a ten-digit locator against a one-sentence body.
    with pytest.raises(ValidationError, match="locator names sentence"):
        _audit_with("One sentence only.", 2125551234)

    # Boundary, both sides: the body's last sentence is admissible, one past
    # it is not.
    assert _audit_with("One. Two. Three.", 3).advisory_warnings
    with pytest.raises(ValidationError, match="locator names sentence"):
        _audit_with("One. Two. Three.", 4)

    # A blank body admits no locator at all -- there is nothing to name.
    with pytest.raises(ValidationError, match="locator names sentence"):
        _audit_with("", 1)


def test_static_warnings_are_not_locator_bound():
    """Static checklist lines carry no locator, so they stay admissible on a
    body of any length -- including none."""
    from atlas_brain.schemas.content_factory import (
        ADVISORY_CTA_REMINDER,
        ADVISORY_OWNER_ROUTING_WARNING,
        EditorialAuditV2,
    )

    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "edited_body_markdown": "",
            "advisory_warnings": [ADVISORY_CTA_REMINDER, ADVISORY_OWNER_ROUTING_WARNING],
        }
    )
    assert len(audit.advisory_warnings) == 2


def test_variant_locators_bind_to_the_variant_body():
    """Each artifact binds to ITS OWN audited text, not a shared one."""
    from atlas_brain.schemas.content_factory import ChannelVariant
    from pydantic import ValidationError

    ok = ChannelVariant.model_validate(
        {
            "channel": "email",
            "body_markdown": "One. Two.",
            "derived_from_claims": ["e1"],
            "advisory_warnings": ["unqualified-answer-claim: sentence 2"],
        }
    )
    assert ok.advisory_warnings

    with pytest.raises(ValidationError, match="locator names sentence"):
        ChannelVariant.model_validate(
            {
                "channel": "email",
                "body_markdown": "One. Two.",
                "derived_from_claims": ["e1"],
                "advisory_warnings": ["unqualified-answer-claim: sentence 2125551234"],
            }
        )


def test_sentence_definition_is_shared_not_duplicated():
    """The contract and the producer must count sentences identically, or the
    bound either false-rejects or leaves a door open (#2189)."""
    from atlas_brain.schemas.content_factory import sentence_spans
    from atlas_brain.services.content_factory_copy_verification import (
        _sentence_structure,
    )

    for text in [
        "One. Two. Three.",
        "Dr. Billing owns refunds. We draft answers.",
        "Acme Inc. We draft answers.",
        "Intro.\n\nWe draft answers. Really?!",
        "",
    ]:
        assert _sentence_structure(text)[1] == sentence_spans(text), text


# --- #2219 review: invisible-only bodies admit no locator ----------------

_INVISIBLE_BODIES = [
    "\u200b",          # zero width space
    "\ufeff",          # BOM / zero width no-break space
    "\u034f",          # combining grapheme joiner (category Mn)
    "\u00ad",          # soft hyphen
    "  \u200b  ",      # padded with ordinary whitespace
    "\u200b\u200c\ufeff",
]


@pytest.mark.parametrize("body", _INVISIBLE_BODIES)
def test_invisible_only_body_admits_no_locator(body):
    """`str.strip()` leaves a zero-width character intact, so an
    invisible-only body counted as one sentence and admitted `sentence 1`
    even though it renders blank -- fail-open on exactly the input class
    #2201 closed elsewhere."""
    from atlas_brain.schemas.content_factory import EditorialAuditV2, sentence_count
    from pydantic import ValidationError

    assert sentence_count(body) == 0
    with pytest.raises(ValidationError, match="locator names sentence"):
        EditorialAuditV2.model_validate(
            {
                "schema": "editorial_audit.v2",
                "project_id": "p",
                "edited_body_markdown": body,
                "advisory_warnings": ["unqualified-answer-claim: sentence 1"],
            }
        )


def test_invisible_span_between_sentences_does_not_undercount():
    """The other direction: an invisible span in the MIDDLE must not shrink
    the bound, or a legitimate trailing locator would be rejected. This is
    why the bound is the last content INDEX, not the count of content spans.
    """
    from atlas_brain.schemas.content_factory import EditorialAuditV2, sentence_count

    body = "One.\n\n\u200b\n\nThree."
    assert sentence_count(body) == 3
    audit = EditorialAuditV2.model_validate(
        {
            "schema": "editorial_audit.v2",
            "project_id": "p",
            "edited_body_markdown": body,
            "advisory_warnings": ["unqualified-answer-claim: sentence 3"],
        }
    )
    assert audit.advisory_warnings


# --- #2189 findings 2 and 3: routing-coverage scope ----------------------

_ROUTING = "owner-routing-coverage"


def _routing_warns(text):
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    return any(w.startswith(_ROUTING) for w in advisory_warnings(text))


# F2: polarity binds to the predicate, not to every negation later in the clause.
_TRAILING_ADJUNCTS = [
    "without delay",
    "without extra cost",
    "with no delay",
    "before the review",
    "after the sync",
    "for each release",
]


@pytest.mark.parametrize("adjunct", _TRAILING_ADJUNCTS)
def test_trailing_adjunct_does_not_deny_the_report_surface(adjunct):
    """#2189: the polarity range ran to the CLAUSE end, so a bounded negation
    in an unrelated trailing adjunct denied the report surface -- `The
    Resolution Audit is provided without delay.` affirmatively provides the
    report with no routing, and must still warn."""
    assert _routing_warns(f"The Resolution Audit is provided {adjunct}.")


@pytest.mark.parametrize(
    "text",
    [
        "The Resolution Audit is not provided.",
        "The Resolution Audit is never provided.",
        "The Resolution Audit cannot be provided.",
    ],
)
def test_genuine_predicate_negation_still_denies_the_surface(text):
    """The other direction: a negation that DOES govern the product term still
    denies it, or the fix would just make everything warn."""
    assert not _routing_warns(text)


# F3: a quantified subject must carry a report item, not any noun.
_QUANTIFIERS = ["Each", "Every", "All"]
_NON_REPORT_NOUNS = ["invoice", "invoices", "customer", "customers", "vendor"]
_REPORT_NOUNS = ["issue", "issues", "ticket", "tickets", "finding", "findings"]


@pytest.mark.parametrize("quantifier", _QUANTIFIERS)
@pytest.mark.parametrize("noun", _NON_REPORT_NOUNS)
def test_quantified_non_report_subject_does_not_cover(quantifier, noun):
    """`every` alone satisfied the anaphor test, so a statement about invoices
    covered a report about issues."""
    assert _routing_warns(
        f"The report ranks issues. {quantifier} {noun} is assigned to Billing."
    )


@pytest.mark.parametrize("quantifier", _QUANTIFIERS)
@pytest.mark.parametrize("noun", _REPORT_NOUNS)
def test_quantified_report_item_subject_still_covers(quantifier, noun):
    assert not _routing_warns(
        f"The report ranks issues. {quantifier} {noun} is assigned to Billing."
    )


@pytest.mark.parametrize("quantifier", _QUANTIFIERS)
@pytest.mark.parametrize(
    "tail", ["is assigned to Billing", "of them is routed to Billing",
             "one is routed to Billing"]
)
def test_bare_and_pro_form_subjects_still_cover(quantifier, tail):
    """Genuinely bare anaphors and pro-forms keep binding: `Each is assigned`,
    `Each of them is routed`, `Each one is routed`. A pro-form carries no
    lexical content, so it continues the reference rather than renaming it."""
    assert not _routing_warns(f"The report ranks issues. {quantifier} {tail}.")


@pytest.mark.parametrize("quantifier", _QUANTIFIERS)
@pytest.mark.parametrize("noun", _NON_REPORT_NOUNS)
def test_partitive_does_not_smuggle_a_non_report_subject(quantifier, noun):
    """Closing the class rather than the example: admitting `of` wholesale
    would reproduce the same hole one token further out, since `each of the
    invoices` is about invoices exactly as `each invoice` is."""
    assert _routing_warns(
        f"The report ranks issues. {quantifier} of the {noun} is routed to Billing."
    )


@pytest.mark.parametrize("quantifier", _QUANTIFIERS)
@pytest.mark.parametrize("noun", _REPORT_NOUNS)
def test_partitive_report_items_still_cover(quantifier, noun):
    assert not _routing_warns(
        f"The report ranks issues. {quantifier} of the {noun} is routed to Billing."
    )


# --- #2189 round 2: emphatic polarity + grammatical subject head ---------


@pytest.mark.parametrize(
    "emphatic",
    ["is not only provided but current",
     "is not just provided but current",
     "not only ranks issues but drafts answers"],
)
def test_emphatic_not_is_not_a_denial(emphatic):
    """The scope model already treats `not only/just` as affirmative. A second
    verbal-negation matcher disagreed with it and dropped the warning, which
    is the two-definitions failure this module keeps paying for (#2189)."""
    assert _routing_warns(f"The Resolution Audit {emphatic}.")


@pytest.mark.parametrize("noun", ["ticket", "tickets", "issue", "findings"])
@pytest.mark.parametrize("modifier", ["in the report", "for this month", "from Support"])
def test_pp_modified_report_subject_still_covers(noun, modifier):
    """The head is the noun the quantifier BINDS, before any post-modifier.
    Taking the last pre-predicate token read `each ticket in the REPORT` as
    being about reports and regressed valid routing."""
    assert not _routing_warns(
        f"The report ranks issues. Each {noun} {modifier} is assigned to Billing."
    )


@pytest.mark.parametrize("noun", ["invoice", "invoices", "customer"])
@pytest.mark.parametrize("modifier", ["for a ticket", "in the issue log", "for each finding"])
def test_pp_modifier_cannot_smuggle_a_report_item(noun, modifier):
    """The other side: a report-item noun sitting inside a MODIFIER must not
    rescue a non-report subject -- `each invoice for a TICKET` is about
    invoices."""
    assert _routing_warns(
        f"The report ranks issues. Each {noun} {modifier} is assigned to Billing."
    )


def test_routing_scope_pass_stays_linear():
    """Regression guard with a number, not a description: the polarity check
    was quadratic mid-review (7.1s reported, 4.6s measured) on a clause with
    many product terms."""
    import time

    body = "The Resolution Audit " * 3200
    start = time.perf_counter()
    from atlas_brain.services.content_factory_copy_verification import advisory_warnings

    advisory_warnings(body)
    assert time.perf_counter() - start < 1.0


# --- #2189 round 3: existence-bound denial, open modifiers, linear scopes ---


@pytest.mark.parametrize(
    "predicate",
    ["does not include owner assignments", "does not list the owning team",
     "does not name an owner", "does not contain routing"],
)
def test_property_negation_does_not_deny_the_report(predicate):
    """A verbal negation denies the report SURFACE only when it denies the
    report's existence. "does not include owner assignments" says the report
    EXISTS and lacks routing -- suppressing there silenced the checklist
    exactly when routing was absent (#2189 round 3)."""
    assert _routing_warns(f"The Resolution Audit {predicate}.")


@pytest.mark.parametrize(
    "predicate",
    ["is not provided", "is never provided", "cannot be provided",
     "is not available", "was not delivered"],
)
def test_existence_negation_still_denies_the_report(predicate):
    assert not _routing_warns(f"The Resolution Audit {predicate}.")


@pytest.mark.parametrize("modifiers", [
    "open",
    "still open",
    "still currently open high priority",
    "still currently open high priority unresolved support",
    "still currently open high priority unresolved escalated customer support",
])
def test_subject_head_survives_variable_length_modifiers(modifiers):
    """Modifier length is open, so a fixed token window is the wrong shape:
    `tickets` fell outside an eight-token slice and the warning fired on a
    correctly routed draft."""
    assert not _routing_warns(
        f"The report ranks issues. Each of the {modifiers} tickets is assigned to Billing."
    )


def test_scope_lookup_scales_with_negation_scopes_present():
    """The earlier linearity guard used a body that creates NO negation
    scopes, so it could not fail -- it measured the one path already fixed.
    This one generates a scope per repetition."""
    import time

    from atlas_brain.services.content_factory_copy_verification import (
        advisory_warnings,
    )

    def elapsed(reps):
        body = "Resolution Audit with no delay " * reps
        start = time.perf_counter()
        advisory_warnings(body)
        return time.perf_counter() - start

    # 4x the input at the pre-fix quadratic rate took 1.55s; linear is ~0.13s.
    # A generous absolute ceiling keeps this stable on shared CI runners while
    # still failing outright if the scan returns.
    assert elapsed(12800) < 1.0
