"""Runtime guard tests (v2 S3, #1931).

Nothing is faked: the real guard over the real catalogue, fed by the real
parser. The parity test is the slice's centerpiece -- the guard's policy
must partition the catalogue exactly, so upstream rule additions cannot
ship unclassified.
"""

from __future__ import annotations

import pytest

from atlas_reddit.fit import parse_fit_decision
from atlas_reddit.fit_guard import (
    ADVISORY_CODES,
    BLOCKING_CODES,
    GuardDecision,
    guard_fit_decision,
)
from atlas_reddit.fit_rules import ALL_RULE_CODES, RULES


def _decision(reason: str, angle: str | None, verdict: str = "yes"):
    return parse_fit_decision(
        {
            "verdict": verdict,
            "reason": reason,
            "angle": angle,
            "risk_flags": [],
        }
    )


# -- parity: the policy partitions the catalogue ---------------------------------


def test_policy_partitions_the_catalogue_exactly() -> None:
    """Every rule code is classified, none twice: a rule added in any
    later slice without a policy decision fails THIS test, not silently
    ships."""
    assert BLOCKING_CODES | ADVISORY_CODES == ALL_RULE_CODES
    assert not (BLOCKING_CODES & ADVISORY_CODES)


def test_all_families_block_in_v1() -> None:
    assert BLOCKING_CODES == ALL_RULE_CODES
    assert ADVISORY_CODES == frozenset()


# -- both sides -------------------------------------------------------------------


def test_clean_advisory_decision_passes() -> None:
    decision = _decision(
        "They describe repeated onboarding questions arriving despite "
        "existing documentation.",
        "Ask what their ticket history shows about which questions keep "
        "coming back; that evidence is worth examining.",
    )
    outcome = guard_fit_decision(decision)
    assert outcome == GuardDecision(ok=True, codes=())


def test_clean_no_verdict_passes() -> None:
    decision = _decision(
        "A hiring post, not a support-operations conversation.", None, "no"
    )
    assert guard_fit_decision(decision).ok is True


@pytest.mark.parametrize(
    ("angle", "expected_code"),
    [
        (
            "An audit could cut their support tickets by 40% and guarantee "
            "a lasting reduction.",
            "GUARANTEED_DEFLECTION",
        ),
        ("Lead with the ROI story and the hours saved.", "ROI_SAVINGS"),
        (
            "Mention that our tool covers this and offer a free trial.",
            "SELF_PROMO_PITCH",
        ),
        (
            "Hey OP, I'd start by auditing those emails -- feel free to DM me.",
            "REPLY_DRAFT",
        ),
        ("Reply to the thread with a clarifying question.", "WRITE_ACTION_POSTURE"),
        (
            "Compare what reaches jane.doe@example.com against the queue.",
            "PII_EMAIL",
        ),
    ],
)
def test_each_family_blocks(angle: str, expected_code: str) -> None:
    decision = _decision("They describe repeated support questions.", angle)
    outcome = guard_fit_decision(decision)
    assert outcome.ok is False
    assert expected_code in outcome.codes


def test_reason_is_guarded_too_not_just_angle() -> None:
    decision = _decision(
        "Their support lead jane.doe@example.com tracks repeat questions.",
        "Ask what the tracking sheet shows.",
    )
    outcome = guard_fit_decision(decision)
    assert outcome.ok is False
    assert "PII_EMAIL" in outcome.codes


def test_angle_start_greeting_blocks() -> None:
    """Fields are scanned separately: a greeting OPENING the angle fires
    even though it would be mid-string in any concatenation (S1 lesson)."""
    decision = _decision(
        "They describe repeated support questions.",
        "Hey OP, you might look at the ticket history first.",
    )
    outcome = guard_fit_decision(decision)
    assert outcome.ok is False
    assert "REPLY_DRAFT" in outcome.codes


def test_codes_are_sorted_deduped_and_text_free() -> None:
    decision = _decision(
        "Reach them at jane@example.com or bob@example.com for details.",
        "We guarantee a 40% reduction in tickets and clear ROI.",
    )
    outcome = guard_fit_decision(decision)
    assert outcome.codes == tuple(sorted(set(outcome.codes)))
    assert "jane@example.com" not in " ".join(outcome.codes)
    assert {"PII_EMAIL", "GUARANTEED_DEFLECTION", "ROI_SAVINGS"} <= set(
        outcome.codes
    )


def test_allowlist_parity_with_the_harness() -> None:
    """The same suppression mechanism the harness uses per-fixture exists
    for parity; runtime callers pass nothing and get an empty default."""
    decision = _decision(
        "They describe repeated support questions.",
        "Ask what reaches support@acme-widgets.com versus the phone line.",
    )
    assert guard_fit_decision(decision).ok is False
    allowed = guard_fit_decision(
        decision, pii_allowlist=frozenset({"support@acme-widgets.com"})
    )
    assert allowed.ok is True
    assert allowed.codes == ()


def test_guard_consumes_only_parsed_decisions() -> None:
    """The guard sits BEHIND the parser: contract enforcement (shape,
    lengths, verdict-conditional angle) already happened. Guarding a
    clean parse of maximal length must not error."""
    decision = _decision("x" * 280, "y" * 280)
    assert guard_fit_decision(decision).ok is True
