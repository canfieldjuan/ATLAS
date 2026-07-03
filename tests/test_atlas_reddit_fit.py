"""Fit contract + prompt builder tests (v2 S2, #1931).

S2 is pure -- no external boundary exists, so nothing is faked: the real
parser, the real prompt builder, the real rule catalogue, and the S1
harness running through the swapped-in parser core.
"""

from __future__ import annotations

import json

import pytest

from atlas_reddit.fit import (
    FIT_OUTPUT_JSON_SCHEMA,
    PROMPT_VERSION,
    FitDecision,
    FitParseError,
    build_fit_prompt,
    fit_decision_problems,
    parse_fit_decision,
)
from atlas_reddit.fit_eval import check_prediction_shape
from atlas_reddit.fit_rules import (
    FIT_RISK_FLAGS,
    FIT_VERDICTS,
    PARSE_ERROR_CODES,
    RULES,
)


def _prediction(**overrides) -> dict:
    base = {
        "verdict": "yes",
        "reason": "They describe repeated support questions.",
        "angle": "Ask what their ticket evidence shows.",
        "risk_flags": [],
    }
    base.update(overrides)
    return base


# -- parser: acceptance ---------------------------------------------------------


def test_parses_canonical_yes() -> None:
    decision = parse_fit_decision(json.dumps(_prediction()))
    assert decision == FitDecision(
        verdict="yes",
        reason="They describe repeated support questions.",
        angle="Ask what their ticket evidence shows.",
        risk_flags=(),
    )


def test_parses_no_with_null_angle_and_canonicalizes_empty() -> None:
    for angle in (None, "", "   "):
        decision = parse_fit_decision(
            _prediction(verdict="no", angle=angle, risk_flags=["vendor_bait"])
        )
        assert decision.angle is None
        assert decision.risk_flags == ("vendor_bait",)


def test_parser_collapses_whitespace() -> None:
    decision = parse_fit_decision(
        _prediction(reason="  spread \n across \t lines  ")
    )
    assert decision.reason == "spread across lines"


def test_parses_fenced_json() -> None:
    fenced = "```json\n" + json.dumps(_prediction()) + "\n```"
    assert parse_fit_decision(fenced).verdict == "yes"


# -- parser: rejection (codes ride in prediction envelopes) ----------------------


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("", "model_empty_response"),
        ("   \n ", "model_empty_response"),
        ("The thread looks like a fit to me.", "model_output_invalid_json"),
        ('{"verdict": "yes",', "model_output_invalid_json"),
        ('["yes"]', "model_output_schema_mismatch"),
        ('"yes"', "model_output_schema_mismatch"),
        ("42", "model_output_schema_mismatch"),
    ],
)
def test_parser_rejects_non_contract_strings(raw: str, code: str) -> None:
    with pytest.raises(FitParseError) as excinfo:
        parse_fit_decision(raw)
    assert excinfo.value.code == code
    assert excinfo.value.code in PARSE_ERROR_CODES


def test_shape_failures_carry_problem_codes() -> None:
    with pytest.raises(FitParseError) as excinfo:
        parse_fit_decision(_prediction(verdict="definitely", confidence=0.9))
    assert excinfo.value.code == "model_output_schema_mismatch"
    assert "verdict_invalid" in excinfo.value.problems
    assert "unknown_keys" in excinfo.value.problems


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"confidence": 0.9}, "unknown_keys"),
        ({"verdict": "definitely"}, "verdict_invalid"),
        ({"reason": "   "}, "reason_missing"),
        ({"reason": "x" * 281}, "reason_too_long"),
        ({"angle": None}, "angle_required"),
        ({"angle": "x" * 281}, "angle_too_long"),
        ({"risk_flags": "promo_risk"}, "risk_flags_invalid"),
        ({"risk_flags": [{"code": "x"}]}, "risk_flags_invalid"),
        ({"risk_flags": ["made_up_flag"]}, "risk_flag_unknown"),
        ({"risk_flags": ["pii_risk", "pii_risk"]}, "risk_flag_duplicate"),
    ],
)
def test_problem_codes_stable_across_the_contract(
    mutation: dict, expected_code: str
) -> None:
    assert expected_code in fit_decision_problems(_prediction(**mutation))


def test_missing_keys_and_no_with_angle() -> None:
    assert "missing_keys" in fit_decision_problems({"verdict": "yes"})
    assert "angle_forbidden_for_no" in fit_decision_problems(
        _prediction(verdict="no", angle="but here is an angle")
    )
    assert fit_decision_problems(_prediction(verdict="no", angle=None)) == ()


# -- the harness measures THIS parser --------------------------------------------


def test_harness_shape_checker_is_the_parser_core() -> None:
    """The S1 ruler and the runtime contract are mechanically one: the
    harness's checker delegates to fit_decision_problems."""
    bad = _prediction(verdict="definitely", confidence=0.9)
    assert check_prediction_shape(bad) == fit_decision_problems(bad)
    good = _prediction()
    assert check_prediction_shape(good) == fit_decision_problems(good) == ()


# -- prompt builder ---------------------------------------------------------------


def _messages(**overrides) -> tuple[dict, ...]:
    kwargs: dict = dict(
        title="Customers keep opening tickets even though we have docs",
        subreddit="CustomerSuccess",
        body="We have KB articles but users still ask the same questions.",
        matched_topics=("repeat-tickets", "help-center"),
        keyword_score=3.0,
        reddit_score=12,
        num_comments=8,
    )
    kwargs.update(overrides)
    return build_fit_prompt(**kwargs)


def test_prompt_embeds_every_catalogue_rule_message() -> None:
    """The told-graded-blocked loop is mechanical: every rule message in
    the catalogue appears verbatim as a prompt boundary bullet."""
    system = _messages()[0]["content"]
    for rule in RULES:
        assert rule.message in system, rule.code


def test_prompt_embeds_posture_and_contract_lines() -> None:
    system = _messages()[0]["content"]
    for needle in (
        "read-only",
        "never draft reply text",
        "does NOT prove future ticket reduction",
        "Do not pitch.",
        '"verdict": "yes|maybe|no"',
        "angle: null",
    ):
        assert needle in system, needle
    for flag in FIT_RISK_FLAGS:
        assert flag in system, flag


def test_prompt_user_message_carries_the_candidate() -> None:
    user = _messages()[1]["content"]
    for needle in (
        "r/CustomerSuccess",
        "Customers keep opening tickets",
        "repeat-tickets, help-center",
        "Keyword score: 3.0",
        "Comments: 8",
    ):
        assert needle in user, needle


def test_prompt_without_body_marks_it_and_invites_low_context() -> None:
    user = build_fit_prompt(title="A title", subreddit="sub", body="  ")[1][
        "content"
    ]
    assert "[no body available" in user
    assert "low_context" in user


def test_prompt_is_deterministic_and_ascii() -> None:
    first, second = _messages(), _messages()
    assert first == second
    for message in first:
        message["content"].encode("ascii")


# -- wire schema -------------------------------------------------------------------


def test_schema_is_strict_mode_safe() -> None:
    """Only keywords every strict-mode backend accepts; the parser is the
    authoritative gate for everything else (length caps etc.)."""
    allowed = {"type", "enum", "required", "additionalProperties", "properties", "items"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            assert set(node) <= allowed, set(node) - allowed
            for key, value in node.items():
                if key == "properties":
                    # keys here are field names, not schema keywords
                    for field_schema in value.values():
                        walk(field_schema)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(FIT_OUTPUT_JSON_SCHEMA)
    assert FIT_OUTPUT_JSON_SCHEMA["additionalProperties"] is False
    assert set(FIT_OUTPUT_JSON_SCHEMA["required"]) == {
        "verdict", "reason", "angle", "risk_flags",
    }
    assert FIT_OUTPUT_JSON_SCHEMA["properties"]["verdict"]["enum"] == list(
        FIT_VERDICTS
    )
    json.dumps(FIT_OUTPUT_JSON_SCHEMA)  # serializable as sent on the wire


def test_prompt_version_exported() -> None:
    assert PROMPT_VERSION == "fit.v1"
