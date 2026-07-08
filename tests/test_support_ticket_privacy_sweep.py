"""Adversarial grammar sweep for the support-ticket privacy predicates.

Generates the marker vocabulary from its semantic families (stems x key
structure x audiences x value shapes x containers) in BOTH error directions,
so the closed rule is exercised across the whole space instead of
per-reviewer-finding spellings. Every case is deterministic.
"""

from __future__ import annotations

import itertools

import pytest

from extracted_content_pipeline.support_ticket_privacy import (
    support_ticket_comment_is_private,
    support_ticket_row_is_private,
)

_PRIVATE_STEMS = ("private", "internal", "hidden", "confidential", "restricted")
_PRIVATE_AUDIENCES = ("agent", "agents", "staff", "admin", "team", "support")
_PUBLIC_AUDIENCES = (
    "customer", "customers", "user", "users", "requester", "client", "viewers",
)
_TRUE_VALUES = (True, "true", "yes", 1, "1", "on")
_FALSE_VALUES = (False, "false", "no", 0, "0.0", "off")


def _private_assertion_cases() -> list[tuple[str, object]]:
    cases = []
    for stem, value in itertools.product(_PRIVATE_STEMS, _TRUE_VALUES):
        for form in (stem, f"is_{stem}", f"{stem}_flag", f"is{stem}"):
            cases.append((form, value))
    return cases


@pytest.mark.parametrize(("key", "value"), _private_assertion_cases())
def test_sweep_private_assertion_keys_fail_closed(key: str, value: object) -> None:
    assert support_ticket_comment_is_private({key: value, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "value"),
    [(stem, value) for stem in _PRIVATE_STEMS for value in _FALSE_VALUES],
)
def test_sweep_private_assertion_false_passes(key: str, value: object) -> None:
    assert support_ticket_comment_is_private({key: value, "body": "x"}) is False


@pytest.mark.parametrize(
    "key",
    [f"visible_to_{aud}" for aud in _PRIVATE_AUDIENCES]
    + [f"{aud}_visible" for aud in _PRIVATE_AUDIENCES]
    + [f"visible_{aud}" for aud in _PRIVATE_AUDIENCES]
    + [f"{aud}_only" for aud in _PRIVATE_AUDIENCES]
    + [f"hidden_from_{aud}" for aud in _PUBLIC_AUDIENCES],
)
def test_sweep_private_audience_visibility_keys_fail_closed(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is True


@pytest.mark.parametrize("key", [f"visible_to_{aud}" for aud in _PUBLIC_AUDIENCES])
def test_sweep_public_audience_visibility(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is False
    assert support_ticket_comment_is_private({key: False, "body": "x"}) is True


@pytest.mark.parametrize(
    "key",
    ["not_public", "not_visible", "not_customer_facing", "non_public",
     "not_publicly_visible"],
)
def test_sweep_negated_public_keys_fail_closed(key: str) -> None:
    assert support_ticket_comment_is_private({key: True, "body": "x"}) is True


def test_sweep_negated_private_key_passes() -> None:
    assert support_ticket_comment_is_private(
        {"not_private": True, "body": "x"}
    ) is False


_PRIVATE_PHRASES = (
    "internal", "private", "agents only", "internal only",
    "for internal use only", "restricted to agents", "private to team",
    "agent_note", "staff note", "internal_response", "support staff only",
    "agent-only", "staffonly",
)
_PUBLIC_PHRASES = (
    "public", "customer", "external", "visible to customer", "published",
    "everyone",
)
_LABEL_KEYS = (
    "visibility", "privacy", "audience", "access", "comment_visibility",
    "privacy_label",
)


@pytest.mark.parametrize(
    ("key", "phrase"),
    list(itertools.product(_LABEL_KEYS, _PRIVATE_PHRASES)),
)
def test_sweep_private_label_phrases_fail_closed(key: str, phrase: str) -> None:
    assert support_ticket_comment_is_private({key: phrase, "body": "x"}) is True


@pytest.mark.parametrize(
    ("key", "phrase"),
    list(itertools.product(
        ("visibility", "privacy", "audience", "access"), _PUBLIC_PHRASES,
    )),
)
def test_sweep_public_label_phrases_pass(key: str, phrase: str) -> None:
    assert support_ticket_comment_is_private({key: phrase, "body": "x"}) is False


@pytest.mark.parametrize(
    "kind_value",
    ["private_note", "internal_comment", "agent_note", "staff_reply",
     "private_response", "support_agent_note"],
)
def test_sweep_private_kind_values_fail_closed(kind_value: str) -> None:
    assert support_ticket_comment_is_private(
        {"type": kind_value, "body": "x"}
    ) is True


@pytest.mark.parametrize(
    "kind_value",
    ["Comment", "question", "customer_note", "customer_reply", "incident"],
)
def test_sweep_public_kind_values_pass(kind_value: str) -> None:
    assert support_ticket_comment_is_private(
        {"type": kind_value, "body": "x"}
    ) is False


@pytest.mark.parametrize("stem", ["private", "internal", "hidden"])
def test_sweep_object_wrapped_polarity(stem: str) -> None:
    assert support_ticket_comment_is_private(
        {stem: {"value": True}, "body": "x"}
    ) is True
    assert support_ticket_comment_is_private(
        {stem: {"value": False}, "body": "x"}
    ) is False


@pytest.mark.parametrize("label", ["internal", "private", "agents only"])
def test_sweep_object_label_subfields_fail_closed(label: str) -> None:
    assert support_ticket_comment_is_private(
        {"visibility": {"name": label}, "body": "x"}
    ) is True


def test_sweep_object_public_forms_pass() -> None:
    assert support_ticket_comment_is_private(
        {"public": {"value": True}, "body": "x"}
    ) is False
    assert support_ticket_comment_is_private(
        {"visibility": {"name": "public", "value": True}, "body": "x"}
    ) is False


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("internal_id", "x-1"),
        ("private_ip", "10.0.0.1"),
        ("publication_date", "2026-01-01"),
        ("hidden_count", 3),
        ("internal_status", "open"),
        ("has_access", True),
        ("support_ticket_id", "T-1"),
        ("end_date", "2026-01-01"),
        ("from_email", "a@b.example"),
        ("external_id", "z"),
        ("team_id", "t-9"),
        ("client_version", "1.2.3"),
        ("user_agent", "Mozilla"),
        ("customer_id", "c-1"),
        ("admin_url", "https://example.test"),
    ],
)
def test_sweep_data_columns_are_never_markers(column: str, value: object) -> None:
    assert support_ticket_row_is_private({"body": "x", column: value}) is False


@pytest.mark.parametrize(
    "column",
    ["public_comment", "public_comments", "private_note", "internal_notes",
     "agent_note", "staff_notes"],
)
def test_sweep_content_columns_do_not_reject_rows(column: str) -> None:
    assert support_ticket_row_is_private(
        {"body": "x", column: "free text content here"}
    ) is False
