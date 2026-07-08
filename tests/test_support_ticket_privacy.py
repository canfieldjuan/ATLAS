from __future__ import annotations

import pytest

from extracted_content_pipeline.support_ticket_privacy import (
    support_ticket_comment_is_private,
    support_ticket_row_is_private,
)


@pytest.mark.parametrize("marker", [
    {"public": False},
    {"public": "false"},
    {"public": "0.0"},
    {"public": "0e0"},
    {"public": 2},
    {"public": "maybe"},
    {"is_private": True},
    {"is_private": "1.0"},
    {"is_private": 2},
    {"private_note": "yes"},
    {"is_private_note": "true"},
    {"internal_comment": "true"},
    {"is_internal_comment": "yes"},
    {"visibility": "internal"},
    {"visibility": {"name": "internal"}},
    {"privacy": "private"},
    {"privacy": {"label": "private"}},
    {"type": "private_note"},
    {"type": "private_reply"},
    {"message_type": "internal_reply"},
    {"Public": False, "public": True},
])
def test_support_ticket_comment_private_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True


@pytest.mark.parametrize("marker", [
    {},
    {"public": True},
    {"public": "true"},
    {"public": "1.0"},
    {"public": "1e0"},
    {"is_public": "yes"},
    {"visibility": "public"},
    {"visibility": {"name": "public"}},
    {"privacy": "customer"},
    {"privacy": {"label": "customer"}},
    {"private": False},
    {"internal": "0.0"},
])
def test_support_ticket_comment_public_markers_pass(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is False


def test_support_ticket_row_private_check_excludes_comment_alias_columns() -> None:
    row = {
        "ticket_id": "T-1",
        "subject": "How do I export reports?",
        "private_note": "Manager-only context",
    }

    assert support_ticket_row_is_private(row) is False


def test_support_ticket_row_private_check_ignores_audience_metadata() -> None:
    row = {
        "ticket_id": "T-1",
        "subject": "How do I export reports?",
        "description": "Where does the export button live?",
        "audience": "enterprise admins",
    }

    assert support_ticket_row_is_private(row) is False


def test_support_ticket_row_private_check_ignores_access_metadata() -> None:
    row = {
        "ticket_id": "T-1",
        "subject": "How do I update billing access?",
        "description": "Account access still needs the billing role.",
        "access": "Account access",
    }

    assert support_ticket_row_is_private(row) is False


@pytest.mark.parametrize("marker", [
    {"public": "false"},
    {"private": "yes"},
    {"visibility": "restricted"},
    {"visibility": {"name": "internal"}},
    {"access": "internal"},
    {"access": "restricted"},
    {"access": "restricted access"},
    {"audience": "agent only"},
    {"audience": "restricted"},
    {"type": "internal_note"},
    {"Public": False, "public": True},
])
def test_support_ticket_row_private_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private(marker) is True


@pytest.mark.parametrize("marker", [
    {"public": "not-a-number"},
    {"private": "not-a-number"},
    {"visibility": "agent-only"},
    {"visibility": {"id": 47}},
])
def test_support_ticket_privacy_unknown_markers_do_not_use_numeric_exceptions(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True


# Round-4 design re-cut: closed token-stem rule instead of key enumeration.
# Each case class below traces to a reviewed finding or an over-rejection
# guard; the rule -- not a key list -- must classify every member.


@pytest.mark.parametrize("marker", [
    {"is_hidden": True},
    {"is_nonpublic": True},
    {"is_public_comment": False},
    {"public_comment": False},
    {"public_reply": False},
    {"publicComment": "false"},
    {"isprivate": "true"},
    {"privatenotes": "yes"},
    {"visible_to_customer": False},
    {"agent_only": True},
    {"is_confidential": True},
])
def test_support_ticket_comment_alias_class_fails_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True


@pytest.mark.parametrize("marker", [
    {"visible_to_customer": True},
    {"is_public_comment": True},
    {"public_reply": "yes"},
])
def test_support_ticket_comment_alias_class_public_passes(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is False


@pytest.mark.parametrize("marker", [
    {"body": "How do I export?", "is_private_note": True},
    {"body": "How do I export?", "is_internal_comment": "yes"},
    {"body": "How do I export?", "private_note": 1},
    {"body": "How do I export?", "internal_use_only": True},
])
def test_support_ticket_row_flag_valued_note_aliases_reject(
    marker: dict[str, object],
) -> None:
    assert support_ticket_row_is_private(marker) is True


@pytest.mark.parametrize("marker", [
    {"visibility": {"name": "public", "value": "internal"}},
    {"privacy": {"label": "customer", "kind": "internal"}},
    {"visibility": {"id": 47, "note": "free text"}},
])
def test_support_ticket_conflicting_object_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True


def test_support_ticket_consistent_public_object_marker_passes() -> None:
    marker = {"visibility": {"name": "public", "value": "public"}}
    assert support_ticket_comment_is_private(marker) is False


@pytest.mark.parametrize("marker", [
    {"public": "1e999999999999999999999"},
    {"public": "0e99999999999999999999999999"},
    {"private": "1e999999999999999999999"},
    {"visibility": "9" * 100000},
])
def test_support_ticket_malformed_numeric_markers_fail_closed_without_raising(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True
    assert support_ticket_row_is_private({"body": "x", **marker}) is True


@pytest.mark.parametrize("row", [
    {"body": "x", "internal_id": "abc-123"},
    {"body": "x", "private_ip": "10.0.0.1"},
    {"body": "x", "publication_date": "2026-01-01"},
    {"body": "x", "hidden_count": 3},
    {"body": "x", "internal_status": "open"},
    {"body": "x", "has_access": True},
    {"body": "x", "external_id": "z-9"},
])
def test_support_ticket_data_columns_are_not_privacy_markers(
    row: dict[str, object],
) -> None:
    assert support_ticket_row_is_private(row) is False
    assert support_ticket_comment_is_private(row) is False


def test_support_ticket_privacy_predicates_never_raise() -> None:
    from collections.abc import Mapping

    class RaisingMapping(Mapping):
        def __getitem__(self, key: str) -> object:
            raise RuntimeError("boom")

        def __iter__(self):
            raise RuntimeError("boom")

        def __len__(self) -> int:
            return 1

    assert support_ticket_comment_is_private(RaisingMapping()) is True
    assert support_ticket_row_is_private(RaisingMapping()) is True


# Round-5 refinements: content-column carveout for public comment keys,
# fail-closed empty object markers, private-audience key flip, plural
# audience labels, and label-suffixed privacy keys.


@pytest.mark.parametrize("row", [
    {"subject": "s", "public_comment": "customer wrote a thing"},
    {"subject": "s", "public_comments": "first reply\nsecond reply"},
])
def test_support_ticket_public_comment_content_columns_stay_admitted(
    row: dict[str, object],
) -> None:
    assert support_ticket_row_is_private(row) is False
    assert support_ticket_comment_is_private(row) is False


@pytest.mark.parametrize("marker", [
    {"public_comment": "false"},
    {"public_comment": False},
    {"public_comments": 0},
])
def test_support_ticket_flag_valued_public_comment_keys_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private(marker) is True


@pytest.mark.parametrize("marker", [
    {"privacy": {}},
    {"visibility": {}},
])
def test_support_ticket_empty_object_markers_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({"body": "x", **marker}) is True
    assert support_ticket_row_is_private({"body": "x", **marker}) is True


@pytest.mark.parametrize("marker", [
    {"audience": "agents only"},
    {"access": "agents only"},
])
def test_support_ticket_plural_agent_audience_labels_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({"body": "x", **marker}) is True


@pytest.mark.parametrize("marker", [
    {"visible_to_agents": True},
    {"visible_to_staff": True},
    {"visible_to_admins": "yes"},
])
def test_support_ticket_private_audience_visibility_keys_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({"body": "x", **marker}) is True
    assert support_ticket_row_is_private({"body": "x", **marker}) is True


def test_support_ticket_customer_audience_visibility_still_passes() -> None:
    assert support_ticket_comment_is_private(
        {"body": "x", "visible_to_customer": True}
    ) is False


@pytest.mark.parametrize("marker", [
    {"privacy_label": "private"},
    {"visibility_status": "internal"},
    {"access_label": "restricted access"},
])
def test_support_ticket_label_suffixed_privacy_keys_fail_closed(
    marker: dict[str, object],
) -> None:
    assert support_ticket_comment_is_private({"body": "x", **marker}) is True


@pytest.mark.parametrize("row", [
    {"body": "x", "visibility_status": "public"},
    {"body": "x", "internal_status": "open"},
    {"body": "x", "access_label": "billing role"},
])
def test_support_ticket_label_suffix_rule_does_not_over_reject(
    row: dict[str, object],
) -> None:
    assert support_ticket_row_is_private(row) is False
