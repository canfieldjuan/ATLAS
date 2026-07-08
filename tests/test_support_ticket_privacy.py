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
