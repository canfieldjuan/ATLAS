"""Tests for the read-only / dry-run follow-up tool-surface qualifier.

Boundary-probed both sides: a read tool that should pass, and every known mutating
tool that should fail; plus mixed, unknown, case/whitespace variants, blank/non-string
entries, and the allowlist self-audit. The load-bearing guarantee is that a qualified
surface can perform no send/mutation.
"""

from __future__ import annotations

import pytest

from atlas_brain.schemas.followup_tool_surface import (
    FOLLOWUP_READONLY_TOOLS,
    KNOWN_MUTATING_TOOLS,
    is_read_only_tool,
    qualify_followup_tool_surface,
)


def test_full_readonly_surface_qualifies():
    q = qualify_followup_tool_surface(sorted(FOLLOWUP_READONLY_TOOLS))
    assert q.qualified is True
    assert q.disallowed == () and q.mutating == ()


@pytest.mark.parametrize("tool", sorted(FOLLOWUP_READONLY_TOOLS))
def test_each_readonly_tool_qualifies(tool):
    assert is_read_only_tool(tool) is True
    assert qualify_followup_tool_surface([tool]).qualified is True


@pytest.mark.parametrize("tool", sorted(KNOWN_MUTATING_TOOLS))
def test_each_mutating_tool_disqualifies(tool):
    q = qualify_followup_tool_surface([tool])
    assert q.qualified is False
    assert tool in q.disallowed
    assert tool in q.mutating  # flagged as a known side-effecting tool
    assert is_read_only_tool(tool) is False


def test_mixed_read_and_send_disqualifies_and_names_the_send_tool():
    q = qualify_followup_tool_surface(["get_contact", "send_email", "get_interactions"])
    assert q.qualified is False
    assert q.disallowed == ("send_email",)
    assert q.mutating == ("send_email",)
    assert "send_email" in q.reason


def test_unknown_tool_disqualifies_but_is_not_flagged_mutating():
    q = qualify_followup_tool_surface(["get_contact", "teleport_customer"])
    assert q.qualified is False
    assert q.disallowed == ("teleport_customer",)
    assert q.mutating == ()  # unknown, not a known mutating tool -- still rejected


def test_empty_surface_is_trivially_qualified():
    q = qualify_followup_tool_surface([])
    assert q.qualified is True
    assert q.offered == () and q.disallowed == ()
    assert "no tools offered" in q.reason


@pytest.mark.parametrize("variant", ["Get_Contact", "GET_CONTACT", "get_contact "])
def test_case_and_whitespace_variants(variant):
    # Whitespace is normalized (a padded real tool qualifies); a case variant is a
    # different, non-allowlisted name and fails closed.
    q = qualify_followup_tool_surface([variant])
    if variant.strip() in FOLLOWUP_READONLY_TOOLS:
        assert q.qualified is True
    else:
        assert q.qualified is False


def test_blank_and_non_string_entries_fail_closed():
    q = qualify_followup_tool_surface(["", "   ", None, 42, "get_contact"])
    assert q.qualified is False
    # get_contact is fine; the four malformed entries are all disallowed
    assert len(q.disallowed) == 4


def test_duplicate_offered_tools_dedupe_in_disallowed():
    q = qualify_followup_tool_surface(["send_sms", "send_sms"])
    assert q.qualified is False
    assert q.disallowed == ("send_sms",)


def test_allowlist_is_disjoint_from_known_mutating_tools():
    # The import-time guard enforces this; assert it here too as the second side.
    assert FOLLOWUP_READONLY_TOOLS & KNOWN_MUTATING_TOOLS == frozenset()


def test_no_send_or_write_verb_in_allowlist():
    # Belt-and-suspenders: no allowlisted tool name begins with a mutating verb.
    mutating_prefixes = ("send", "create", "update", "delete", "approve", "record", "void", "mark", "log", "make")
    for tool in FOLLOWUP_READONLY_TOOLS:
        assert not tool.startswith(mutating_prefixes), tool
