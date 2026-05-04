"""Verify atlas's ``ReasoningAgentState`` is a TypedDict refinement of
``extracted_reasoning_core.state.ReasoningAgentState``.

PR-C4b establishes the typing seam: anything the core graph engine
declares it consumes (via the core TypedDict) must accept an atlas state
literal without conversion. These tests pin that contract so a future
refactor can't accidentally break the is-a relationship by reverting
atlas's TypedDict to a sibling definition.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import get_type_hints

from atlas_brain.reasoning.state import ReasoningAgentState
from extracted_reasoning_core.state import ReasoningAgentState as CoreReasoningAgentState

_ATLAS_STATE_SRC = (
    Path(__file__).resolve().parent.parent
    / "atlas_brain"
    / "reasoning"
    / "state.py"
)


def test_atlas_state_source_declares_core_typeddict_as_base() -> None:
    # Source-level inheritance check: parse atlas's state.py and verify
    # ReasoningAgentState's class declaration lists the core TypedDict as
    # a base. We pin this at the AST level rather than via runtime
    # markers (``__orig_bases__`` / ``__mro__``) because TypedDict's
    # runtime representation varies by Python version -- 3.11+ exposes
    # ``__orig_bases__`` but 3.10 doesn't, and the runtime class is just
    # ``dict`` either way. The seam is the source-level declaration.
    tree = ast.parse(_ATLAS_STATE_SRC.read_text())
    classdef = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "ReasoningAgentState"
    )
    base_names = {base.id for base in classdef.bases if isinstance(base, ast.Name)}
    assert "_CoreReasoningAgentState" in base_names, (
        f"atlas ReasoningAgentState must declare _CoreReasoningAgentState as a base; "
        f"saw bases={sorted(base_names)}"
    )


def test_atlas_state_inherits_every_core_field() -> None:
    # Every key declared on core's TypedDict must be reachable on atlas's
    # TypedDict. This guards against a future "let's just redeclare the
    # fields on atlas" refactor that would silently break the seam.
    core_keys = set(get_type_hints(CoreReasoningAgentState).keys())
    atlas_keys = set(get_type_hints(ReasoningAgentState).keys())
    missing = core_keys - atlas_keys
    assert not missing, f"atlas state missing core fields: {missing}"


def test_atlas_state_adds_expected_atlas_specific_fields() -> None:
    # The point of the subclass is to extend core with atlas's granular
    # context slots. Pin the extension surface so a refactor can't drop
    # one of them without an explicit test update.
    atlas_keys = set(get_type_hints(ReasoningAgentState).keys())
    core_keys = set(get_type_hints(CoreReasoningAgentState).keys())
    atlas_extension = atlas_keys - core_keys

    expected = {
        "crm_context",
        "email_history",
        "voice_turns",
        "calendar_events",
        "sms_messages",
        "graph_facts",
        "recent_events",
        "market_context",
        "news_context",
        "b2b_churn",
    }
    assert expected.issubset(atlas_extension), (
        f"atlas state missing extension fields: {expected - atlas_extension}"
    )


def test_atlas_state_literal_is_assignable_to_core_typed_param() -> None:
    # Round-trip a literal through a function annotated with the core
    # TypedDict. At runtime TypedDicts are dicts, so this verifies the
    # values flow without rejection -- the static-typing intent is that
    # the literal type-checks against the core annotation, which the
    # subclass makes true by construction.
    def _accepts_core(state: CoreReasoningAgentState) -> str:
        return state.get("event_id", "")

    atlas_state: ReasoningAgentState = {
        "event_id": "evt-1",
        "event_type": "vendor.archetype_assigned",
        "source": "reasoning_core",
        "b2b_churn": {"vendor": "Acme"},
    }
    assert _accepts_core(atlas_state) == "evt-1"


def test_atlas_state_token_tracking_smoke() -> None:
    # Mirrors the literal construction in atlas_brain/test_token_tracking.py
    # (lines 117-122). Smoke-tests that token-tracking fields still flow
    # through the inherited TypedDict after the inheritance change.
    state: ReasoningAgentState = {
        "event_id": "evt-token",
        "total_input_tokens": 100,
        "total_output_tokens": 50,
    }
    assert state["total_input_tokens"] == 100
    assert state["total_output_tokens"] == 50


def test_atlas_state_atlas_specific_field_construction() -> None:
    # Construct with only atlas-specific fields populated -- ensures the
    # extension surface is real and not a phantom.
    state: ReasoningAgentState = {
        "crm_context": {"contact_id": "c1"},
        "b2b_churn": {"score": 0.8},
        "voice_turns": [{"role": "user", "text": "hi"}],
    }
    assert state["crm_context"] == {"contact_id": "c1"}
    assert state["b2b_churn"] == {"score": 0.8}
    assert state["voice_turns"][0]["text"] == "hi"
