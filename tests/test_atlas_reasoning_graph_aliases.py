"""Pin atlas's backward-compat re-exports of the core graph helpers.

PR-C4e1 promoted the helpers into ``extracted_reasoning_core.graph_helpers``
and ``atlas_brain.reasoning.graph`` re-exports them under the existing
private names (``_parse_llm_json``, ``_node_plan_actions``, etc.) so
internal callers like ``reflection.py`` don't need to change import
sites. Without explicit coverage, a typo in the re-export layer would
slip through CI -- existing atlas tests pin
``_build_notification_fallback`` / ``_sanitize_notification_summary``
/ ``_valid_uuid`` indirectly, but ``_parse_llm_json`` /
``_clean_summary_text`` / ``_has_suspicious_trailing_fragment`` /
``_node_plan_actions`` are unpinned.

These tests use ``is`` identity checks rather than just ``hasattr``:
the alias must point to the exact same callable as the core export, so
a future refactor that accidentally redefines the helper inside
``atlas_brain.reasoning.graph`` (instead of importing) would surface
here.
"""

from __future__ import annotations

from atlas_brain.reasoning import graph as atlas_graph
from extracted_reasoning_core import graph_helpers as core_helpers


def test_parse_llm_json_alias_identity() -> None:
    assert atlas_graph._parse_llm_json is core_helpers.parse_llm_json


def test_valid_uuid_alias_identity() -> None:
    assert atlas_graph._valid_uuid is core_helpers.valid_uuid_str


def test_clean_summary_text_alias_identity() -> None:
    assert atlas_graph._clean_summary_text is core_helpers.clean_summary_text


def test_has_suspicious_trailing_fragment_alias_identity() -> None:
    assert (
        atlas_graph._has_suspicious_trailing_fragment
        is core_helpers.has_suspicious_trailing_fragment
    )


def test_build_notification_fallback_alias_identity() -> None:
    assert (
        atlas_graph._build_notification_fallback
        is core_helpers.build_notification_fallback
    )


def test_sanitize_notification_summary_alias_identity() -> None:
    assert (
        atlas_graph._sanitize_notification_summary
        is core_helpers.sanitize_notification_summary
    )


def test_node_plan_actions_alias_identity() -> None:
    assert atlas_graph._node_plan_actions is core_helpers.plan_actions
