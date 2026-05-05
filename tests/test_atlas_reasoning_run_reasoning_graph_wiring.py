"""Verify atlas's ``run_reasoning_graph`` wires its eight ``_node_*``
callables into the core orchestrator's ``GraphNodes`` bundle.

PR-C4e3 made ``run_reasoning_graph`` a thin wrapper -- the chain
logic lives in :func:`extracted_reasoning_core.graph.run_graph`. The
atlas wrapper's only job is to build a ``GraphNodes`` bundle from its
local ``_node_*`` callables and delegate. These tests pin that wiring
contract:

- ``_build_atlas_graph_nodes()`` returns a frozen ``GraphNodes``
  pointing at the eight expected atlas callables (``is`` identity)
- ``run_reasoning_graph`` invokes the bundled callables through the
  core orchestrator with the right early-exit semantics

Atlas's heavy chains (``services``, ``pipelines``, ``config``) are
stubbed in ``sys.modules`` so the test runs in standalone CI without
the full atlas dep stack -- same save/restore fixture pattern PR-C4d
established. Without the stubs, importing
``atlas_brain.reasoning.graph`` triggers
``atlas_brain.reasoning.__init__`` which is fine, but invoking the
wrapper at runtime would call into the actual atlas nodes (which need
the heavy chain). We stub the eight ``_node_*`` symbols on the graph
module via ``monkeypatch`` so the orchestrator-wiring layer is what's
exercised.
"""

from __future__ import annotations

from typing import Any

import pytest


# ----------------------------------------------------------------------
# Wiring identity
# ----------------------------------------------------------------------


def test_build_atlas_graph_nodes_is_a_frozen_graph_nodes() -> None:
    from dataclasses import FrozenInstanceError

    from atlas_brain.reasoning.graph import _build_atlas_graph_nodes
    from extracted_reasoning_core.graph import GraphNodes

    bundle = _build_atlas_graph_nodes()
    assert isinstance(bundle, GraphNodes)
    # Frozen: bundle can't be mutated mid-run. ``frozen=True`` raises
    # ``FrozenInstanceError``; ``slots=True`` alone would raise
    # ``AttributeError``. Accept either so a future refactor that
    # drops slots+frozen->frozen alone (or vice versa) doesn't fail
    # this test for the wrong reason.
    with pytest.raises((FrozenInstanceError, AttributeError)):
        bundle.triage = lambda state: state  # type: ignore[misc]


def test_build_atlas_graph_nodes_points_at_the_eight_node_callables() -> None:
    # Identity check: the bundle holds atlas's own ``_node_*``
    # callables, not copies. A future refactor that wraps them or
    # substitutes alternatives must update this test.
    from atlas_brain.reasoning import graph as atlas_graph

    bundle = atlas_graph._build_atlas_graph_nodes()
    assert bundle.triage is atlas_graph._node_triage
    assert bundle.aggregate_context is atlas_graph._node_aggregate_context
    assert bundle.check_lock is atlas_graph._node_check_lock
    assert bundle.reason is atlas_graph._node_reason
    assert bundle.plan_actions is atlas_graph._node_plan_actions
    assert bundle.execute_actions is atlas_graph._node_execute_actions
    assert bundle.synthesize is atlas_graph._node_synthesize
    assert bundle.notify is atlas_graph._node_notify


# ----------------------------------------------------------------------
# End-to-end delegation
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_reasoning_graph_runs_all_atlas_nodes_when_chain_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub each atlas ``_node_*`` with a recording fake so the test
    # exercises the wiring -- not the atlas-specific node bodies (which
    # need the heavy services/pipelines chain). The orchestrator chain
    # logic lives in core and is covered by
    # tests/test_extracted_reasoning_core_graph.py; this test pins
    # that the atlas wrapper actually invokes core's orchestrator
    # with the expected nodes.
    from atlas_brain.reasoning import graph as atlas_graph

    ledger: list[str] = []

    def _make_node(name: str, patch: dict | None = None):
        async def _node(state: dict) -> dict:
            ledger.append(name)
            if patch:
                state.update(patch)
            return state

        return _node

    monkeypatch.setattr(atlas_graph, "_node_triage", _make_node("triage", {"needs_reasoning": True}))
    monkeypatch.setattr(atlas_graph, "_node_aggregate_context", _make_node("aggregate_context"))
    monkeypatch.setattr(atlas_graph, "_node_check_lock", _make_node("check_lock", {"queued": False}))
    monkeypatch.setattr(atlas_graph, "_node_reason", _make_node("reason"))
    monkeypatch.setattr(atlas_graph, "_node_plan_actions", _make_node("plan_actions"))
    monkeypatch.setattr(atlas_graph, "_node_execute_actions", _make_node("execute_actions"))
    monkeypatch.setattr(atlas_graph, "_node_synthesize", _make_node("synthesize"))
    monkeypatch.setattr(atlas_graph, "_node_notify", _make_node("notify"))

    state: dict[str, Any] = {"event_id": "evt-1"}
    result = await atlas_graph.run_reasoning_graph(state)

    assert ledger == [
        "triage",
        "aggregate_context",
        "check_lock",
        "reason",
        "plan_actions",
        "execute_actions",
        "synthesize",
        "notify",
    ]
    # Token totals zero-initialized by the orchestrator regardless of
    # the bundle's source.
    assert result["total_input_tokens"] == 0
    assert result["total_output_tokens"] == 0


@pytest.mark.asyncio
async def test_run_reasoning_graph_short_circuits_on_no_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Triage classifies the event as not needing reasoning -- the
    # orchestrator's early-exit branch should fire and skip the rest.
    from atlas_brain.reasoning import graph as atlas_graph

    ledger: list[str] = []

    def _make_node(name: str, patch: dict | None = None):
        async def _node(state: dict) -> dict:
            ledger.append(name)
            if patch:
                state.update(patch)
            return state

        return _node

    monkeypatch.setattr(atlas_graph, "_node_triage", _make_node("triage", {"needs_reasoning": False}))
    for name in (
        "aggregate_context", "check_lock", "reason", "plan_actions",
        "execute_actions", "synthesize", "notify",
    ):
        monkeypatch.setattr(atlas_graph, f"_node_{name}", _make_node(name))

    await atlas_graph.run_reasoning_graph({})

    assert ledger == ["triage"]


@pytest.mark.asyncio
async def test_run_reasoning_graph_short_circuits_on_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Lock check finds the entity locked. Orchestrator returns
    # without running reason/plan/execute/synthesize/notify.
    from atlas_brain.reasoning import graph as atlas_graph

    ledger: list[str] = []

    def _make_node(name: str, patch: dict | None = None):
        async def _node(state: dict) -> dict:
            ledger.append(name)
            if patch:
                state.update(patch)
            return state

        return _node

    monkeypatch.setattr(atlas_graph, "_node_triage", _make_node("triage", {"needs_reasoning": True}))
    monkeypatch.setattr(atlas_graph, "_node_aggregate_context", _make_node("aggregate_context"))
    monkeypatch.setattr(atlas_graph, "_node_check_lock", _make_node("check_lock", {"queued": True}))
    for name in (
        "reason", "plan_actions", "execute_actions", "synthesize", "notify",
    ):
        monkeypatch.setattr(atlas_graph, f"_node_{name}", _make_node(name))

    await atlas_graph.run_reasoning_graph({})

    assert ledger == ["triage", "aggregate_context", "check_lock"]
