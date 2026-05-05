"""Unit tests for ``extracted_reasoning_core.graph.run_graph``.

PR-C4e3 promoted the reasoning-graph orchestrator into core. The
orchestrator chains eight nodes in a fixed order with two early-exit
branches inherited from atlas's pre-extraction
``run_reasoning_graph``. These tests pin:

- node call order
- the ``needs_reasoning=False`` short-circuit after triage
- the ``queued=True`` short-circuit after lock check
- token-total zero-initialization

Each test uses a recording fake for every node so we can assert
exactly which nodes ran (and which were skipped). All tests are pure
-- no atlas imports, no I/O.
"""

from __future__ import annotations

from typing import Any

import pytest

from extracted_reasoning_core.graph import GraphNodes, run_graph


# ----------------------------------------------------------------------
# Test doubles
# ----------------------------------------------------------------------


class _RecordingNode:
    """Async fake that records each call and applies a state patch."""

    def __init__(
        self,
        name: str,
        ledger: list[str],
        patch: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._ledger = ledger
        self._patch = patch or {}

    async def __call__(self, state: dict) -> dict:
        self._ledger.append(self._name)
        state.update(self._patch)
        return state


def _build_nodes(
    ledger: list[str],
    *,
    triage_patch: dict | None = None,
    check_lock_patch: dict | None = None,
) -> GraphNodes:
    """Build a GraphNodes bundle with recording fakes.

    ``triage_patch`` and ``check_lock_patch`` let individual tests
    drive the early-exit branches by writing the right state field
    inside the corresponding node.
    """
    return GraphNodes(
        triage=_RecordingNode("triage", ledger, triage_patch),
        aggregate_context=_RecordingNode("aggregate_context", ledger),
        check_lock=_RecordingNode("check_lock", ledger, check_lock_patch),
        reason=_RecordingNode("reason", ledger),
        plan_actions=_RecordingNode("plan_actions", ledger),
        execute_actions=_RecordingNode("execute_actions", ledger),
        synthesize=_RecordingNode("synthesize", ledger),
        notify=_RecordingNode("notify", ledger),
    )


# ----------------------------------------------------------------------
# Happy-path order
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_graph_chains_all_eight_nodes_in_order() -> None:
    ledger: list[str] = []
    nodes = _build_nodes(
        ledger,
        triage_patch={"needs_reasoning": True},
        check_lock_patch={"queued": False},
    )
    state: dict = {}

    await run_graph(state, nodes)

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


@pytest.mark.asyncio
async def test_run_graph_initializes_token_totals_to_zero() -> None:
    ledger: list[str] = []
    nodes = _build_nodes(
        ledger,
        triage_patch={"needs_reasoning": True},
        check_lock_patch={"queued": False},
    )
    # Pre-existing non-zero values must be reset -- the orchestrator
    # owns the contract that nodes can do
    # ``state["total_input_tokens"] += usage["input_tokens"]`` without
    # checking for the key.
    state: dict = {"total_input_tokens": 999, "total_output_tokens": 999}

    await run_graph(state, nodes)

    assert state["total_input_tokens"] == 0
    assert state["total_output_tokens"] == 0


# ----------------------------------------------------------------------
# Early-exit branches
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_graph_short_circuits_when_needs_reasoning_is_false() -> None:
    # Triage classifies the event as not needing deeper reasoning --
    # the rest of the pipeline (context aggregation, lock check, the
    # heavy LLM nodes, action execution, notification) is wasted work.
    ledger: list[str] = []
    nodes = _build_nodes(
        ledger,
        triage_patch={"needs_reasoning": False},
    )
    state: dict = {}

    await run_graph(state, nodes)

    # Only triage runs.
    assert ledger == ["triage"]
    assert state["needs_reasoning"] is False


@pytest.mark.asyncio
async def test_run_graph_short_circuits_when_queued_is_true() -> None:
    # Lock check found the entity locked by another agent. The event
    # has been queued for drain; running the rest of the pipeline now
    # would race with the lock holder.
    ledger: list[str] = []
    nodes = _build_nodes(
        ledger,
        triage_patch={"needs_reasoning": True},
        check_lock_patch={"queued": True},
    )
    state: dict = {}

    await run_graph(state, nodes)

    # Triage + aggregate_context + check_lock run; reason/plan/execute
    # /synthesize/notify do not.
    assert ledger == ["triage", "aggregate_context", "check_lock"]
    assert state["queued"] is True


@pytest.mark.asyncio
async def test_run_graph_does_not_short_circuit_when_queued_is_false() -> None:
    # Sanity check the inverse: ``queued: False`` flows through normally.
    # Catches a regression where the early-exit condition gets flipped
    # to ``if "queued" in state`` (truthiness vs key presence).
    ledger: list[str] = []
    nodes = _build_nodes(
        ledger,
        triage_patch={"needs_reasoning": True},
        check_lock_patch={"queued": False},
    )

    await run_graph({}, nodes)

    # All eight nodes run.
    assert len(ledger) == 8
    assert "notify" in ledger


# ----------------------------------------------------------------------
# State pass-through
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_graph_threads_same_state_through_all_nodes() -> None:
    # Each node mutates state and returns it. The orchestrator threads
    # the same dict through, so a mutation in node N is visible in
    # node N+1.
    seen: list[dict] = []

    class _Capture:
        def __init__(self, name: str, patch: dict | None = None) -> None:
            self._name = name
            self._patch = patch or {}

        async def __call__(self, state: dict) -> dict:
            seen.append(dict(state))  # snapshot
            state.update(self._patch)
            state[f"saw_{self._name}"] = True
            return state

    nodes = GraphNodes(
        triage=_Capture("triage", {"needs_reasoning": True}),
        aggregate_context=_Capture("aggregate_context"),
        check_lock=_Capture("check_lock", {"queued": False}),
        reason=_Capture("reason"),
        plan_actions=_Capture("plan_actions"),
        execute_actions=_Capture("execute_actions"),
        synthesize=_Capture("synthesize"),
        notify=_Capture("notify"),
    )

    final = await run_graph({"event_id": "evt-1"}, nodes)

    # Each node's final-state snapshot includes the marker the
    # previous node left.
    assert seen[1]["saw_triage"] is True  # aggregate_context sees triage's marker
    assert seen[7]["saw_synthesize"] is True  # notify sees synthesize's marker
    # Final state has all markers + the original event_id.
    assert final["event_id"] == "evt-1"
    assert all(final.get(f"saw_{name}") for name in (
        "triage", "aggregate_context", "check_lock", "reason",
        "plan_actions", "execute_actions", "synthesize", "notify",
    ))


# ----------------------------------------------------------------------
# GraphNodes immutability
# ----------------------------------------------------------------------


def test_graph_nodes_is_frozen() -> None:
    # The orchestrator runs read-only against the bundle; mutating
    # mid-run would risk inconsistent behavior. Pin the frozen=True
    # contract so a future refactor can't accidentally drop it.
    # ``frozen=True`` raises ``FrozenInstanceError``; ``slots=True``
    # alone would raise ``AttributeError``. Accept either so the test
    # doesn't have to know which decorator combination ships.
    from dataclasses import FrozenInstanceError

    nodes = _build_nodes([])
    with pytest.raises((FrozenInstanceError, AttributeError)):
        nodes.triage = lambda state: state  # type: ignore[misc]
