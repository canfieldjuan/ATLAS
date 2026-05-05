"""Host-agnostic reasoning-graph orchestrator.

PR-C4e3 (this module) is the third sub-slice of the audit's PR 6
graph-engine extraction. The orchestrator -- the function that chains
the eight reasoning nodes in order with early-exit branches -- moves
into core, parameterized on a :class:`GraphNodes` bundle that the host
populates with its own node callables.

Two of the eight nodes (``triage`` and ``synthesize``) live fully in
core via :mod:`extracted_reasoning_core.graph_nodes` since PR-C4e2.
The other six remain host-side -- four are explicitly atlas-coupled
(``aggregate_context`` / ``check_lock`` / ``execute_actions`` /
``notify`` read the host's DB, atlas-specific APIs, and ntfy HTTP),
``reason`` keeps an atlas-specific extended-state prompt builder,
and ``plan_actions`` is the pure filter from :mod:`graph_helpers`
that atlas re-exports under its private name. The bundle abstracts
away those origins -- core just chains callables.

Atlas's ``run_reasoning_graph`` becomes a thin wrapper that builds a
``GraphNodes`` from its ``_node_*`` callables and delegates to
:func:`run_graph`. The wrapper is the only atlas seam this module
needs to know about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from .state import ReasoningAgentState


# Each node transforms state in place + returns it. The Awaitable
# return enables nodes to await DB queries / LLM calls without
# forcing the orchestrator to thread an event loop concept through
# its signature.
_NodeFn = Callable[[ReasoningAgentState], Awaitable[ReasoningAgentState]]


@dataclass(frozen=True, slots=True)
class GraphNodes:
    """Callable bundle the orchestrator chains together.

    Each field is an async function ``(state) -> state``. Hosts
    populate the bundle with their concrete implementations:

    - ``triage`` / ``synthesize`` typically come from
      :mod:`extracted_reasoning_core.graph_nodes` (host wraps the
      core node with a closure that resolves its workload-specific
      LLM client).
    - ``plan_actions`` typically comes from
      :func:`extracted_reasoning_core.graph_helpers.plan_actions`.
    - ``aggregate_context`` / ``check_lock`` / ``reason`` /
      ``execute_actions`` / ``notify`` are host-specific: each host
      contributes its own implementation, and core never reaches
      into the host's storage / API layer directly.

    Frozen + slots so a built bundle can't be mutated mid-run --
    the orchestrator is read-only against this dataclass.
    """

    triage: _NodeFn
    aggregate_context: _NodeFn
    check_lock: _NodeFn
    reason: _NodeFn
    plan_actions: _NodeFn
    execute_actions: _NodeFn
    synthesize: _NodeFn
    notify: _NodeFn


async def run_graph(
    state: ReasoningAgentState,
    nodes: GraphNodes,
) -> ReasoningAgentState:
    """Execute the full reasoning graph: triage -> context -> lock check ->
    reason -> plan -> execute -> synthesize -> notify.

    Two early-exit branches preserve atlas's pre-extraction semantics:

    1. After triage, if ``state["needs_reasoning"]`` is falsy, return
       immediately. Triage may classify an event as not worth deeper
       reasoning (e.g. a routine system tick) and the rest of the
       pipeline is wasted work in that case.
    2. After lock check, if ``state["queued"]`` is True, return
       immediately. The entity was locked by another agent and the
       event has been queued for drain after the lock releases.

    Token totals are zero-initialized at the start so each node can
    accumulate via ``state.get("total_*_tokens", 0) + ...`` without
    worrying about whether the field is present.
    """
    state["total_input_tokens"] = 0
    state["total_output_tokens"] = 0

    state = await nodes.triage(state)
    if not state.get("needs_reasoning"):
        return state

    state = await nodes.aggregate_context(state)
    state = await nodes.check_lock(state)
    if state.get("queued"):
        return state

    state = await nodes.reason(state)
    state = await nodes.plan_actions(state)
    state = await nodes.execute_actions(state)
    state = await nodes.synthesize(state)
    state = await nodes.notify(state)
    return state


__all__ = ["GraphNodes", "run_graph"]
