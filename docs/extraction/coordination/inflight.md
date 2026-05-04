# In-Flight PRs

Last updated: 2026-05-04T20:09Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C4a, in flight) | PR-C4a: Add EventSink + TraceSink ports (PR 6 from reasoning boundary audit) | EDIT: `extracted_reasoning_core/ports.py` (add `EventSink` Protocol for host event-bus emission and `TraceSink` Protocol for host tracing/span emission; existing `LLMClient`/`SemanticCacheStore`/`ReasoningStateStore`/`Clock` ports unchanged). NEW: `tests/test_extracted_reasoning_core_event_trace_ports.py` (8 unit tests: Protocol satisfaction, async emit round-trip, span open/close round-trip, optional metadata, error status). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire the new test). First slice of PR 6 (graph/state engine ports) -- adds the two contract ports the audit names; subsequent PR-C4b/c/d/e slices wire atlas adapters and port the graph/agent/state/reflection/context_aggregator engines. | claude-2026-05-03 | `extracted_reasoning_core/ports.py`; `tests/test_extracted_reasoning_core_event_trace_ports.py`; `scripts/run_extracted_pipeline_checks.sh` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
