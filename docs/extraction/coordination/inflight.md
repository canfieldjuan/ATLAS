# In-Flight PRs

Last updated: 2026-05-04T21:53Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C4c, in flight) | PR-C4c: Atlas adapters for EventSink + TraceSink ports (PR 6 third slice) | NEW: `atlas_brain/reasoning/port_adapters.py` (`AtlasEventSink` wraps `emit_event` from `atlas_brain/reasoning/events.py` and stringifies the UUID; `AtlasTraceSink` wraps `tracer.start_span`/`end_span` from `atlas_brain/services/tracing.py` with operation_type defaulted to `"reasoning"` and Port-side `status: ok/error` mapped to atlas's `completed/failed` vocabulary). NEW: `tests/test_atlas_reasoning_port_adapters.py` (Protocol isinstance checks against the `@runtime_checkable` ports from PR-C4a; monkeypatched call-shape verification for both adapters; status-mapping coverage). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (wire new test from first commit, learning from PR-C4b CI miss). No runner wiring yet -- PR-C4d/e instantiate these and feed them through `ReasoningPorts`. Pure additive atlas-side surface; existing `emit_event`/`tracer` callers unchanged. | claude-2026-05-03 | `atlas_brain/reasoning/port_adapters.py`; `tests/test_atlas_reasoning_port_adapters.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
