# In-Flight PRs

Last updated: 2026-05-05T01:59Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C4d, in flight) | PR-C4d: agent.py consumes ReasoningPorts.trace_sink (PR 6 fourth slice) | EDIT: `atlas_brain/reasoning/port_adapters.py` (`AtlasTraceSink` extracts typed kwargs from metadata: `start_span` extracts `model_name`/`model_provider`/`session_id`; `end_span` extracts `input_tokens`/`output_tokens`/`input_data`/`output_data`/`error_message`/`error_type`. `reasoning` and `business` keys stay in metadata so atlas tracer's `_derive_reasoning_text` promotion + business-context capture still fire). EDIT: `atlas_brain/reasoning/agent.py` (`ReasoningAgentGraph.__init__` accepts `ports: ReasoningPorts | None = None` and builds defaults via `_build_default_ports()`; `process_event` replaces direct `tracer.start_span/end_span` calls with `self._ports.trace_sink.*` and packs the lifted kwargs into metadata). EDIT: `tests/test_atlas_reasoning_port_adapters.py` (extend to cover the new extraction behavior; `reasoning`/`business` passthrough). NEW: `tests/test_atlas_reasoning_agent_port_consumption.py` (process_event with fake TraceSink; verify call order, packed metadata, status mapping on success + failure). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (add new test + agent.py path). Reflection.py's missing tracing is a separate gap, deferred. | claude-2026-05-03 | `atlas_brain/reasoning/port_adapters.py`; `atlas_brain/reasoning/agent.py`; `tests/test_atlas_reasoning_port_adapters.py`; `tests/test_atlas_reasoning_agent_port_consumption.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
