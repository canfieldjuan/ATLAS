# In-Flight PRs

Last updated: 2026-05-05T02:32Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C4d2, in flight) | PR-C4d2: reflection.py tracing-gap fix | EDIT: `atlas_brain/reasoning/reflection.py` (`run_reflection` accepts `ports: ReasoningPorts \| None = None`, lazy-builds defaults via the existing `agent._build_default_ports` factory, opens a `reasoning.reflection` span via `ports.trace_sink.start_span` and closes it on success/error with counts metadata). NEW: `tests/test_atlas_reasoning_reflection_tracing.py` (sys.modules-stubbed test of span open/close + status mapping + counts metadata, mirrors PR-C4d's save/restore fixture pattern). EDIT: `scripts/run_extracted_pipeline_checks.sh` + `.github/workflows/extracted_pipeline_checks.yml` (wire test + reflection.py path). Sister slice to PR-C4d -- closes the observability gap on the cron-driven reflection cycle that PR-C4d's audit conversation surfaced. Pure additive instrumentation; existing reflection behavior unchanged. | claude-2026-05-03 | `atlas_brain/reasoning/reflection.py`; `tests/test_atlas_reasoning_reflection_tracing.py`; `scripts/run_extracted_pipeline_checks.sh`; `.github/workflows/extracted_pipeline_checks.yml` |
| #164 | docs: log cross-product standalone % audit | `docs/extraction/cross_product_audit_2026-05-04.md` | canfieldjuan | Avoid editing the cross-product audit doc until PR #164 lands |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
