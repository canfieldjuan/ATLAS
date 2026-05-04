# In-Flight PRs

Last updated: 2026-05-04T02:50Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C1j, in flight) | PR-C1j: Route `extracted_content_pipeline/reasoning/temporal.py` through reasoning core wrapper | EDIT: `extracted_content_pipeline/reasoning/temporal.py` (drop ~466-line drifted fork; replace with thin re-export wrapper from `extracted_reasoning_core.temporal` + `extracted_reasoning_core.types`). All temporal types/constants/`TemporalEngine` were already promoted to core in PR-C1b/PR-C1c, so no drift-forward needed. Existing `tests/test_extracted_reasoning_temporal.py` keeps green against the wrapper. | claude-2026-05-03 | `extracted_content_pipeline/reasoning/temporal.py`; `tests/test_extracted_reasoning_temporal.py` |
| (PR-D9, in flight) | Add AI Content Ops draft review/status update path | NEW: `extracted_content_pipeline/campaign_postgres_review.py`; NEW: `scripts/review_extracted_campaign_drafts.py`; EDIT: content-pipeline docs/status/manifest/check script; NEW focused review tests | codex-2026-05-03 | Avoid `extracted_reasoning_core/**`, `extracted_content_pipeline/reasoning/**`, `extracted_content_pipeline/docs/reasoning_state_audit.md`, and `extracted_quality_gate/**` |
| (PR-B5c, in flight) | PR-B5c: Source-quality pack (deterministic core + Atlas re-export) | NEW: `extracted_quality_gate/source_quality_pack.py` (`apply_witness_render_gate`, `evaluate_source_quality`, `compute_coverage_ratio`, `row_count`, `build_non_empty_text_check`). EDIT: `extracted_quality_gate/{__init__.py, manifest.json, README.md, STATUS.md}`. EDIT: `atlas_brain/services/b2b/witness_render_gate.py` (slimmed to thin re-export). EDIT: `atlas_brain/services/b2b/source_impact.py` (private helpers route through pack). EDIT: `atlas_brain/autonomous/tasks/_b2b_specificity.py` (re-export `_contains_term`/`_normalize_text` for `services/blog_quality.py`, fixing PR-B5b regression). NEW: `tests/test_extracted_quality_gate_source_quality_pack.py` (34 tests). | claude-2026-05-03-b | `extracted_quality_gate/source_quality_pack.py`; `atlas_brain/services/b2b/witness_render_gate.py`; `atlas_brain/services/b2b/source_impact.py`; the new test file |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
