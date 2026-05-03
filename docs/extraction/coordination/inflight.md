# In-Flight PRs

Last updated: 2026-05-03T22:08Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-A1.5, queued) | Apply Copilot fixes that missed PR #87 merge | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed; opening immediately after PR-A2 |
| #94 | PR-C1a: Move archetypes + evidence_map to reasoning core | `extracted_reasoning_core/{archetypes.py, evidence_map.yaml}`; `tests/test_extracted_reasoning_core_archetypes.py`. Subsequent PR-C1 slices (temporal, types, evidence_engine slim, review_enrichment, api, wrappers, test renames, audit amendment) will land as separate atomic PRs after this one merges. | claude-2026-05-03 | `extracted_reasoning_core/**`; the three content_pipeline reasoning forks; `atlas_brain/reasoning/evidence_engine.py` (later PR-C1 slices) |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
