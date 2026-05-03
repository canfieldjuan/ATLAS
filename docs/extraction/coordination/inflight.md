# In-Flight PRs

Last updated: 2026-05-03T22:03Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-A1.5, queued) | Apply Copilot fixes that missed PR #87 merge | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed; opening immediately after PR-A2 |
| (PR-C1, in flight) | Reasoning evidence/temporal/archetypes consolidation | NEW: `extracted_reasoning_core/{archetypes,evidence_engine,evidence_map.yaml,temporal}.py`; `atlas_brain/reasoning/review_enrichment.py`. EDIT: `atlas_brain/reasoning/evidence_engine.py` (slim); `extracted_reasoning_core/{api,types}.py`; `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py` (-> wrappers); `tests/test_extracted_reasoning_core_api.py`. RENAME: `tests/test_extracted_reasoning_*.py` -> `tests/test_extracted_reasoning_core_*.py`. Audit-doc amendment to `docs/extraction/reasoning_boundary_audit_2026-05-03.md` in same commit. | claude-2026-05-03 | all listed files; especially `atlas_brain/reasoning/evidence_engine.py` and the three content_pipeline reasoning forks |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
