# In-Flight PRs

Last updated: 2026-05-03T23:39Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #90 | Re-apply Copilot fixes that missed PR #87 merge (PR-A1.5) | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed |
| #95 | Add drift report (NEW CODE, PR-A4a) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/drift.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_drift.py` | claude-2026-05-03-b | `services/cost/drift.py` or its test |
| #96 | Add runtime budget gate (NEW CODE, PR-A4b) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/budget.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_budget.py` | claude-2026-05-03-b | `services/cost/budget.py` or its test |
| #98 | Add OpenAI billing fetcher (NEW CODE, PR-A4c) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/openai_billing.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_openai_billing.py` | claude-2026-05-03-b | `services/cost/openai_billing.py` or its test |
| (PR-C1g, in flight) | PR-C1g: Wire `score_archetypes` + `build_temporal_evidence` stubs in api.py | EDIT: `extracted_reasoning_core/api.py` (impl 2 of 3 stubs; `evaluate_evidence` waits for PR-C1d's slim engine). EDIT: `tests/test_extracted_reasoning_core_api.py` (drop the 2 now-implemented stubs from the fail-closed list; add behavioral tests for the wired entry points). | claude-2026-05-03 | `extracted_reasoning_core/api.py`; `tests/test_extracted_reasoning_core_api.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
