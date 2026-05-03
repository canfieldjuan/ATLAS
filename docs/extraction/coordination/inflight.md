# In-Flight PRs

Last updated: 2026-05-03T23:35Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #90 | Re-apply Copilot fixes that missed PR #87 merge (PR-A1.5) | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed |
| #95 | Add drift report (NEW CODE, PR-A4a) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/drift.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_drift.py` | claude-2026-05-03-b | `services/cost/drift.py` or its test |
| #96 | Add runtime budget gate (NEW CODE, PR-A4b) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/budget.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_budget.py` | claude-2026-05-03-b | `services/cost/budget.py` or its test |
| #98 | Add OpenAI billing fetcher (NEW CODE, PR-A4c) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/openai_billing.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_openai_billing.py` | claude-2026-05-03-b | `services/cost/openai_billing.py` or its test |
| #102 | PR-C1c: Promote temporal types + add evidence-result types | EDIT: `extracted_reasoning_core/types.py` (rich `TemporalEvidence` + 4 sub-types + `ConclusionResult` + `SuppressionResult`). EDIT: `extracted_reasoning_core/temporal.py` (import 5 promoted dataclasses from `.types`). NEW: `tests/test_extracted_reasoning_core_types.py` (9 smoke tests). | claude-2026-05-03 | `extracted_reasoning_core/{types.py, temporal.py}`; `tests/test_extracted_reasoning_core_types.py` |
| (PR-D5, branch `codex/content-pipeline-migration-runner`) | Add AI Content Ops migration runner | `extracted_content_pipeline/storage/migration_runner.py`; `scripts/run_extracted_content_pipeline_migrations.py`; content-pipeline docs/status/manifest; focused migration tests | codex-2026-05-03 | Do not touch `extracted_reasoning_core/**`, `extracted_content_pipeline/reasoning/{archetypes,evidence_engine,temporal}.py`, or LLM-infra files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
