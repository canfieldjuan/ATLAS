# In-Flight PRs

Last updated: 2026-05-03T23:34Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #90 | Re-apply Copilot fixes that missed PR #87 merge (PR-A1.5) | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed |
| #95 | Add drift report (NEW CODE, PR-A4a) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/drift.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_drift.py` | claude-2026-05-03-b | `services/cost/drift.py` or its test |
| #96 | Add runtime budget gate (NEW CODE, PR-A4b) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/budget.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_budget.py` | claude-2026-05-03-b | `services/cost/budget.py` or its test |
| #98 | Add OpenAI billing fetcher (NEW CODE, PR-A4c) | `extracted_llm_infrastructure/{manifest.json, services/cost/__init__.py, services/cost/openai_billing.py, README.md, STATUS.md}`; `tests/test_extracted_llm_infrastructure_openai_billing.py` | claude-2026-05-03-b | `services/cost/openai_billing.py` or its test |
| #100 | PR-C1b: Move temporal.py to reasoning core | NEW: `extracted_reasoning_core/temporal.py` (atlas canonical + content_pipeline's `_numeric_value` / `_row_get` defensive helpers + parameterized `MIN_DAYS_FOR_PERCENTILES` constructor arg). NEW: `tests/test_extracted_reasoning_core_temporal.py` (18 smoke tests). | claude-2026-05-03 | `extracted_reasoning_core/temporal.py`; `tests/test_extracted_reasoning_core_temporal.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
