# In-Flight PRs

Last updated: 2026-05-03T22:04Z by codex-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-A1.5, queued) | Apply Copilot fixes that missed PR #87 merge | `extracted_llm_infrastructure/{skills/__init__.py, _standalone/config.py, STATUS.md}`; `scripts/smoke_extracted_llm_infrastructure_imports.py`; `scripts/smoke_extracted_llm_infrastructure_standalone.py` | claude-2026-05-03-b | the 5 files listed; opening immediately after PR-A2 |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
