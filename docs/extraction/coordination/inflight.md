# In-Flight PRs

Last updated: 2026-05-19T02:00Z by codex-2026-05-18

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| TBD | PR-LLM-Registry-Service-Export | `extracted_llm_infrastructure/services/__init__.py`, `extracted_content_pipeline/campaign_llm_client.py`, `tests/test_extracted_llm_infrastructure_registry_export.py`, `tests/test_extracted_campaign_llm_client.py`, `plans/PR-LLM-Registry-Service-Export.md` | codex-2026-05-18 | extracted LLM namespace/package init work; content LLM adapter chat message normalization |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
