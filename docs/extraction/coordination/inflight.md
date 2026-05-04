# In-Flight PRs

Last updated: 2026-05-04T07:03Z by codex-2026-05-04-content

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C2, in flight) | PR-C2: Semantic cache split (PR 4 from reasoning boundary audit) | NEW: `extracted_reasoning_core/semantic_cache_keys.py` (pure: `compute_evidence_hash`, `apply_decay`, `CacheEntry`, `STALE_THRESHOLD`, `row_to_cache_entry`). NEW: `tests/test_extracted_reasoning_core_semantic_cache_keys.py`. EDIT: `extracted_llm_infrastructure/reasoning/semantic_cache.py` (import pure code from core; keep `SemanticCachePool` Protocol + `SemanticCache` storage class). EDIT: `extracted_competitive_intelligence/reasoning/semantic_cache.py` (replace atlas bridge with explicit re-exports from core + LLM-infra; closes PR 4 acceptance criterion #3). Atlas-side migration deferred to PR 7. | claude-2026-05-03 | `extracted_reasoning_core/semantic_cache_keys.py`; `extracted_llm_infrastructure/reasoning/semantic_cache.py`; `extracted_competitive_intelligence/reasoning/semantic_cache.py`; `tests/test_extracted_reasoning_core_semantic_cache_keys.py` |
| (PR-D10, in flight) | Add AI Content Ops queued send worker CLI | NEW: `extracted_content_pipeline/campaign_postgres_send.py`; NEW: `scripts/send_extracted_campaigns.py`; NEW: `tests/test_extracted_campaign_postgres_send.py`. EDIT: `extracted_content_pipeline/{README.md,STATUS.md,manifest.json}`; `extracted_content_pipeline/docs/{host_install_runbook.md,standalone_productization.md}`; `scripts/run_extracted_pipeline_checks.sh`; `tests/test_extracted_campaign_manifest.py`. | codex-2026-05-04-content | `extracted_content_pipeline/campaign_postgres_send.py`; `scripts/send_extracted_campaigns.py`; `tests/test_extracted_campaign_postgres_send.py`; listed content-pipeline docs and manifest files |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
