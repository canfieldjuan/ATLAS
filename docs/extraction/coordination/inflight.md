# In-Flight PRs

Last updated: 2026-05-04T08:30Z by claude-2026-05-03-b

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C2, in flight) | PR-C2: Semantic cache split (PR 4 from reasoning boundary audit) | NEW: `extracted_reasoning_core/semantic_cache_keys.py` (pure: `compute_evidence_hash`, `apply_decay`, `CacheEntry`, `STALE_THRESHOLD`, `row_to_cache_entry`). NEW: `tests/test_extracted_reasoning_core_semantic_cache_keys.py`. EDIT: `extracted_llm_infrastructure/reasoning/semantic_cache.py` (import pure code from core; keep `SemanticCachePool` Protocol + `SemanticCache` storage class). EDIT: `extracted_competitive_intelligence/reasoning/semantic_cache.py` (replace atlas bridge with explicit re-exports from core + LLM-infra; closes PR 4 acceptance criterion #3). Atlas-side migration deferred to PR 7. | claude-2026-05-03 | `extracted_reasoning_core/semantic_cache_keys.py`; `extracted_llm_infrastructure/reasoning/semantic_cache.py`; `extracted_competitive_intelligence/reasoning/semantic_cache.py`; `tests/test_extracted_reasoning_core_semantic_cache_keys.py` |
| (PR-A5d-cleanup, in flight) | PR-A5d-cleanup: Address Copilot review on PR-A5d (#148) | EDIT: `atlas_brain/services/llm/anthropic.py` (annotate `_async_client: Any \| None` to match documented pre-`load()` None state). EDIT: `atlas_brain/services/b2b/anthropic_batch.py` (drop now-unused `AnthropicLLM` import after isinstance sites moved to Protocol). EDIT: `tests/test_anthropic_batchable_protocol.py` (drop unused `Any` and `pytest` imports). SYNC: 2 extracted_llm_infrastructure files. No behavior change. | claude-2026-05-03-b | `atlas_brain/services/llm/anthropic.py`; `atlas_brain/services/b2b/anthropic_batch.py`; `tests/test_anthropic_batchable_protocol.py`; the 2 synced extracted_llm_infrastructure copies |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
