# PR: Content Ops Exact Cache Adapter

## Why this slice exists

PR-Content-Ops-Exact-Cache-Policy added the source-side safety gate for
Content Ops LLM caching but deliberately did not perform cache lookup or store.
The next slice should use that policy rather than creating another cache rule
layer. It also needs to reuse the existing exact-cache infrastructure without
coupling Content Ops to the B2B or LLM Gateway feature flags.

This slice is intentionally narrow: when the Content Ops policy returns
`exact`, the LLM client can look up and store exact-cache responses for
account-scoped, non-customer-data calls. When the policy returns `no_store`, or
when lookup/store fails, generation continues through the provider.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a Content Ops exact-cache adapter path in `PipelineLLMClient.complete`.
2. Reuse the shared exact-cache request-envelope, lookup, and store helpers
   instead of creating a separate cache table or hashing implementation.
3. Let product-owned callers bypass the shared helper's legacy namespace flag
   when their own policy has already returned `exact`; keep the existing default
   behavior for B2B and LLM Gateway callers.
4. Trace cache hit/miss/store/error outcomes without capturing raw prompt or
   response payloads in LLM traces.
5. Keep cache failures non-fatal. A cache outage should become a provider call,
   not a generation failure.
6. Add focused tests for hit, miss/store, no-store, and cache failure behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Exact-Cache-Adapter.md` | Plan doc for the adapter slice. |
| `extracted_content_pipeline/campaign_llm_client.py` | Add exact-cache adapter hooks to the Content Ops LLM client. |
| `atlas_brain/services/b2b/llm_exact_cache.py` | Add an explicit namespace-flag bypass option for policy-gated callers. |
| `extracted_llm_infrastructure/services/b2b/llm_exact_cache.py` | Synced extracted copy of the shared exact-cache helper. |
| `tests/test_extracted_campaign_llm_client.py` | Cover Content Ops cache hit/miss/store/failure behavior. |
| `tests/test_b2b_intelligence_validation.py` | Cover the shared exact-cache helper bypass option. |

## Mechanism

`PipelineLLMClient` gets an injected exact-cache adapter with lazy defaults that
import the shared exact-cache helper only when needed. For each call:

1. Resolve the LLM and evaluate `ContentOpsExactCachePolicy`.
2. If the decision is not cacheable, call the provider exactly as today.
3. If the decision is cacheable, build the same deterministic request envelope
   the shared exact-cache helper uses, scoped by provider, model, messages,
   max tokens, and temperature.
4. Attempt lookup with the policy-provided namespace and account id.
5. On hit, return an `LLMResponse` from the cached text and trace
   `cache_result=hit`, zero provider usage for the current request, and
   cache-specific token metadata for savings/diagnostics.
6. On miss, call the provider, then store the successful response under the same
   namespace/account id and trace `cache_result=miss` plus the store result.
7. On cache lookup/store error, trace the error metadata and continue generation.

The shared exact-cache helper gets a keyword-only
`require_namespace_enabled: bool = True`. Existing callers keep the default
namespace flag behavior. Content Ops passes `False` only after its own policy
returns `exact`, so B2B/Gateway behavior is unchanged while Content Ops remains
controlled by the source-side policy from the previous slice.

## Intentional

- This does not cache support-ticket/customer-data prompts. The prior slice
  makes those policy decisions `no_store`, and this adapter only runs on
  cacheable decisions.
- This does not add UI controls. Backend behavior should prove safe before the
  UI exposes knobs.
- This does not add a new cache table or new hash format. Reusing the shared
  exact-cache helpers avoids a second cache implementation.
- This does not fail generation when the cache is unavailable. Cache is an
  optimization, not a dependency for generation.

## Deferred

- Future PR: UI/control-surface cache controls once the adapter is live and
  review-approved.
- Future PR: cache savings/cost rollup for Content Ops if the exact cache gets
  meaningful traffic.
- Future PR: affirmative `safe_to_cache` allowlist posture if Content Ops gains
  non-executor LLM paths that do not inherit the support-ticket marker context.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_campaign_llm_client.py tests/test_b2b_intelligence_validation.py::test_lookup_cached_text_can_bypass_namespace_flag_for_policy_gated_callers tests/test_b2b_intelligence_validation.py::test_store_cached_text_can_bypass_namespace_flag_for_policy_gated_callers -q — 24 passed.
- python -m compileall -q extracted_content_pipeline/campaign_llm_client.py atlas_brain/services/b2b/llm_exact_cache.py extracted_llm_infrastructure/services/b2b/llm_exact_cache.py tests/test_extracted_campaign_llm_client.py tests/test_b2b_intelligence_validation.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_llm_infrastructure_checks.sh — 35 passed, 1 warning.
- bash scripts/run_extracted_pipeline_checks.sh — 2496 passed, 7 skipped, 1 warning.
- python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q — 9 passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~100 |
| LLM client adapter | ~190 |
| Shared exact-cache helper | ~50 |
| Tests | ~300 |
| **Total** | **~640** |

This is above the 400 LOC soft cap because the adapter, shared-helper seam, and
tests need to land together; otherwise the PR would either add an unused helper
or wire cache behavior without proving the safety contract.
