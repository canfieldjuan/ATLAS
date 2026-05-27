# PR: Content Ops Exact Cache Policy

## Why this slice exists

Content Ops now surfaces usage and account-period budget controls, but exact
cache wiring is still only a deferred idea. The repo already has exact-cache
infrastructure, but Content Ops generation often uses customer source material
such as support tickets. Wiring cache lookup/store directly into generation
would risk storing raw customer payloads before we have an explicit policy.

This slice adds the source-side policy seam first: decide whether a Content Ops
LLM call is cache-eligible, require account scope for exact cache, default to
no-store, and make support-ticket/customer-data payloads no-store unless a later
slice introduces a safer redacted/digest-only cache shape.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a Content Ops exact-cache policy helper that returns an explicit
   cache/no-store decision with reason, namespace, and account scope.
2. Add settings/config fields for the policy, defaulting exact cache off and
   customer-data exact cache off.
3. Wire `PipelineLLMClient` to evaluate the policy and include the decision in
   trace metadata.
4. Keep cache behavior read-only in this slice: no cache lookup, no cache store,
   and no changes to generated output.
5. Add focused tests for policy branches and LLM trace integration.
6. Enroll the new policy tests in the extracted pipeline CI runner.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Exact-Cache-Policy.md` | Plan doc for the cache-policy seam. |
| `extracted_content_pipeline/content_ops_cache_policy.py` | Add Content Ops exact-cache policy decisions. |
| `extracted_content_pipeline/campaign_llm_client.py` | Evaluate policy and add cache decision metadata to LLM traces. |
| `extracted_content_pipeline/settings.py` | Add extracted Content Ops cache settings defaults. |
| `extracted_content_pipeline/manifest.json` | Register the new owned policy module. |
| `tests/test_atlas_content_ops_infrastructure.py` | Update host infrastructure trace assertion for cache decision metadata. |
| `tests/test_extracted_content_ops_cache_policy.py` | Cover policy branch decisions. |
| `tests/test_extracted_campaign_llm_client.py` | Cover trace metadata carrying the cache decision. |
| `scripts/run_extracted_pipeline_checks.sh` | Enroll the new extracted policy test. |

## Mechanism

`content_ops_cache_policy.py` introduces a small immutable
`ContentOpsExactCachePolicy` plus `ContentOpsCacheDecision`. The policy accepts
the LLM call metadata already threaded through `PipelineLLMClient.complete` and
returns a decision:

- `no_store` by default when exact cache is disabled.
- `no_store` when account scope is missing, so Content Ops does not fall back
  to the exact-cache sentinel account.
- `no_store` for support-ticket/customer-data source markers, even when exact
  cache is enabled.
- `exact` only when the call explicitly requests exact cache, exact cache is
  enabled, the asset type is supported, account scope is present, and no
  customer-data marker blocks caching.

`PipelineLLMClient` builds the policy from config/settings and evaluates it for
each call. The result is added to trace metadata as decision fields such as
`cache_mode`, `cache_reason`, and `cache_namespace`. This gives operators and
follow-up slices a concrete signal without changing generation behavior.

## Intentional

- This does not call exact-cache lookup or store. Policy and observability land
  before behavior.
- This does not cache support-ticket prompts. Current prompts can include raw
  customer data, so support-ticket/customer-data source markers remain no-store.
- This does not add UI controls. The backend policy contract should exist
  before product controls expose it.
- This does not reuse the B2B cache-strategy registry. Content Ops needs a
  separate policy because customer upload/source-material privacy is a first
  order concern.

## Deferred

- Future PR: exact-cache adapter that can lookup/store only when this policy
  returns `exact`.
- Future PR: redacted/digest-only cache envelopes for support-ticket inputs if
  we decide customer-source generation should be cacheable.
- Future PR: UI controls once the backend cache behavior is live and safe.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_atlas_content_ops_infrastructure.py::test_build_content_ops_llm_client_uses_pipeline_tracing_client tests/test_extracted_content_ops_cache_policy.py tests/test_extracted_campaign_llm_client.py -q — 29 passed, 1 warning.
- python -m compileall -q extracted_content_pipeline/content_ops_cache_policy.py tests/test_extracted_content_ops_cache_policy.py — passed.
- python -m pytest tests/test_extracted_content_ops_cache_policy.py tests/test_extracted_campaign_llm_client.py -q — 27 passed, 1 warning.
- python -m compileall -q extracted_content_pipeline/content_ops_cache_policy.py extracted_content_pipeline/campaign_llm_client.py extracted_content_pipeline/settings.py tests/test_extracted_content_ops_cache_policy.py tests/test_extracted_campaign_llm_client.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python -m pytest tests/test_audit_extracted_pipeline_ci_enrollment.py -q — 9 passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~110 |
| Policy module | ~170 |
| LLM client/settings/manifest wiring | ~90 |
| Tests and CI enrollment | ~230 |
| **Total** | **~595** |

This is above the 400 LOC soft cap because the policy module, integration
wiring, and tests need to land together for the source-side cache contract to be
meaningful and enforceable.
