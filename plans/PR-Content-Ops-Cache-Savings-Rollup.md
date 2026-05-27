# PR: Content Ops Cache Savings Rollup

## Why this slice exists

PR-Content-Ops-Exact-Cache-Adapter made Content Ops exact-cache hits return
zero provider usage for the current request and preserve cached token counts as
diagnostic metadata. The usage summary can now show cache hits and avoided
tokens, but it still has no dollar savings rollup, so operators can see that
cache is working without seeing what it saved.

This slice keeps the fix source-side: stamp a numeric cache-savings value when
the cache hit happens, then aggregate that field through the existing Content
Ops usage summary route. That avoids re-deriving costs in the read path and
matches the existing LLM Gateway cache-savings pattern.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add a cache-hit savings estimate to the Content Ops LLM adapter using the
   shared pricing configuration.
2. Preserve cache-hit provider usage as zero for the current request.
3. Add guarded `cache_savings_usd` aggregation to the Content Ops usage summary
   query and response payload.
4. Add focused tests for the cache-hit trace metadata and usage summary rollup.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Savings-Rollup.md` | Plan doc for the cache-savings rollup slice. |
| `extracted_content_pipeline/campaign_llm_client.py` | Estimate and stamp cache-hit savings metadata. |
| `extracted_content_pipeline/content_ops_usage_summary.py` | Aggregate numeric cache-savings metadata in the usage summary. |
| `tests/test_extracted_campaign_llm_client.py` | Pin cache-hit savings metadata on traced calls. |
| `tests/test_extracted_content_ops_usage_summary.py` | Pin summary and breakdown cache-savings rollups. |

## Mechanism

On an exact-cache hit, `_response_from_cache_hit(...)` already has the cached
entry's original input/output token counts. This slice runs those counts through
the shared `settings.ftl_tracing.pricing.cost_usd(...)` helper, using the cache
entry provider/model. The result is written to trace metadata as a numeric
`cache_savings_usd` field alongside `cached_input_tokens` and
`cached_output_tokens`.

The usage summary query then sums only numeric JSON values:

```sql
CASE
  WHEN jsonb_typeof(metadata->'cache_savings_usd') = 'number'
  THEN (metadata->>'cache_savings_usd')::float
  ELSE 0
END
```

That mirrors the LLM Gateway guard and prevents malformed string metadata from
breaking the whole usage route.

## Intentional

- This does not add UI display yet. The API rollup should be proven before the
  usage card adds another number.
- This does not write to `llm_cache_savings`; Content Ops already writes
  `llm_usage` rows through the hosted tracer, and the existing usage route reads
  those rows.
- This keeps cache-hit `input_tokens` and `output_tokens` at zero so naive cost
  rollups do not bill cache hits as fresh provider calls.
- If pricing lookup fails, the cache hit remains successful and savings defaults
  to `0.0`; telemetry cannot block generation.

## Deferred

- Future PR: show cache savings in the Content Ops usage card once the backend
  response contract is reviewed.
- Future PR: decide whether Content Ops should also mirror cache-hit savings
  into `llm_cache_savings` for cross-product cost dashboards.
- Parked hardening: none. Root `HARDENING.md` was scanned; the current parked
  item belongs to `content-ops/faq-generator`, not this cost-surfacing lane.

## Verification

- python -m pytest tests/test_extracted_campaign_llm_client.py tests/test_extracted_content_ops_usage_summary.py -q — 24 passed, 2 skipped, 1 warning.
- python -m compileall -q extracted_content_pipeline/campaign_llm_client.py extracted_content_pipeline/content_ops_usage_summary.py tests/test_extracted_campaign_llm_client.py tests/test_extracted_content_ops_usage_summary.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| LLM client savings metadata | ~40 |
| Usage summary aggregation | ~25 |
| Tests | ~25 |
| **Total** | **~185** |

This stays below the 400 LOC soft cap.
