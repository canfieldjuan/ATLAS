# PR: Content Ops LLM Usage Tracing

## Why this slice exists

The extracted LLM infrastructure already has the `llm_usage` tracing surface
needed for cost dashboards, cache savings, drift reports, and budget gates. The
Content Ops live generation path currently records provider usage only inside
generated draft metadata, so operators can inspect one saved asset but cannot
roll up spend by account, output type, source type, run, model, or repair loop.

This is the first cost/caching integration slice. Before wiring budget gates or
exact caching, Content Ops needs actual LLM usage rows for the calls it already
makes through `PipelineLLMClient`.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Trace `PipelineLLMClient.complete` calls through the existing LLM
   infrastructure `trace_llm_call` bridge after successful provider calls.
2. Trace failed provider calls with failure status, provider/model identity when
   available, duration, and error type.
3. Preserve caller metadata and add stable Content Ops attribution fields so
   later cost dashboards can group usage without parsing draft metadata.
4. Keep prompt/response bodies out of the trace in this first slice to avoid
   persisting uploaded support-ticket text into cost logs.
5. Add focused tests proving success and failure traces are emitted and that
   tracing failures do not break generation.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-LLM-Usage-Tracing.md` | Plan doc for the first Content Ops cost-surfacing slice. |
| `extracted_content_pipeline/campaign_llm_client.py` | Emit safe usage traces from the Content Ops LLM adapter. |
| `tests/test_extracted_campaign_llm_client.py` | Cover success, failure, metadata, and trace-failure behavior. |

## Mechanism

`PipelineLLMClient.complete` already owns the direct provider call for the
Content Ops generation services. This slice measures the call duration, converts
the provider response to `LLMResponse`, normalizes usage fields, and calls the
existing trace bridge with:

- span name: `content_ops.llm.complete`
- provider/model identity from the resolved LLM and response
- input/output tokens plus cache-read/cache-write token counters when present
- metadata merged from the generation service plus `product=content_ops`,
  `workload`, and `llm_adapter=pipeline`

No `input_data` or `output_data` is sent in this slice. That keeps customer CSV
content out of the cost table while still making spend queryable.

## Intentional

- This does not wire exact caching yet. Cache policy needs an explicit
  support-ticket privacy/no-store decision before generated response text is
  stored.
- This does not add a runtime budget gate yet. Budget decisions should be based
  on actual traced usage first.
- The trace hook is best-effort. A telemetry failure must not fail the user's
  generation request.
- No database migration is needed; the shared `llm_usage` table already exists.

## Deferred

- Future PR: expose Content Ops usage/cost rollups in the generated-asset or
  control-surface API.
- Future PR: add a `BudgetGate` integration once Content Ops usage rows are
  present and queryable.
- Future PR: add exact-cache wiring with a support-ticket privacy policy,
  account scoping, and no-store behavior.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_campaign_llm_client.py -q — 15 passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline — completed with synced files refreshed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| LLM client tracing | ~165 |
| Tests | ~120 |
| **Total** | **~375** |
