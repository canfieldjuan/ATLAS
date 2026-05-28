# PR: Content Ops Run Usage Summary

## Why this slice exists

Content Ops now records LLM usage, exposes account-scoped usage summaries, and
shows a 7-day usage card. The remaining operator gap is per-run visibility:
after an execute call completes, the screen still does not connect that specific
generation to the usage rows it created.

The source fix is to stamp each hosted execute request with a stable request id,
carry that id into Content Ops LLM trace metadata, summarize usage for that
request, and render the result beside the execution output.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add hosted execute request-id tracing without changing customer inputs.
2. Return an optional run-scoped usage summary from `/content-ops/execute` when
   the usage database is configured.
3. Map and render that usage summary in the Content Ops execution panel.
4. Pin the backend trace/request contract and frontend mapper/UI contract.

### Files touched

- `plans/PR-Content-Ops-Run-Usage-Summary.md` - Plan doc for this slice.
- `extracted_content_pipeline/content_ops_execution.py` - Accept extra trace metadata and merge it into per-step LLM trace context.
- `extracted_content_pipeline/api/control_surfaces.py` - Stamp execute requests, summarize matching usage, and return it.
- `tests/test_extracted_content_control_surface_api.py` - Cover request-id trace context and execute usage-summary response wiring.
- `atlas-intel-ui/src/api/contentOps.ts` - Add execute response usage-summary wire fields.
- `atlas-intel-ui/src/domain/contentOps/types.ts` - Add execute response request id and usage summary domain fields.
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts` - Map execute usage summary into the domain model.
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` - Render run usage inside the execution panel.
- `atlas-intel-ui/scripts/content-ops-usage-summary.test.mjs` - Pin execution mapper and UI labels for run usage.
- `docs/extraction/validation/support_ticket_blog_observed_shell_live_telemetry_2026-05-28.md` - Mark the per-run usage UI follow-up addressed by this slice.

## Mechanism

The hosted `/execute` route generates a request id before calling the execution
engine. The execution engine accepts an optional `trace_metadata` mapping and
merges it into the same context that already carries account and cache policy
metadata. That keeps request tracking in infrastructure code instead of adding a
user-facing input field.

After execution, the route resolves the account-scoped usage pool if one is
configured and calls `summarize_content_ops_llm_usage(..., request_id=<id>)`.
The usage summary is returned as `usage_summary`; failures to read usage do not
fail generation, because generation may be valid even if telemetry is
temporarily unavailable.

The UI reuses the existing usage-summary wire/domain types and renders a compact
"This run" usage block in the execution panel.

## Intentional

- This does not expose prompts, responses, or support-ticket bodies. The result
  only returns aggregate tokens, calls, cost, savings, and cache counters.
- This does not infer per-run usage from the 7-day card. The request id is
  written into the trace context and the summary reads by that id.
- This does not require the usage database for execute to succeed. Missing usage
  infrastructure omits the run summary instead of blocking generation.

## Deferred

- Future PR: add a deeper per-run model/cache breakdown drawer if the compact
  metrics are not enough.
- Future PR: add a manual refresh for run usage if async usage persistence ever
  becomes delayed.
- Parked hardening: none. Root `HARDENING.md` has no active cost-surfacing
  parked items; `ATLAS-HARDENING.md` contains older blog/deep-dive content
  items outside this lane.

## Verification

- Focused backend route pytest for the execute usage summary and trace-context
  tests - 2 passed.
- Frontend usage-summary contract test script - 6 passed.
- Python compile over the touched backend route, executor, and backend test file
  - passed.
- Broader executor/control-surface pytest covering direct executor callers and
  hosted execute route behavior - 57 passed.
- Intel UI lint - passed.
- Whitespace check - passed.
- Intel UI production build - passed.
- Recurring/stale-value grep for the old per-run usage follow-up and stale LLM
  usage hardening title - no matches.
- Plan placeholder grep for the old "pending implementation" marker - no
  matches.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~88 |
| Backend request-id/usage response | ~70 |
| Backend tests | ~55 |
| Frontend types/mapper/UI | ~85 |
| Frontend test updates | ~35 |
| Validation follow-up doc | ~3 |
| **Total** | **~336** |

This stays below the 400 LOC soft cap.
