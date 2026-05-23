# PR: Support Ticket Input Provider Route Parity

## Why this slice exists

The support-ticket input provider is now the handoff point between host-owned
ticket ingestion and Content Ops generation. The provider has direct adapter
tests and one preview-route canary, but the route surface that matters to users
is broader: `/content-ops/preview`, `/content-ops/plan`, and
`/content-ops/execute` should all apply the same provider-built support-ticket
package before they validate, plan, or run generation.

This slice adds focused functional validation for that route parity. It does not
add generator behavior or host upload wiring.

## Scope (this PR)

Ownership lane: content-ops/input-provider-ticket-package

Slice phase: Functional validation

1. Keep the existing preview-route support-ticket provider test.
2. Fix the support-ticket package so FAQ source-type filters match the
   normalized rows they will be applied to.
3. Avoid applying a strict FAQ date-window filter when the uploaded ticket rows
   do not contain row dates; the source-period context still says the upload is
   the last 90 days.
4. Add a plan-route test proving provider-generated FAQ Report defaults produce
   a runnable `faq_markdown` generation plan.
5. Add an execute-route test proving the deterministic `faq_markdown` service
   can run from provider-expanded support-ticket material.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_support_ticket_input_provider.py`
- `plans/PR-Support-Ticket-Input-Provider-Route-Parity.md`

## Mechanism

The tests use the existing `create_content_ops_control_surface_router(...)`
input-provider seam and `SupportTicketInputProvider(source_material_loader=...)`.
The loader asserts tenant scope and request payload shape where useful, then
returns in-memory ticket rows.

The package now derives `faq_source_types` from normalized source rows instead
of hardcoding `support_ticket`, because hosts can pass source rows labeled
`ticket`, `case`, or another supported ticket-source alias. It only includes
`faq_window_days` when every normalized row has a parseable `created_at` value,
because otherwise the FAQ Markdown service treats the value as a strict
evidence date filter and drops undated or malformed-date CSV rows even when the
upload itself represents the last 90 days.

The plan-route test calls the mounted `/ops/plan` endpoint directly and checks
that the provider package resolves to the `TicketFAQMarkdownService.generate`
runner. The execute-route test injects a `ContentOpsExecutionServices` bundle
with the deterministic FAQ Markdown service and verifies the generated markdown
contains the ticket question.

## Intentional

- No Atlas host changes. Host wiring is covered separately in
  `PR-Content-Ops-Host-Ticket-Input-Provider`.
- No LLM-backed output execution. `faq_markdown` is deterministic and gives the
  provider route path an end-to-end execution canary without model cost.
- No standalone FAQ article contract. That remains owned by the FAQ session.
- The date-window change is limited to the support-ticket input package. Direct
  `faq_window_days` callers still get the existing strict date filter.
- Mixed dated/undated support-ticket uploads omit the strict window filter and
  favor keeping customer evidence over applying a partial window. The
  `source_period` text still carries the upload's intended last-90-days context.

## Deferred

- Future PR owned by the host wiring lane: add Atlas API route tests once the
  host provider mount lands.
- Future PR owned by the ingestion lane: test persisted upload/import lookup
  when a loader contract exists.
- Parked hardening: none.

## Verification

- `py_compile` for changed Python files - passed.
- `pytest tests/test_extracted_support_ticket_input_provider.py tests/test_extracted_support_ticket_input_package.py -q` - 26 passed.
- `scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `scripts/check_ascii_python.sh` - passed.
- `scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Package source-type/date-window fix | ~45 |
| Package regression tests | ~55 |
| Route parity tests | ~55 |
| **Total** | **~240** |
