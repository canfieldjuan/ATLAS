# PR-Atlas-Support-Ticket-Provider-Execute-Proof

## Why this slice exists

The support-ticket input-provider chain had route proof for preview and plan,
but the session had not locked the final provider-backed execute handoff. A
real support-ticket CSV should move through the Atlas provider, the
Content Ops execute route, and the FAQ Markdown generator without a DB or LLM
dependency.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add a focused execute-route regression test for the Atlas support-ticket
   input provider.
2. Use the packaged support-ticket CSV fixture instead of synthetic inline-only
   rows.
3. Keep the execution services offline and narrowly scoped to FAQ Markdown so
   this slice proves the provider handoff without spending model calls.

### Files touched

- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-Atlas-Support-Ticket-Provider-Execute-Proof.md`

## Mechanism

The new test creates a Content Ops control-surface router with:

- `input_provider=build_content_ops_input_provider()`
- `execution_services_provider` configured with `TicketFAQMarkdownService`
- a tenant scope provider

It posts the repository support-ticket CSV rows to `/content-ops/execute` with
`outputs=["faq_markdown"]`, then asserts the route completes, the provider
normalizes all four ticket rows, and the FAQ output checks pass.

## Intentional

- This does not call the production Atlas-mounted execute route because that
  route resolves DB-backed services. The goal here is to prove the provider
  handoff into execute/generation with controlled offline services.
- This does not add new FAQ generation behavior. FAQ implementation remains
  owned by the FAQ lane.
- This keeps the fixture small and committed. Larger CFPB-derived artifacts
  remain local validation fixtures rather than new repository data.

## Deferred

- Full 1,000+ row CFPB execution should use the file-ingestion or persisted
  import path. Inline source material is bounded by the control-surface request
  model and can reject very large real rows before the provider runs.
- Persisted import lookup remains with the file-ingestion/backend validation
  lane.
- Parked hardening: none.

## Verification

- Python compile check for `tests/test_atlas_content_ops_input_provider.py` -
  passed.
- Pytest focused provider suite: `tests/test_atlas_content_ops_input_provider.py`
  - 10 passed, 1 warning.
- Manual provider-backed route smoke with
  `extracted_content_pipeline/examples/support_ticket_sources.csv`: preview
  runnable, plan executable, FAQ execute completed, 4 ticket rows accepted,
  all output checks true.
- Manual provider-backed route smoke with the local CFPB-derived 1,000-row
  fixture showed inline execute works through 900 rows and rejects 999+ rows
  with the existing request-size guard.
- FAQ scale smoke against the local CFPB 1,000-row artifact - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Execute-route provider test | ~75 |
| **Total** | **~150** |
