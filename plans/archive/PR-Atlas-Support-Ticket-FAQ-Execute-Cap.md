# PR-Atlas-Support-Ticket-FAQ-Execute-Cap

## Why this slice exists

PR #912 proved the Atlas-mounted Content Ops preview and plan routes apply the
support-ticket input provider. The remaining thin route-level validation gap is
execute: the host should prove a support-ticket request can reach the real
deterministic FAQ generator through the API route at the synchronous 1,000-row
cap, without a database or model provider.

This slice closes that gap with one focused host-route test.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

Slice phase: Functional validation.

1. Add one Atlas host execute-route test that wires:
   - `build_content_ops_input_provider()`
   - `create_content_ops_control_surface_router(...)`
   - real `TicketFAQMarkdownService`
2. Send 1,000 inline support-ticket rows through `/content-ops/execute` using
   only the deterministic `faq_markdown` output.
3. Assert the route returns completed FAQ output with 1,000 accepted and
   rendered ticket sources, passing output checks, and no saved IDs.

### Files touched

- `plans/PR-Atlas-Support-Ticket-FAQ-Execute-Cap.md`
- `tests/test_atlas_content_ops_input_provider.py`

## Mechanism

The test constructs the extracted Content Ops control-surface router in-memory
with the Atlas support-ticket input provider and a `ContentOpsExecutionServices`
bundle containing `TicketFAQMarkdownService()`. It then calls the route endpoint
directly with 1,000 support-ticket rows under `inputs.source_material` and
`outputs=["faq_markdown"]`.

The explicit output keeps the route focused on the deterministic FAQ service
instead of requiring landing-page or blog services.

## Intentional

- No production code changes.
- No database assertion. Persistence is already covered by FAQ lifecycle smokes;
  this route proof keeps the service repository-free.
- No 50,000-row hosted route run. The synchronous route cap is 1,000 rows; larger
  uploads remain a background-job/product slice.

## Deferred

- Hosted persisted FAQ execute proof can follow with a fake or local Postgres
  repository if reviewers need route-level `saved_ids`.
- Public/help-center FAQ rendering remains outside this route-validation slice.
- Parked hardening: none.

## Verification

- Focused host input-provider pytest passed after rebase: 11 passed, 1 warning.
- Related support-ticket provider pytest sweep passed after rebase: 37 passed, 1 warning.
- Py compile passed for the host input-provider test file.
- Local PR review passed: `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 80 |
| Host route test | 70 |
| **Total** | 150 |
