# PR-FAQ-Macro-Writeback-Publish-Route

## Why this slice exists

The FAQ macro writeback core loop now exists behind
`FAQMacroWritebackPublishService`: preview gate, provider publish, idempotent
adapter behavior, and conservative FAQ draft status transition. The next
missing product slice is a scoped entry point so the generated asset review API
can trigger that service for one approved FAQ draft.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add an optional macro publish provider factory to the generated asset router.
2. Add a FAQ-only `POST /content-assets/faq_markdown/drafts/{draft_id}/publish-macros`
   route.
3. Validate the route is FAQ-only, tenant-scoped, provider-backed, and returns
   the service summary without hiding non-clean outcomes.
4. Add focused API tests for clean publish, provider missing, wrong asset, and
   pending-reconcile result handling.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-Route.md` — plan for this slice.
- `extracted_content_pipeline/api/generated_assets.py` — publish-macros route and provider injection.
- `tests/test_extracted_content_asset_api.py` — route coverage.

## Mechanism

The generated asset router gains a `macro_publish_provider` callable, resolved
with the existing `_resolve_required_provider` helper. The new route:

1. Accepts only `asset == "faq_markdown"`.
2. Requires a UUID-shaped `draft_id` before touching Postgres.
3. Resolves pool, tenant scope, and macro provider.
4. Calls `FAQMacroWritebackPublishService(PostgresTicketFAQRepository(pool), provider)`.
5. Returns `summary.as_dict()` plus `account_id` and `asset`.
6. Returns 404 when the tenant-scoped FAQ draft is not found.

The route does not create a provider itself. Host code decides whether the
provider is Zendesk, dry-run, or another support platform adapter.

## Intentional

- No UI button in this slice. The route is the smallest real trigger that can
  be tested end to end against the existing API boundary.
- No live Zendesk credential wiring here. The router accepts a provider factory
  so deployment code can inject the already-built adapter without the API module
  owning secrets.
- Pending reconciliation remains a non-clean 200 summary, not a duplicate POST
  retry and not an HTTP 500. The provider completed its contract; the operator
  needs the summary state surfaced.

## Deferred

- `PR-FAQ-Macro-Writeback-Provider-Wiring`: instantiate the Zendesk provider
  from host credentials / mapping repository in the deployed app.
- `PR-FAQ-Macro-Writeback-Pending-Reconcile`: backfill pending Zendesk mappings
  by looking up reserved title/category metadata.
- `PR-FAQ-Macro-Writeback-Publish-UI`: add the review UI action once the route
  is available.

Parked hardening: none

## Verification

- python -m pytest tests/test_extracted_content_asset_api.py -k 'publish_macros or ticket_faq' — 7 passed, 56 deselected.
- python -m py_compile extracted_content_pipeline/api/generated_assets.py tests/test_extracted_content_asset_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- python scripts/check_extracted_imports.py — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py — passed.
- python scripts/smoke_extracted_pipeline_imports.py — passed.
- python scripts/smoke_extracted_pipeline_standalone.py — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-route.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~90 |
| API route | ~70 |
| Tests | ~140 |
| Total | ~300 |
