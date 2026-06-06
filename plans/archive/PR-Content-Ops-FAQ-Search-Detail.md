# PR-Content-Ops-FAQ-Search-Detail

## Why this slice exists
The FAQ deflection search route returns compact result rows with `faq_id`, which
is enough to rank matches but not enough for the searchable demo to render the
full generated FAQ artifact. The product direction is to keep search retrieval
fast and compact, then hydrate the selected FAQ by id. This slice adds that
thin, tenant-scoped detail path without changing generation or indexing.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a single-draft read method to the FAQ Markdown repository.
2. Add a detail route under the existing FAQ deflection search router.
3. Return the full generated FAQ body, items, checks, warnings, and metadata for
   the requested `faq_id`.
4. Preserve tenant fail-closed behavior: missing `account_id` is rejected and a
   different tenant receives 404.
5. Reject malformed FAQ ids as 400 before database access.
6. Cover the new route with fake-pool tests and one real-Postgres route
   composition test.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Detail.md`
- `extracted_content_pipeline/api/faq_search.py`
- `extracted_content_pipeline/ticket_faq_ports.py`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `tests/test_extracted_ticket_faq_postgres.py`
- `tests/test_extracted_ticket_faq_search_api.py`

## Mechanism
`PostgresTicketFAQRepository.get_draft(...)` selects one row from
`ticket_faq_markdown` by `id` and `account_id`, returning the existing
`TicketFAQDraft` shape or `None`. `create_faq_deflection_search_router(...)`
accepts a second repository factory for FAQ Markdown hydration and mounts
`GET /content-ops/faq-deflection-search/{faq_id}`. The route validates `faq_id`
as a UUID before database access, resolves the same host-provided tenant scope
and database pool used by search, then returns `TicketFAQDraft.as_dict()` with
an explicit `account_id` field for the API envelope. Misses return 404.

## Intentional
- Search results remain compact. The full Markdown artifact is fetched only
  after a caller selects a `faq_id`.
- This route uses the existing router factory and tenant resolver so host
  mounting, bearer auth dependencies, and pool lifecycle stay in one place.
- The detail route does not filter by status. Possession of the tenant-scoped
  `faq_id` is enough for internal/demo hydration; review status still lives on
  the returned draft.
- The real-Postgres test skips when `EXTRACTED_DATABASE_URL` or `DATABASE_URL`
  is absent, matching the lane's integration-test pattern.

## Deferred
- The hosted large-upload/backpressure item remains parked in `HARDENING.md`
  as `FAQSCALE-1`; this read route does not exercise upload or generation load.
- A deployed-host contract checker for the detail route is deferred until the
  demo consumes this endpoint shape.
- Returning a redacted/public detail envelope is deferred; this slice is for the
  authenticated content-ops API mounted with host dependencies.

## Verification
- pytest tests/test_extracted_ticket_faq_postgres.py tests/test_extracted_ticket_faq_search_api.py -q passed with 28 tests and 1 skip because no database URL is configured in this checkout.
- python -m compileall over the changed Python files passed.
- Extracted package guardrail commands passed: validate_extracted_content_pipeline, forbid_atlas_reasoning_imports, audit_extracted_standalone --fail-on-debt, and check_ascii_python.
- bash scripts/local_pr_review.sh passed from a temporary clean worktree at this commit because unrelated untracked files exist in the shared checkout.
- python scripts/audit_pr_session_drift.py origin/main passed in the primary worktree and checked 0 open PRs.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 68 |
| Repository + port | 30 |
| API route | 70 |
| Tests | 180 |
| **Total** | **348** |
