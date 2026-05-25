# PR-Content-Ops-FAQ-Search-Backfill

## Why this slice exists
PR-Content-Ops-FAQ-Search-Review-Index wires new FAQ review actions into
`ticket_faq_search_documents`, but already-approved FAQ drafts remain outside the
search projection until someone re-reviews them. This slice adds the thin ops path
needed to backfill approved historical drafts through the same projection helpers.
The slice is over the usual 400 LOC budget because the backfill is a Postgres
composition and now carries the lane-default real-Postgres integration coverage:
multi-tenant projection is tested in the same PR as the write path.

## Scope (this PR)
Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a dry-run-first backfill function for approved ticket FAQ drafts.
2. Reuse the review-index projection helpers and search repository write path.
3. Add a small CLI wrapper for operators to run the backfill with `--apply`.
4. Test dry-run counts, applied writes, account filtering, and skipped incomplete
   projection keys.
5. Pin multi-tenant per-row account projection with fake-pool and real-Postgres
   coverage.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Backfill.md`
- `extracted_content_pipeline/ticket_faq_postgres.py`
- `scripts/backfill_ticket_faq_search_documents.py`
- `tests/test_extracted_ticket_faq_postgres.py`
- `tests/test_extracted_ticket_faq_search_postgres.py`

## Mechanism
`backfill_ticket_faq_search_documents(...)` scans `ticket_faq_markdown` rows by
status, optional account, and optional limit. Each row is converted into the same
`TicketFAQDraft` shape used by review indexing. The backfill resolves the
projection key with the row's persisted `account_id`, builds documents with
`build_ticket_faq_search_documents(...)`, and calls
`PostgresTicketFAQSearchRepository.replace_documents(...)` only when `apply=True`.

Dry-run mode returns counts without writing. Rows missing a complete projection
key are skipped and counted instead of causing a partial backfill crash.
The real-Postgres integration test inserts two accounts with a unique status,
runs dry-run and apply, and verifies each account can only search its own
projected rows.

## Intentional
- The default CLI mode is dry-run. Operators must pass `--apply` to mutate
  `ticket_faq_search_documents`.
- The backfill lives beside the FAQ Postgres adapter so it can reuse the
  canonical row-to-draft conversion and projection path.
- Existing approved drafts with empty FAQ items still clear stale projection rows
  when their account/corpus/FAQ key is complete.
- The integration test uses a unique non-default status to avoid scanning or
  rewriting unrelated approved rows in a shared developer database.

## Deferred
- Scheduled or automatic backfill is deferred; this PR only adds the explicit ops
  command.
- Retry/report persistence for per-row projection failures is deferred until a
  real legacy-data failure appears during an applied run.

## Verification
- `pytest tests/test_extracted_ticket_faq_postgres.py tests/test_extracted_ticket_faq_search.py tests/test_extracted_ticket_faq_search_postgres.py -q` passed with 33 tests and 3 skips because no database URL is configured in this checkout.
- Python compile check for the changed backfill module, CLI, and focused tests passed.
- `python scripts/backfill_ticket_faq_search_documents.py --help` passed.
- Package guardrails passed: validate extracted content pipeline, forbid Atlas reasoning imports, standalone audit, ASCII Python.

## Estimated diff size
| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Backfill function | 110 |
| CLI | 60 |
| Tests | 256 |
| **Total** | **502** |
