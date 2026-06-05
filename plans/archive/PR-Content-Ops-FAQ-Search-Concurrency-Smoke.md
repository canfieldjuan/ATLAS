# PR-Content-Ops-FAQ-Search-Concurrency-Smoke

## Why this slice exists

The FAQ generator has batch-scale coverage and the search route is now mounted,
but retrieval under concurrent demo traffic is a separate read-path risk. A few
users querying different uploaded ticket corpora at the same time can expose
tenant/corpus filter mistakes, slow full-text retrieval, or connection-pool
behavior that batch ingestion tests do not exercise.

This slice adds an operator smoke that uses the real Postgres search projection
and produces a machine-readable result artifact with latency and isolation
signals.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: robust testing.

1. Add a Postgres-backed FAQ search concurrency smoke script.
2. Seed multiple accounts/corpora into `ticket_faq_markdown` and
   `ticket_faq_search_documents`.
3. Run concurrent filtered hit/miss searches through
   `PostgresTicketFAQSearchRepository`.
4. Emit JSON with request counts, p50/p95/max latency, error count, and isolation
   failures.
5. Add focused unit tests for the smoke's deterministic planning, latency, and
   gate-summary helpers.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Concurrency-Smoke.md`
- `scripts/smoke_content_ops_faq_search_concurrency.py`
- `tests/test_smoke_content_ops_faq_search_concurrency.py`

## Mechanism

The script reads `--database-url`, `EXTRACTED_DATABASE_URL`, `DATABASE_URL`, or
the Atlas DB settings fallback. It applies the existing FAQ Markdown and FAQ
search migrations, seeds shared corpus IDs across multiple accounts for the run,
then runs concurrent repository searches with:

- hit queries that must return only the requested account/corpus;
- miss queries that must return zero rows;
- a semaphore-limited concurrency level so operators can vary pressure without
  changing code.

The JSON result includes `ok`, `latency`, `requests`, `errors`, `isolation`, and
`seed` sections. Any search exception, wrong-tenant row, wrong-corpus row,
expected hit with no rows, or unexpected miss result fails the smoke. Sharing
corpus IDs across accounts makes a dropped `account_id` predicate observable as a
wrong-tenant row instead of being masked by corpus filtering.

## Intentional

- This does not call the hosted HTTP route. The route is a thin wrapper over the
  same repository; this slice targets the durable read pattern directly.
- No latency threshold fails the run yet. The first smoke captures measured
  latency so we can set a defensible threshold after seeing real numbers.
- Seeded rows are deleted by default. `--keep-data` exists only for local
  debugging.

## Deferred

- HTTP-route concurrency with auth tokens remains deferred until a deployed
  backend URL/token is available.
- Latency SLO thresholds are deferred until we have baseline output from this
  smoke in the target environment.

## Verification

- pytest tests/test_smoke_content_ops_faq_search_concurrency.py -q - 5 passed.
- python -m py_compile scripts/smoke_content_ops_faq_search_concurrency.py tests/test_smoke_content_ops_faq_search_concurrency.py - passed.
- python scripts/smoke_content_ops_faq_search_concurrency.py --account-count 3 --corpora-per-account 2 --documents-per-corpus 3 --iterations 24 --concurrency 6 --pool-size 4 --output-result tmp/faq_search_concurrency_smoke.json --json - passed with EXTRACTED_DATABASE_URL constructed from local `.env.backup` ATLAS_DB_* fields; 24 requests, 0 isolation failures, p50 1.010468 ms, p95 6.789891 ms, max 7.104989 ms.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Smoke script | 378 |
| Tests | 144 |
| **Total** | **610** |

This is above the 400 LOC target because the script includes DB setup, seed
cleanup, concurrency, result serialization, and focused tests in one usable
operator slice.
