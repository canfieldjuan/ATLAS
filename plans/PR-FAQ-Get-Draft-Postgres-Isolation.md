# FAQ Get Draft Postgres Isolation

## Why this slice exists

#1121 proved the execute route can load a selected FAQ draft ID through the
Atlas input-provider repository seam and feed both landing/blog generators. The
reviewer correctly called out the remaining completeness proof: the in-process
smoke verifies scope-threading, but not the real Postgres fetch isolation.

This slice adds the real-Postgres check that account B cannot load account A's
saved FAQ draft by ID.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Functional validation

1. Add an env-gated Postgres integration test for
   `PostgresTicketFAQRepository.get_draft(...)`.
2. Save a FAQ draft under one generated account, fetch it successfully with
   that account scope, and assert a different account scope returns `None` for
   the same draft ID.
3. Reuse the existing `EXTRACTED_DATABASE_URL` / `DATABASE_URL` integration-test
   convention and clean up inserted rows by generated account ID.

### Files touched

- `tests/test_extracted_ticket_faq_postgres.py`
- `plans/PR-FAQ-Get-Draft-Postgres-Isolation.md`

## Mechanism

The test imports `asyncpg` with `pytest.importorskip`, reads the same database
URL environment variables used by existing FAQ search integration tests, applies
the `325_ticket_faq_markdown.sql` migration, and uses the real
`PostgresTicketFAQRepository` to save a draft.

It then calls:

```python
await repo.get_draft(saved_id, scope=TenantScope(account_id=account_a))
await repo.get_draft(saved_id, scope=TenantScope(account_id=account_b))
```

The first call must return the draft; the second must return `None`.

## Intentional

- This is repository-level, not another execute-route smoke. #1121 already
  proved execute wiring; this slice proves the SQL-backed isolation boundary
  that the execute route relies on.
- The test is env-gated like the existing Postgres FAQ/search tests, so normal
  CI without a database skips it instead of failing.
- No production code changes are expected. If the test fails against a real DB,
  the repository query would be fixed here.

## Deferred

- Future PR: hosted/live execute runbook artifact if operators need a recorded
  environment proof that combines route execution and live Postgres in one
  smoke.
- Future PR: richer saved-FAQ picker with search/status filters if operators
  need more than the recent list.
- Parked hardening: none.

## Verification

Run locally:

- Command: python -m pytest tests/test_extracted_ticket_faq_postgres.py::test_get_draft_is_tenant_scoped_against_postgres -q
  - skipped locally: `EXTRACTED_DATABASE_URL` / `DATABASE_URL` not set
- Command: python -m pytest tests/test_extracted_ticket_faq_postgres.py -q
  - 18 passed, 1 skipped
- Command: python -m py_compile tests/test_extracted_ticket_faq_postgres.py
  - passed
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-get-draft-postgres-isolation.md

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Postgres isolation test | ~55 |
| Plan doc | ~78 |
| **Total** | **~133** |
