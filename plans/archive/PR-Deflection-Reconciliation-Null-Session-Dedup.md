# PR-Deflection-Reconciliation-Null-Session-Dedup

## Why this slice exists

The paid-but-report-missing reconciliation ledger (#1462) promises idempotency
"on (account_id, request_id, stripe_session_id) so Stripe retries do not create
duplicate ledger rows" -- but that guarantee is false when the session id is
NULL. Migration 336 made `stripe_session_id` nullable with
`UNIQUE (account_id, request_id, stripe_session_id)`, and Postgres treats NULL as
DISTINCT in a UNIQUE constraint, so `ON CONFLICT (...) DO NOTHING` does not dedup
NULL-session rows. The billing webhook forwarded `session_id or None`
(`billing.py`), so a missing/empty Stripe session id became NULL and a Stripe
at-least-once redelivery -- or a sibling event type (`completed` +
`async_payment_succeeded`) for the same checkout -- wrote a duplicate row.

Low probability in normal operation (real Stripe events always carry a session
id), but it is a money-path data-integrity gap, and the #1517 apply test only
exercised the non-null dedup, so the test gave false confidence about an
idempotency guarantee the code did not hold.

## Scope (this PR)

Ownership lane: content-ops/deflection-billing
Slice phase: Production hardening

1. `atlas_brain/storage/migrations/337_content_ops_deflection_reconciliation_null_session.sql`:
   collapse existing NULL-equivalent duplicate rows (lowest id wins), backfill
   `NULL -> ''`, then `SET DEFAULT ''` + `SET NOT NULL` so the existing UNIQUE
   dedups uniformly. Re-runnable; no CONCURRENTLY (transaction-safe).
2. `atlas_brain/content_ops_deflection_reconciliation.py`: bind
   `stripe_session_id or ""` in the insert so a missing session is never NULL
   (protects every caller against the NOT NULL column).
3. `atlas_brain/api/billing.py`: forward `session_id or ""` (not `or None`) at
   the reconciliation call site.
4. `tests/test_deflection_migrations_apply.py`: apply 337 in the chain; assert
   `stripe_session_id` is NOT NULL, that the going-forward empty-session dedup
   holds, and (new test) that 337 collapses two pre-existing NULL duplicates to
   one row backfilled to `''`.
5. `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`: new unit
   test asserting an empty Stripe session id binds `''` (not None) in the
   reconciliation insert.
6. `.github/workflows/atlas_deflection_migration_apply_checks.yml`: add migration
   337 to the path triggers.

### Review Contract
- Acceptance criteria:
  - [ ] After 337, `stripe_session_id` is NOT NULL and defaults to `''`.
  - [ ] 337 collapses pre-existing NULL-session duplicates for the same
        (account_id, request_id) to one row, backfilled to `''`.
  - [ ] Two missing-session events for the same checkout dedup to one ledger row
        (the gap NULL left open).
  - [ ] `record_paid_report_missing` / the billing handler bind `''`, not None,
        for a missing session id.
  - [ ] No change to the non-null dedup behavior or the 2xx/409 disposition.
- Affected surfaces: the reconciliation ledger schema + insert and the billing
  webhook reconciliation call site, plus the migration-apply and billing tests.
  No buyer-facing report shape change; no API change.
- Risk areas: money-path data integrity; migration safety against existing rows;
  test isolation (both apply tests share one CI database).
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R4
  (data and migration safety), R5 (backward compatibility), R6 (webhook), R8
  (billing / payment), R10 (maintainability), R12 (CI enrollment).

### Files touched

- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `atlas_brain/api/billing.py`
- `atlas_brain/content_ops_deflection_reconciliation.py`
- `atlas_brain/storage/migrations/337_content_ops_deflection_reconciliation_null_session.sql`
- `plans/PR-Deflection-Reconciliation-Null-Session-Dedup.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

The fix is two-sided so the gap cannot reappear. The schema can no longer hold a
NULL (337 enforces NOT NULL DEFAULT ''), and both the producer (the billing
handler and `record_paid_report_missing`) coerce a missing session to `''`. The
empty string is a normal, conflict-eligible value, so the existing
`(account_id, request_id, stripe_session_id)` UNIQUE and the unchanged
`ON CONFLICT (...) DO NOTHING` now dedup a missing-session retry. The migration
collapses any rows the gap already produced by `COALESCE(stripe_session_id, '')`,
keeping the lowest id, then backfills the survivor.

The apply tests share one CI service database, so each test drops the deflection
tables first (`_reset`) -- test 1 applies 337 (NOT NULL), which would otherwise
break test 2's pre-337 NULL insert.

## Intentional

- Coerce at both the schema and the producer, not just one: NOT NULL stops a
  future caller from re-introducing the gap; the `or ""` keeps inserts valid
  against the NOT NULL column.
- Keep the existing 3-column UNIQUE + ON CONFLICT unchanged (no expression
  index) -- once NULL is impossible, the plain constraint dedups `''` correctly,
  which is simpler and lower-risk than an expression index.
- Only the reconciliation call site changes; `mark_paid`'s
  `payment_reference=session_id or None` (a different, nullable column with no
  dedup concern) is left alone.

## Deferred

None.

Parked hardening: none.

## Verification

Ran locally against a fresh throwaway Postgres and the unit suite:

```
ATLAS_MIGRATION_TEST_DATABASE_URL=... pytest tests/test_deflection_migrations_apply.py
pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py
python -m py_compile atlas_brain/api/billing.py atlas_brain/content_ops_deflection_reconciliation.py
bash scripts/check_ascii_python.sh
```

- Migration apply tests: 2 passed (chain applies with NOT NULL; empty-session
  dedup holds; 337 collapses two pre-existing NULL duplicates to one `''` row).
- Billing reconciliation tests: 47 passed, including the new empty-session bind
  assertion.
- ASCII + py_compile clean.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/337_content_ops_deflection_reconciliation_null_session.sql` | ~40 |
| `tests/test_deflection_migrations_apply.py` | ~120 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | ~35 |
| `atlas_brain/content_ops_deflection_reconciliation.py` | ~6 |
| `atlas_brain/api/billing.py` | ~5 |
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | ~2 |
| `plans/PR-Deflection-Reconciliation-Null-Session-Dedup.md` | ~115 |
| **Total** | **~323** |
