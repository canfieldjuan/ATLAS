# PR: Deflection migration apply check (real DB)

## Why this slice exists

The #1462 reconciliation slice added migration `336_content_ops_deflection_paid_reconciliation.sql`
and the handler relies on its `ON CONFLICT (account_id, request_id, stripe_session_id)`
idempotency. Nothing exercised that DDL against a real database: the
migrations-runner CI lane is no-DB (fake pool + Path checks), and a repo-wide
`run_migrations` fresh apply is not viable -- migration 076 references a
`product_metadata` table that no migration creates (the app creates it
out-of-band), so a full fresh apply fails at 076.

This slice adds the operator-approved real-DB check, scoped to the deflection
chain (which has no `product_metadata` dependency): apply 328 -> 332 -> 336 to a
fresh Postgres and verify the tables and the reconciliation table's idempotency
constraint. It is the deferred follow-up from the #1462 review.

## Scope (this PR)

Ownership lane: content-ops/deflection-billing
Slice phase: Robust testing

1. Add `tests/test_deflection_migrations_apply.py`: env-guarded
   (`ATLAS_MIGRATION_TEST_DATABASE_URL`), applies the three deflection migration
   files in order to a fresh database, asserts the three tables exist, asserts
   `content_ops_deflection_paid_reconciliation` has its expected columns, and
   proves the `(account_id, request_id, stripe_session_id)` uniqueness dedupes a
   second insert (the `record_paid_report_missing` ON CONFLICT guarantee).
2. Add `.github/workflows/atlas_deflection_migration_apply_checks.yml`: a
   Postgres service (fresh `atlas_migration_tests` DB) + the env var so the test
   runs against a real database in CI.

Out of scope: making the full migration set fresh-appliable (the missing
`product_metadata` migration) -- separate migration-debt issue; any production
code change.

### Review Contract

- Acceptance criteria:
  - [ ] The deflection chain (328 -> 332 -> 336) applies in order to a fresh Postgres.
  - [ ] All three `content_ops_deflection*` tables exist after apply.
  - [ ] `content_ops_deflection_paid_reconciliation` has its expected columns.
  - [ ] A second insert on `(account_id, request_id, stripe_session_id)` dedupes
        (the `ON CONFLICT DO NOTHING` guarantee `record_paid_report_missing` relies on).
  - [ ] The test skips cleanly when `ATLAS_MIGRATION_TEST_DATABASE_URL` is unset, so
        local unit runs are unaffected.
- Affected surfaces: one env-guarded test (`tests/test_deflection_migrations_apply.py`)
  and one CI workflow (`.github/workflows/atlas_deflection_migration_apply_checks.yml`,
  Postgres service).
  No production code path changes.
- Risk areas: CI-only behavior; the test must skip without the env var; the job stays
  dependency-light (asyncpg only, resolves the migrations dir by path with no
  `atlas_brain` import).
- Reviewer rules triggered: R2 (failure-branch / fixtures), R4, R12 (CI runs the
  enrolled test).

### Files touched

- `.github/workflows/atlas_deflection_migration_apply_checks.yml`
- `plans/PR-Deflection-Migration-Apply-Check.md`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

The test resolves the migrations directory by path
(`Path(__file__).resolve().parents[1] / "atlas_brain" / "storage" / "migrations"`)
rather than importing `atlas_brain`, so the CI job needs no application
dependencies beyond `asyncpg` (no `pydantic`/`pydantic-settings`). It executes
each chain file's SQL with `conn.execute` (autocommit), then queries `pg_tables`
/ `information_schema.columns` and performs a two-insert ON CONFLICT dedupe
check, cleaning up its test rows in `finally`. Without the env var it skips, so
local unit runs are unaffected.

The chain (328 reports, 332 deliveries, 336 reconciliation) is standalone --
neither 328 nor 332 has foreign keys to other tables, so the three files apply
to an empty database in order.

## Intentional

- Scoped to the deflection chain, not `run_migrations` over the full set, which
  fails at 076 (`product_metadata`) -- a pre-existing, out-of-scope migration-set
  property.
- Path-based migrations-dir resolution keeps the CI job dependency-light and
  avoids the `atlas_brain` import chain (the pydantic gap that bit the
  migrations-runner workflow).
- The dedupe assertion targets the #1462 money-path guarantee specifically (the
  reconciliation ledger must not double-record a retried checkout event).

## Deferred

- Making the full migration set fresh-appliable (add a `product_metadata`
  migration / fix 076 ordering) -> separate migration-debt issue.

Parked hardening: none.

## Verification

- Ran `tests/test_deflection_migrations_apply.py` against a fresh throwaway
  Postgres database (chain applied; tables + reconciliation columns asserted;
  ON CONFLICT dedupe verified) -- passed.
- Confirmed the test skips cleanly when `ATLAS_MIGRATION_TEST_DATABASE_URL` is
  unset (local unit runs unaffected).
- Verified the chain (328/332/336) applies to a fresh database with no missing
  dependencies.
- Non-ASCII scan of the new test + workflow -- clean.
- Python compile check for `tests/test_deflection_migrations_apply.py` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_deflection_migration_apply_checks.yml` | 59 |
| `plans/PR-Deflection-Migration-Apply-Check.md` | 97 |
| `tests/test_deflection_migrations_apply.py` | 120 |
| **Total** | **276** |
