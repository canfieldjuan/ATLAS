# PR-EOM-Public-Onboarding-Schema-Repair

## Why this slice exists

The public Website → Tracker → Atlas onboarding path is deployed but cannot be
safely enabled. A live Tracker-to-Atlas probe reached the authenticated Atlas
route and received `503 Public onboarding is not enabled`. Enabling its three
runtime settings in a no-write preflight then made Atlas's startup readiness
guard fail because the production token table is incomplete, despite its
current migration ledger recording the public-onboarding migration.

### Problem-derived contract

- Root cause: a prior, version-collided
  `382_eom_public_onboarding_tokens` migration was recorded under a synthetic
  negative ledger version after it created a partial token relation. The later
  `383_eom_public_onboarding_tokens` migration is `CREATE TABLE IF NOT
  EXISTS`, so PostgreSQL left that pre-existing relation unchanged while the
  runner recorded 383 as applied. The live, empty relation consequently lacks
  the immutable signing and prefill fields that enabled public-onboarding
  readiness requires.
- Correct fix must touch/change: add one new, atomic, forward-only migration
  that detects the known incomplete table shape, refuses to invent immutable
  values for a nonempty legacy relation, and repairs the empty relation in
  place with the exact signing/prefill column semantics required by migration
  383. Add a real-PostgreSQL regression test that uses the migration runner to
  prove both the empty-table repair and the nonempty-table fail-closed fence.
  The migration must be a new normal ledger entry; it must not alter migration
  history.
- Must not change: `382_*` and `383_*`; Website and Tracker code; public-route,
  HMAC, bearer, issuance, redemption, revocation, email, customer-handoff,
  payroll, receivables, commercial-billing, or CRM lifecycle semantics; and
  runtime public-onboarding configuration. Configuration remains a separate,
  post-merge production operation only after this migration proves readiness.

## Scope (this PR)

Ownership lane: eom-public-onboarding/schema-repair
Slice phase: Production hardening
Max files: 3

1. Repair an empty legacy `eom_public_onboarding_tokens` relation whose
   migration ledger already records 383 but whose immutable public-onboarding
   projection columns are absent.
2. Add database-level regression proof for the repair, fail-closed nonempty
   legacy state, and no-op behavior on an already complete relation.

### Review Contract

- Acceptance criteria:
  1. `384_eom_public_onboarding_tokens_schema_repair.sql` is an
     `-- atlas: atomic-bookkeeping` migration and is the only migration source
     changed. It contains an explicit pre-mutation fence: if any required
     immutable column is missing and the relation has rows, migration execution
     raises rather than adding unknown values.
  2. On the exact known empty legacy shape, the migration runner records 384
     and the repaired relation has each 383-only field with the 383 type,
     nullability, and signing-key check semantics. Settled by
     `tests/test_eom_public_onboarding_migration_repair.py` against a real
     disposable PostgreSQL schema.
  3. On the known nonempty legacy shape, migration execution fails and leaves
     the table's columns and row count unchanged. Settled by the same
     real-PostgreSQL test.
  4. On an already complete 383 relation that has a valid token row, migration
     execution records 384 without changing that row. Settled by the same
     real-PostgreSQL test.
  5. The existing 382 and 383 migration files remain byte-for-byte outside the
     diff. Settled by the final cold diff audit.
  6. The migration is run through `run_migrations(..., only={...})`, so its
     DDL and ledger record are proven together under the production runner's
     atomic-bookkeeping path rather than by executing a copied SQL string.
- Reachability proof: after merge/deployment, the live `atlas-api` startup
  runner applies 384 before public-onboarding configuration is set. With the
  explicit public settings then present, a Tracker public-session request with
  a fake bearer must traverse the configured service and return the expected
  unavailable-link `404`, not the current configuration-disabled `503`.
- Affected surfaces: Atlas migration ledger; the durable public-onboarding
  token relation required by `require_eom_funnel_data_store`; and the deployed
  Website → Tracker → Atlas onboarding handoff only.
- Risk areas: destructive schema repair, impossible immutable-data backfill,
  migration/ledger atomicity, legacy relation compatibility, and accidental
  configuration activation before datastore readiness.
- Reviewer rules triggered: R1, R2, R4, R5, R12, R14.

### Boundary-change enumeration

N/A - no application guard, validator, resolver, router, or admission
boundary changes. The migration restores the existing datastore contract;
`require_eom_funnel_data_store` remains unchanged.

### Deployed-config probing

N/A - this code diff changes no configuration lookup or fallback. The live
preflight established that Atlas's funnel API and Tracker bearer are set,
while all public-onboarding settings are absent. The explicit runtime
activation is deliberately sequenced after the merged migration and fresh
service startup; it is not a PR code change.

### Files touched

- `atlas_brain/storage/migrations/384_eom_public_onboarding_tokens_schema_repair.sql`
- `plans/PR-EOM-Public-Onboarding-Schema-Repair.md`
- `tests/test_eom_public_onboarding_migration_repair.py`

## Mechanism

Migration 384 first asks PostgreSQL whether any of the nine immutable
signing/prefill columns are absent. If so, it checks the legacy relation before
DDL: a nonempty table aborts because neither the signing-key fingerprint nor
the snapshot values can be reconstructed truthfully. For the observed empty
legacy relation, it adds the missing columns with the same type, nullability,
and fingerprint check used by 383. The atomic-bookkeeping marker makes the
schema changes and new ledger row one transaction under the existing runner.

## Intentional

- Do not rewrite the accidentally colliding 382 ledger record or edit 383:
  deployed databases may already record both, so a new forward migration is
  the recoverable compatibility path.
- Do not backfill or relax immutable fields for a nonempty legacy relation.
  Such data would be fabricated; fail closed and require an explicitly scoped
  forensic migration instead.
- Do not set the public enablement flag in source. The operational service must
  start successfully with 384 before its public authority is activated.

## Deferred

If a different deployment has a nonempty incomplete relation, resolve its
individual records and provenance in a separate operator-approved migration;
this repair intentionally refuses it.

Parked hardening: none.

## Verification

- Passed: `ATLAS_MIGRATION_TEST_DATABASE_URL=... pytest -q
  tests/test_eom_public_onboarding_migration_repair.py` — 3 passed against a
  disposable schema in the local development PostgreSQL database.
- Passed: `pytest -q tests/test_migrations_runner.py` — 30 passed, 1 skipped.
- Passed: `pytest -q tests/test_eom_public_onboarding.py` — 39 passed.
- Passed: Ruff check and formatter checks for
  `tests/test_eom_public_onboarding_migration_repair.py`.
- Environment limitation recorded: the two existing public-onboarding
  readiness integration tests skip because the configured test role cannot
  administer disposable PostgreSQL roles; the new migration test exercises
  the actual runner and DDL without that unavailable privilege.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/384_eom_public_onboarding_tokens_schema_repair.sql` | 55 |
| `plans/PR-EOM-Public-Onboarding-Schema-Repair.md` | 155 |
| `tests/test_eom_public_onboarding_migration_repair.py` | 400 |
| **Total** | **610** |
