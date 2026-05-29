# PR-Content-Ops-FAQ-Startup-Migration-Drift

## Why this slice exists

The content-ops FAQ route now depends on Atlas API startup being quiet enough
to reveal real readiness failures. A parked hardening item records startup
noise from host migrations:
`column "updated_at" of relation "b2b_campaigns" does not exist`.

The schema drift is in the migration chain itself. Migration 066 creates
`b2b_campaigns` without `updated_at`, migration 068 only includes
`updated_at` in a `CREATE TABLE IF NOT EXISTS` fallback, and migration 309 later
assumes the column exists while cancelling campaigns attached to superseded
sequences. This slice fixes the migration source instead of suppressing the
warning.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Production hardening

1. Add a missing pre-068 migration that adds `b2b_campaigns.updated_at` for
   existing 066-created tables before later campaign-sequence migrations run.
2. Add a focused migration-chain test that proves the new migration fills the
   exact column required by migration 309, sorts before it, and executes against
   Postgres when `ATLAS_MIGRATION_TEST_DATABASE_URL` is set.
3. Enroll that test in a path-triggered GitHub Actions job with a Postgres
   service so CI executes the migration repair instead of only string-checking
   the SQL.
4. Sync the migration into `extracted_content_pipeline` through the package
   manifest so standalone content-pipeline installs get the same schema repair.
5. Remove the corresponding parked `HARDENING.md` item.

### Files touched

| File | Purpose |
|---|---|
| `.github/workflows/atlas_b2b_campaign_migration_checks.yml` | Path-triggered CI lane with a Postgres service for campaign migration tests. |
| `atlas_brain/storage/migrations/067_b2b_campaigns_updated_at.sql` | Backfill the missing timestamp column in the canonical migration chain. |
| `extracted_content_pipeline/manifest.json` | Map the new host migration into the standalone content pipeline package. |
| `extracted_content_pipeline/storage/migrations/067_b2b_campaigns_updated_at.sql` | Synced package copy of the migration. |
| `tests/test_atlas_b2b_campaign_migrations.py` | Lock the migration ordering, 309 dependency, and executable Postgres repair. |
| `tests/test_extracted_campaign_manifest.py` | Lock the package manifest copy with the other core campaign migrations. |
| `HARDENING.md` | Drain the startup migration warning item. |
| `plans/PR-Content-Ops-FAQ-Startup-Migration-Drift.md` | Slice contract. |

## Mechanism

The new migration runs between 066 and 068:

```sql
ALTER TABLE b2b_campaigns
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
```

On databases where 066 already created the table and 068 is already recorded,
the new 067 migration is still pending and sorts before 309, so it repairs the
schema before 309 reaches `SET updated_at = NOW()`. On first-time databases,
066 creates the table, 067 adds the column, and later migrations keep the same
shape.

The static test verifies that `067_b2b_campaigns_updated_at.sql` is ordered
after 066 and before 309, adds `updated_at`, and that 309 still depends on that
column. The executable test uses `asyncpg` when `ATLAS_MIGRATION_TEST_DATABASE_URL`
is configured: it applies 066 in a temporary schema, proves the 309-style
`updated_at = NOW()` update fails before 067, applies 067, asserts the column is
`TIMESTAMPTZ NOT NULL` with no null backfill rows, proves the 309-style update
succeeds, and reapplies 067 to prove idempotency.

The new workflow provisions Postgres 16 and sets
`ATLAS_MIGRATION_TEST_DATABASE_URL`, so CI always runs the executable branch.
The same migration is mapped into `extracted_content_pipeline/manifest.json`,
then synced into `extracted_content_pipeline/storage/migrations/067_b2b_campaigns_updated_at.sql`.

## Intentional

- This does not edit migration 309. The failure is caused by an earlier schema
  omission, and adding a missing pre-309 migration lets pending host databases
  repair themselves before 309 runs.
- This does not add a broad full-chain migration runner. The executable test is
  intentionally scoped to the exact 066 -> 067 -> 309-style failure and repair
  that produced the startup warning.

## Deferred

Parked hardening: none.

## Verification

To run before opening the PR:

```bash
python -m py_compile tests/test_atlas_b2b_campaign_migrations.py tests/test_extracted_campaign_manifest.py
python -m pytest tests/test_atlas_b2b_campaign_migrations.py tests/test_extracted_campaign_manifest.py -q
bash scripts/validate_extracted_content_pipeline.sh
python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
python scripts/audit_extracted_standalone.py --fail-on-debt
bash scripts/check_ascii_python.sh
bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/content-ops-faq-startup-migration-drift-pr-body.md
```

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 114 |
| Workflow | 62 |
| Host migration | 8 |
| Extracted package sync | 12 |
| Tests | 101 |
| Hardening cleanup | 11 |
| **Total** | **308** |
