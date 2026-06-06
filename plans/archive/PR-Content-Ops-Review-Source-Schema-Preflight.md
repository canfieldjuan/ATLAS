# PR-Content-Ops-Review-Source-Schema-Preflight

## Why this slice exists

The live G2 review-source Postgres smoke now proves readiness, export, and
ingestion inspection, but on a host database without extracted Content Ops
migrations it fails late with `UndefinedTableError` when import reaches
`campaign_opportunities`. Operators need the smoke to fail before import with
the migration command to run.

## Scope (this PR)

- Add a schema preflight to `scripts/smoke_content_ops_review_source_postgres.py`.
- Verify the configured opportunity table and `b2b_campaigns` exist before
  importing source rows or generating persisted drafts.
- Add a regression test for a missing `campaign_opportunities` table.
- Update README, host runbook, status, and coordination docs.

### Files touched

- `scripts/smoke_content_ops_review_source_postgres.py`
- `tests/test_smoke_content_ops_review_source_postgres.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Source-Schema-Preflight.md`

## Mechanism

The smoke uses `SELECT to_regclass($1)::text` for the configured opportunity
table and `b2b_campaigns` after source export and ingestion inspection pass.
If either relation is missing, the smoke returns `ok=false` with an actionable
`scripts/run_extracted_content_pipeline_migrations.py` instruction and skips
import/generation.

## Intentional

- Keep migrations explicit; the smoke does not apply schema changes.
- Preserve the existing success path and target-id validation.
- Keep the opportunity table configurable through the existing
  `--opportunity-table` argument.

## Deferred

- Live migration execution on the local Atlas database.
- A broader host DB readiness command for every generated-asset table.
- Automatic migration application from the smoke command.

## Verification

- Focused Postgres smoke tests -> `8 passed`.
- Python compile check for smoke script/tests -> passed.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Smoke script | 30 |
| Tests | 36 |
| Docs/status | 16 |
| Coordination and plan | 69 |
| Total | 151 |
