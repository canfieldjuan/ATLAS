# PR-Content-Ops-FAQ-Lifecycle-DB-1000-Run

## Why this slice exists

The FAQ generator has passed a 1,000-row file scale smoke and a 1,000-row
fake-pool lifecycle test. The remaining confidence gap is the real local
Postgres lifecycle: source rows -> generated FAQ Markdown -> persisted draft ->
draft export -> review status update -> reviewed export at 1,000 rows.

This slice records that real database run and logs any issues surfaced during
the test. If the run exposes a failure required for the flow to work, the fix
will land in this PR; non-blocking cleanup goes to `HARDENING.md`.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-io-tests

1. Use the derived Atlas local database DSN from `atlas_brain.storage.config`.
2. Apply pending extracted Content Ops migrations if the dry run shows the FAQ
   lifecycle table is not present.
3. Run `scripts/smoke_content_ops_faq_lifecycle.py` against the existing
   1,000-row CFPB source JSONL.
4. Record command output, source profile, lifecycle summary, and any failures in
   a validation doc.

### Files touched

- `plans/PR-Content-Ops-FAQ-Lifecycle-DB-1000-Run.md`
- `docs/extraction/validation/content_ops_faq_lifecycle_db_1000_row_run_2026-05-23.md`
- `HARDENING.md`

## Mechanism

The run will use the existing ignored source artifact:

```text
tmp/content_ops_faq_1000/cfpb_1000_source_rows.jsonl
```

The database URL is not checked into `.env`; it is derived at runtime through
`atlas_brain.storage.config.db_settings.dsn` and passed via
`EXTRACTED_DATABASE_URL` without printing credentials.

The lifecycle command writes its full JSON result to ignored `tmp/`. The
source-controlled validation doc records only aggregate proof points:

- input profile and lifecycle summary,
- saved/exported row counts,
- review status,
- output checks,
- issues surfaced and resolution status.

## Intentional

- This is a validation/evidence slice first. No production code changes are
  planned unless the real DB run fails in a way that blocks the lifecycle.
- Large generated JSON and Markdown artifacts remain under ignored `tmp/`.
- The local database is allowed to change because the point of this slice is a
  real DB lifecycle proof.
- The validation doc masks the database URL and does not include secrets.

## Deferred

- Browser upload coverage remains deferred until the UI upload path is active.
- A repeatable CI database container for this lifecycle smoke remains deferred;
  current CI coverage uses the fake async pool.
- Root `HARDENING.md` was scanned. The migration dry-run mismatch surfaced by
  this validation pass was parked there because it does not block the FAQ
  lifecycle flow.

## Verification

- Passed: derived Atlas DB settings without printing DSN (`localhost:5433/atlas`, user `atlas`, no password set).
- Observation: `python scripts/run_extracted_content_pipeline_migrations.py --dry-run --json` reported 28 would-apply migrations.
- Passed: `python scripts/run_extracted_content_pipeline_migrations.py --json` reported 0 applied, 28 skipped.
- Passed: `python scripts/smoke_content_ops_faq_lifecycle.py ... --min-source-rows 1000 --json` exited 0, saved one FAQ, exported one draft row, updated it to `published`, and exported one reviewed row.
- Passed: `python -m json.tool tmp/faq_lifecycle_smoke_20260523_scale1000.json`
- Observation: repeated migration dry-run still reports 28 dry-run entries after the actual run; parked in `HARDENING.md`.
- Pending local run: `bash scripts/local_pr_review.sh --allow-dirty`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Validation doc | ~205 |
| Parked hardening note | ~10 |
| **Total** | **~305** |
