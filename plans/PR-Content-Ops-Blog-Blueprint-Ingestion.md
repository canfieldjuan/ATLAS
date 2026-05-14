# PR: Content Ops Blog Blueprint Ingestion

## Why this slice exists

The deferred AI Content Ops backlog lists the blog blueprint population path as
the next P1 gap. Blog execution can now read `blog_blueprints`, but a standalone
host still needs a product-owned way to populate that table without hand-written
SQL or Atlas autonomous task fanout.

## Scope

Add a small JSON-to-Postgres ingestion seam for blog blueprints:

- Normalize host JSON rows into the existing `BlogBlueprint` model.
- Add a CLI that validates in dry-run mode or writes through the existing
  `PostgresBlogBlueprintRepository`.
- Document the host install command and include the new tests in the extracted
  Content Ops check runner.
- Replace the stale merged #505 coordination row with this active claim.

### Files touched

- NEW: `extracted_content_pipeline/blog_blueprint_ingest.py`
- NEW: `scripts/load_extracted_blog_blueprints.py`
- NEW: `tests/test_extracted_blog_blueprint_ingest.py`
- NEW: `plans/PR-Content-Ops-Blog-Blueprint-Ingestion.md`
- EDIT: `scripts/run_extracted_pipeline_checks.sh`
- EDIT: `extracted_content_pipeline/manifest.json`
- EDIT: `extracted_content_pipeline/README.md`
- EDIT: `extracted_content_pipeline/STATUS.md`
- EDIT: `extracted_content_pipeline/docs/host_install_runbook.md`
- EDIT: `extracted_content_pipeline/docs/standalone_productization.md`
- EDIT: `docs/extraction/coordination/inflight.md`

## Mechanism

The loader accepts one JSON object, a JSON array, or an object containing a
`blueprints` array. Each row must resolve to a target mode from the row or CLI
default; rows that cannot are skipped with a structured warning. The CLI reuses
the existing `EXTRACTED_DATABASE_URL` / `DATABASE_URL` convention and supports a
dry-run mode that does not open a database connection.

## Intentional

- JSON only for this first seam. CSV blueprint shape is less obvious than
  campaign opportunities and should not be guessed.
- `--account-id` is required. A blueprint write without tenant ownership is a
  dangerous host-install footgun.
- Existing generator, repository, and schema behavior are unchanged.

## Deferred

- Autonomous reasoning or intervention output providers that create blueprints.
- A hosted UI for editing blueprint rows.
- Bulk replace/delete semantics for obsolete blueprints.

## Verification

- Python compile for the new loader, CLI, and tests.
- Focused pytest for the new ingestion suite.
- Blog-adjacent pytest across blueprint storage, generation, and ingestion.
- Local PR review script.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Loader module | ~220 |
| CLI | ~130 |
| Tests | ~205 |
| Docs, manifest, coordination, and runner | ~115 |
| Total | ~670 |
