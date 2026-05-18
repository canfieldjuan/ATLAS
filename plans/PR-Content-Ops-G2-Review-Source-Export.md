# Content Ops G2 Review Source Export

## Why this slice exists

The hosted ingestion path can now load source rows, and the Atlas database has
clean canonical G2 review rows. We need a read-only bridge that exports one
reliable review source into Content Ops source-row JSONL without mixing noisy
community sources or contradictory evidence lanes.

This slice is above the preferred 400 LOC budget because the script and its
contract tests are one boundary: the reviewer needs to see the SQL filter,
phrase-lane gate, JSONL shape, and ingestion command together to validate that
we are not mutating Atlas data or feeding contradictory review text.

## Scope (this PR)

Add a host-facing script that reads canonical enriched `b2b_reviews` rows and
emits Content Ops source rows, defaulting to G2.

### Files touched

- `scripts/export_content_ops_review_sources.py`
- `tests/test_export_content_ops_review_sources.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-G2-Review-Source-Export.md`

## Mechanism

- Query only canonical review rows: `duplicate_of_review_id IS NULL`,
  enriched rows, non-empty review text, and a configurable source.
- Default source is `g2`.
- Filter rows to quote-grade subject-vendor phrase metadata with negative or
  mixed polarity.
- Export the coherent phrase lane as source-row `text`, while preserving the
  full review and provenance fields for audit.
- Render JSONL so the existing `--source-rows --source-format jsonl` ingestion
  path can consume the file directly.

## Intentional

- Read-only against Atlas Postgres.
- No mutation of `b2b_reviews` duplicate markers.
- No Reddit/community source mixing.
- No Content Ops runtime changes.

## Deferred

- Trustpilot and TrustRadius exports can reuse this script by passing
  `--source`, but the first verified run stays on G2.
- Importing the exported rows into a live Content Ops database is a separate
  operator step using the existing ingestion/import CLI.

## Verification

- Focused pytest for the exporter, source adapters, and ingestion diagnostics -
  63 passed.
- Py-compile for the exporter script - passed.
- ASCII Python check - passed.
- Live read-only G2 export for Slack - exported 5 rows.
- Ingestion inspection of the exported JSONL - ok, 5 opportunities.
- Dry-run source-row import of the exported JSONL - would insert 5 rows.
- Offline campaign generation from the exported JSONL - generated 2 drafts.
- Whitespace diff check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Export script | ~395 |
| Tests | ~220 |
| Pipeline check wiring | ~5 |
| README/runbook docs | ~45 |
| Coordination and plan docs | ~70 |
| **Total** | ~735 |
