# PR-Content-Ops-Review-Source-Quote-Grade-Prefilter

## Why this slice exists

After #599, Capterra passed the review-source Postgres smoke, but TrustRadius
did not. The source summary reported 31 quote-grade TrustRadius rows, while
the source-row exporter returned 0 rows. The mismatch is in the exporter:
`fetch_review_source_rows()` reads a small urgency/date window and only then
applies quote-grade phrase filtering in Python, so eligible lower-ranked rows
can be missed.

## Scope (this PR)

- Add the same quote-grade phrase predicate used by source summary to the
  row-export SQL query.
- Keep the existing Python quote-grade conversion as a defensive second gate.
- Update row-query tests for vendor-scoped and source-wide exports.
- Record the Capterra pass and TrustRadius gap in status/backlog docs.

### Files touched

- `scripts/export_content_ops_review_sources.py`
- `tests/test_export_content_ops_review_sources.py`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-Review-Source-Quote-Grade-Prefilter.md`

## Mechanism

`build_review_source_query()` now accepts `allowed_polarities` and
`allowed_fields`, appends those as SQL parameters, and adds an `EXISTS` clause
over `enrichment->'phrase_metadata'` requiring subject-vendor, verbatim,
allowed polarity, and allowed field. `fetch_review_source_rows()` passes
through its existing phrase filter arguments.

## Intentional

- No change to summary semantics.
- No change to generated draft behavior.
- No removal of the Python quote-grade gate.
- TrustRadius closeout is included because the fixed exporter was rerun live.

## Deferred

- Broader review-source ranking changes beyond quote-grade eligibility.
- Trustpilot data re-enrichment.

## Verification

- Focused review-source exporter/Postgres smoke tests -> `21 passed`.
- Python compile check for exporter script/tests -> passed.
- Live Capterra Postgres smoke -> passed.
- Live TrustRadius Postgres smoke -> passed after the SQL prefilter.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Exporter query | 35 |
| Tests | 45 |
| Docs/status | 25 |
| Coordination and plan | 65 |
| Total | 170 |
