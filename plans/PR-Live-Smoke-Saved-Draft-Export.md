# PR: Live Smoke Saved Draft Export

## Why this slice exists

The support-ticket live smoke now saves real landing-page and blog-post drafts,
and the export paths can fetch exact draft ids. The operator still has to copy
the saved id into a second command to inspect the generated artifact. For live
validation, the smoke should optionally write that exact saved draft export in
the same run.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add an optional saved-draft export flag to the live Content Ops smoke.
2. Export the exact saved landing-page or blog-post draft ids before closing the
   DB pool.
3. Include the export payload in the smoke result and write it to disk when the
   flag is supplied.
4. Add focused tests for exact saved-id export and no-saved-id failure behavior.

### Files touched

- `plans/PR-Live-Smoke-Saved-Draft-Export.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `extracted_content_pipeline/README.md`

## Mechanism

The smoke extracts saved ids from the executed output step. When
`--export-saved-draft` is present, it calls the same generated-asset export
helpers used by the CLI/API with `status=None`, exact ids, and the current
tenant scope. The export happens before the smoke closes the database so the
host pool is still available.

## Intentional

- Optional operator artifact only; no hosted route, generation, FAQ, or
  file-ingestion behavior changes.
- Exact export remains tenant-scoped by the existing repositories.
- The smoke fails loudly if export is requested but generation returns no saved
  ids.

## Deferred

- No HTML rendering in this slice. JSON export is the thinnest artifact that
  preserves readiness, metadata, and source-derived fields for review.
- Parked hardening: none. `HARDENING.md` was scanned; current entries are FAQ
  scale and file-ingestion concurrency work outside this support-ticket provider
  validation slice.

## Verification

- Focused smoke tests:
  `pytest tests/test_smoke_content_ops_live_generation.py -q`
  - 22 passed.
- Py compile for the smoke script and test file - passed.
- Git whitespace check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~70 |
| Script | ~90 |
| Tests | ~270 |
| Docs | ~10 |
| **Total** | **~440** |
