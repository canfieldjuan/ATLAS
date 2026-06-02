# PR: Content Ops Upload Source Lean Import

## Why this slice exists

PR #1232 switched the New Run apply action from inline `source_material` to
persisted `source_import_target_ids`. That made imported target IDs the source
of truth for execute, but the UI still asks the import endpoint to include the
full normalized `source_material` payload. That payload is now redundant in the
run handoff and can be large for real CSV uploads.

This slice removes that redundant request from the New Run import path while
leaving the lower-level API opt-in available for diagnostics and future tools.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Product polish

1. Stop requesting full `source_material` from New Run import calls for both
   multipart file imports and inline compatibility imports.
2. Keep inspect calls unchanged; inspect still uses compact diagnostics and
   never requests full source material from this UI.
3. Keep the API/domain `includeSourceMaterial` opt-in intact for non-New-Run
   callers.
4. Extend the enrolled upload-source frontend test so the UI contract fails if
   New Run starts requesting full material again.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs`
- `plans/PR-Content-Ops-Upload-Source-Lean-Import.md`

## Mechanism

`handleImportIngestion` will pass `include_source_material: false` for file
imports and `includeSourceMaterial: false` for inline imports. The success
response still carries `import.target_ids`, and #1232 already made
`IngestionImportResult` apply those IDs to `inputs.source_import_target_ids`.

The API/domain test that proves callers can opt into full source material stays
unchanged. The New Run source test changes from requiring the old `true`
request to rejecting it in `ContentOpsNewRun.tsx`.

## Intentional

- No backend change. The import endpoint still supports full source material
  when an explicit caller needs it.
- No UI fallback to inline material. The persisted target-id path is the safer
  execute contract, and dry-run imports already fail closed.
- No browser E2E in this slice; this is a request-shape cleanup covered by the
  enrolled source-level UI test and build.

## Deferred

- Future PR: browser E2E against a live uploaded support-ticket CSV producing
  an approved public landing/blog asset.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing entries are dependency audit and blog content-quality issues, not
  this lean import request.

## Verification

- `cd atlas-intel-ui && npm run test:content-ops-upload-source-run-handoff`
  - 3 passed.
- `cd atlas-intel-ui && npm run test:content-ops-ingestion-routing` - 4
  passed.
- `cd atlas-intel-ui && npm run build` - passed; generated 17 sitemap URLs
  and prerendered 16 public routes.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/content-ops-upload-source-lean-import-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~80 |
| UI import request flags | ~2 |
| Frontend test updates | ~10 |
| **Total** | **~92** |
