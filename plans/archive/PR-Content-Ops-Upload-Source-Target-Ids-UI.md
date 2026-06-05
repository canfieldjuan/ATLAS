# PR: Content Ops Upload Source Target IDs UI

## Why this slice exists

PR #1228 proved the upload/import to run handoff by inlining normalized rows
into `inputs.source_material`. PR #1231 then landed the safer backend execution
path: once imported rows exist in `campaign_opportunities`, execute can load
selected `source_import_target_ids` by tenant scope. The UI still applies the
old inline row payload, so operators can keep carrying large source material in
the run request even though the persisted target-id contract now exists.

This slice closes that handoff without changing ingestion or execute APIs: the
New Run import result applies committed import `targetIds` into
`inputs.source_import_target_ids` and clears stale inline `source_material`.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Vertical slice

1. Replace the New Run import-result apply action so it writes imported
   `targetIds` to `inputs.source_import_target_ids`.
2. Fail closed for dry-run or empty target-id imports so the UI does not create
   a run request that references rows which were not persisted.
3. Remove stale `source_material` when target IDs are applied, so persisted
   rows are the source of truth for execute.
4. Extend the existing upload-source frontend test script, which is already
   enrolled in `.github/workflows/atlas_intel_ui_checks.yml`.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-upload-source-run-handoff.test.mjs`
- `plans/PR-Content-Ops-Upload-Source-Target-Ids-UI.md`

## Mechanism

`IngestionImportResult` already receives mapped `result.targetIds` and
`result.dryRun`. The apply button will pass both to a new helper:

```ts
updateSourceImportTargetIdsInputJson(inputsJson, targetIds)
```

The helper parses the current inputs JSON, deduplicates non-empty target IDs,
sets `source_import_target_ids`, and deletes `source_material`. The handler
refuses to apply dry-run imports before calling the helper because those target
IDs are not durable backend rows.

No backend contract changes are needed; #1231 already made
`source_import_target_ids` executable through the Atlas input provider.

## Intentional

- No backend change. The persisted-source execute path already landed in
  #1231.
- No live browser E2E in this slice. The existing source-level UI test already
  covers the import-result wiring and is enrolled in Atlas Intel UI CI.
- Keep `include_source_material` on import for now. It preserves the current
  diagnostics envelope while the apply action switches to persisted target IDs.
- Dry-run imports cannot be applied as persisted source IDs. They are useful for
  diagnostics, not execute.

## Deferred

- Future PR: stop requesting full `source_material` during import once the UI
  no longer needs it for diagnostics or fallback inspection.
- Future PR: browser E2E against a live uploaded support-ticket CSV producing
  an approved public landing/blog asset.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned;
  existing entries are dependency audit and blog content-quality issues, not
  this target-id UI handoff.

## Verification

- `cd atlas-intel-ui && npm run test:content-ops-upload-source-run-handoff`
  - 3 passed.
- `cd atlas-intel-ui && npm run test:content-ops-ingestion-routing` - 4
  passed.
- `cd atlas-intel-ui && npm run build` - passed; generated 17 sitemap URLs
  and prerendered 16 public routes.
- `bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/content-ops-upload-source-target-ids-ui-pr-body.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~90 |
| UI target-id apply helper + result wiring | ~45 |
| Frontend test updates | ~25 |
| **Total** | **~160** |
