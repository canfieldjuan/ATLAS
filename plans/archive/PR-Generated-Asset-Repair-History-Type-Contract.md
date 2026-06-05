# PR-Generated-Asset-Repair-History-Type-Contract

## Why this slice exists

PR #736 surfaced landing-page repair history in the generated asset review
drawer. Reviewer feedback noted that the drawer reads
`generation_quality_repair_history` and `quality_repair_history` through the
`GeneratedAssetDraft` catch-all index signature, so a field-name typo would
compile and fail closed to an empty panel.

This slice makes the repair-history telemetry keys explicit on the generated
asset API type.

## Scope (this PR)

1. Declare the generated asset repair-history entry shape used by the review
   drawer.
2. Declare the top-level and metadata fallback repair-history fields on
   `GeneratedAssetDraft`.
3. Keep the existing drawer parser and backend payloads unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Generated-Asset-Repair-History-Type-Contract.md` | Plan doc for this review-follow-up slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Adds explicit generated asset repair-history telemetry fields. |

## Mechanism

`GeneratedAssetDraft` gains two explicit optional telemetry fields:

```ts
generation_quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
quality_repair_history?: GeneratedAssetRepairHistoryEntry[] | string
```

The same keys are declared on a `GeneratedAssetDraftMetadata` shape because the
review drawer also accepts metadata-nested values. The fields continue to allow
either arrays or JSON-stringified payloads, matching the fail-soft parser shipped
in #736.

## Intentional

- No drawer rendering changes; this is a contract tightening follow-up to #736.
- No backend migration or payload rename; the UI type now names the fields the
  backend already emits.
- The draft row catch-all index signature stays because generated assets still
  carry asset-specific fields beyond this telemetry.

## Deferred

- Queue-level repair-history filters and export columns remain deferred to a
  future generated-asset triage slice.
- A broader generated-asset response-schema contract can follow if more
  review-drawer fields become cross-layer drift risks.

## Verification

- `npm run lint` from `atlas-intel-ui`.
- `npm run build` from `atlas-intel-ui`.
- `git diff --check`.
- `bash scripts/local_pr_review.sh origin/main`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| API type | ~15 |
| Total | ~75 |
