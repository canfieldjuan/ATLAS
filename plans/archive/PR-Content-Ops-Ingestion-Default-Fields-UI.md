# PR-Content-Ops-Ingestion-Default-Fields-UI

## Why this slice exists

The hosted Content Ops ingestion API already supports `default_fields` for source-row imports, but the Atlas Intel New Run screen cannot send them. Operators loading real review, ticket, or transcript exports still have to edit source files to bind fallback account, vendor, or contact metadata. This slice exposes the existing backend contract in the UI without changing ingestion semantics.

## Scope (this PR)

1. Add `defaultFields` to the Atlas Intel Content Ops ingestion domain request types.
2. Serialize that field as backend `default_fields` for inspect and import requests.
3. Add a small default-fields JSON input to the New Run ingestion panel.
4. Parse and validate the JSON input before inspect/import.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Ingestion-Default-Fields-UI.md`

## Mechanism

`ContentOpsNewRun` keeps a JSON draft string for fallback fields. Inspect and import parse that draft into an object and pass it through `toWireIngestionInspectRequest`, which now emits `default_fields`. Empty input stays `{}`. Invalid JSON or non-object values fail client-side before a request is sent.

## Intentional

- No backend changes: the FastAPI route and source adapter already support this contract.
- No CSV-specific UI path: file loading still normalizes JSON/JSONL/CSV rows into the same rows textarea.
- No provider-specific presets in this slice. Operators can paste the fallback fields they need.

## Deferred

- A richer key/value editor can replace the JSON input if repeated operator usage justifies it.
- Source-specific saved presets remain separate from this generic fallback field surface.

## Verification

- `npm ci` in `atlas-intel-ui` - passed; npm reported 6 existing audit findings.
- `npm run build` in `atlas-intel-ui` - passed.
- `git diff --check` - passed.
- Local PR review - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| UI domain/wire types | ~15 |
| New Run panel parsing + input | ~80 |
| Plan + coordination | ~60 |
| **Total** | **~155** |

This is below the 400 LOC review budget.
