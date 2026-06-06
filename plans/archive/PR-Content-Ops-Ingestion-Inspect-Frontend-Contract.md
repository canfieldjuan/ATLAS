# Content Ops Ingestion Inspect Frontend Contract

## Why this slice exists

PR #579 added a hosted `POST /content-ops/ingestion/inspect` route. The Atlas
Intel frontend API adapter still has no typed wrapper or fixture pin for that
route, so a future UI would either use raw fetch calls or invent its own shape.

## Scope (this PR)

1. Add wire types and a typed fetch wrapper for ingestion inspection.
2. Add domain types and a wire-to-domain mapper for the diagnostics response.
3. Add a canonical fixture and compile-time contract checks.
4. Refresh the frontend contract doc and coordination state.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/api/contentOps.contract.ts`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/ingestion-inspect.json`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/contract.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Inspect-Frontend-Contract.md`

## Mechanism

The API adapter posts a bounded `ContentOpsIngestionInspectRequest` to
`/ingestion/inspect`. The domain mapper preserves raw sample/warning rows as
records while camel-casing the stable summary fields.

## Intentional

- No new React UI.
- No file upload handling.
- No backend changes.
- No changes to existing preview/plan/execute request types.

## Deferred

- Operator UI for pasting/uploading rows.
- Browser-side CSV parsing.
- Runtime tests; current frontend gate is compile-time contract checking.

## Verification

- Run the frontend type/build gate.
- Run local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| API wire types/wrapper | ~60 |
| Domain mapper/types | ~80 |
| Fixture/contracts | ~55 |
| Docs/coordination/plan | ~90 |
| **Total** | ~285 |
