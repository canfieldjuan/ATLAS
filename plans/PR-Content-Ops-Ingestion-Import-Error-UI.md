# PR: Content Ops Ingestion Import Diagnostics UI

## Why this slice exists

The hosted import route returns structured diagnostics when pasted rows are not
ready to import, but the frontend import action currently collapses that 400
response into a generic API error. Operators should see the same row warnings
they would see from the inspect action.

## Scope (this PR)

Preserve `ingestion_not_ready` diagnostics through the frontend API wrapper and
render them in the New Run import result panel.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Import-Error-UI.md`

## Mechanism

- Change the import API wrapper to return a discriminated outcome, matching the
  existing execute wrapper pattern for meaningful non-2xx responses.
- Decode 400 responses whose detail has `reason="ingestion_not_ready"` and a
  diagnostics payload.
- Render that diagnostics payload as an import-blocked state and keep the
  inspect panel synchronized with the same diagnostics.

## Intentional

- No backend API changes.
- No changes to successful import response mapping.
- No file upload or import history UI.
- No retry behavior for database or validation failures.

## Deferred

- Richer generic validation-error rendering for 422 responses.
- Browser file upload import flow.
- Import history/audit browser.

## Verification

- Run the atlas-intel-ui build.
- Run git diff --check.
- Run the local PR review script from the repo root.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Import API outcome handling | ~83 |
| New Run blocked-import UI | ~69 |
| Docs/coordination/plan | ~73 |
| **Total** | ~225 |
