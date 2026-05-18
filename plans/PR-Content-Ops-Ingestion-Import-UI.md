# PR: Content Ops Ingestion Import UI

## Why this slice exists

Hosted Content Ops now exposes an ingestion import API, but the New Run screen
can only inspect pasted opportunity/source rows. This slice closes the browser
loop so operators can dry-run or import accepted rows from the same panel.

## Scope (this PR)

Wire the existing import route into the frontend API/domain contract and add a
small import action to the New Run ingestion panel.

### Files touched

- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/api/contentOps.contract.ts`
- `atlas-intel-ui/src/api/__fixtures__/contentOps/ingestion-import.json`
- `atlas-intel-ui/src/domain/contentOps/types.ts`
- `atlas-intel-ui/src/domain/contentOps/fromWire.ts`
- `atlas-intel-ui/src/domain/contentOps/contract.ts`
- `atlas-intel-ui/src/domain/contentOps/index.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Import-UI.md`

## Mechanism

- Add wire/domain request and response types for
  `POST /content-ops/ingestion/import`.
- Add mapper coverage and a fixture-backed contract pin.
- Reuse the same pasted row parser and target-mode/source controls from the
  inspect panel.
- Add dry-run and replace-existing toggles, defaulting to dry-run for safety.
- Render import counts, target ids, and warnings after a successful import.

## Intentional

- No browser CSV upload in this PR.
- No import history or audit table UI.
- No backend API or database changes.
- No generated-asset execution changes.

## Deferred

- File upload import flow.
- Import history/audit browser.
- Rich 400 diagnostics rendering for failed imports; the current route still
  surfaces a safe error message through the shared API wrapper.

## Verification

- Run the atlas-intel-ui build.
- Run git diff --check.
- Run the local PR review script from the repo root.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Frontend API/domain contract | ~100 |
| New Run import UI | ~190 |
| Docs/coordination/fixture/plan | ~95 |
| **Total** | ~385 |
