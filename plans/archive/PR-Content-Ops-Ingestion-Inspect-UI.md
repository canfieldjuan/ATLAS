# PR: Content Ops Ingestion Inspect UI

## Why this slice exists

PR #579 added the hosted ingestion inspect route and PR #580 added the
frontend API/domain contract. Operators still cannot call the route from the
Content Ops console. This slice adds a small inspector panel to the existing
new-run screen so pasted customer rows can be checked before preview, plan, or
execution.

## Scope (this PR)

Add a read-only ingestion diagnostics panel to the hosted Content Ops new-run
screen and refresh the coordination/frontend contract docs for that UI surface.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-Inspect-UI.md`

## Mechanism

- Import the existing `inspectContentOpsIngestion` API wrapper and domain
  mapper.
- Add local UI state for pasted rows, source label, source-row mode, and
  diagnostics.
- Parse a pasted row object, row array, or wrapper object with `rows`,
  `opportunities`, or `source_rows`.
- Render readiness, counts, warnings, and sampled normalized rows from
  `ContentOpsIngestionDiagnostics`.

## Intentional

- No backend route changes.
- No generation, preview, plan, or execute behavior changes.
- No file upload UI yet; pasted JSON is enough for the first hosted operator
  seam.

## Deferred

- File upload and CSV parsing in the browser.
- Component tests once the frontend test harness exists.
- Persisting inspected rows into a run.

## Verification

- `npm --prefix atlas-intel-ui run build`
- `git diff --check`
- Run `scripts/local_pr_review.sh` from the repo root.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| New-run ingestion panel and helpers | ~250 |
| Frontend contract docs | ~5 |
| Coordination updates | ~4 |
| Plan doc | ~55 |
| **Total** | ~314 |
