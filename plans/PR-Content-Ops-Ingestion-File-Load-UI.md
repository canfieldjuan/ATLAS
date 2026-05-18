# PR: Content Ops Ingestion File Load UI

## Why this slice exists

The New Run ingestion panel can inspect and import pasted rows, but hosts still
need to manually paste JSON. A lightweight browser file loader removes that
friction for JSON and JSONL opportunity/source exports while reusing the
existing inspect/import path.

## Scope (this PR)

Add JSON and JSONL file loading to the Content Ops New Run ingestion panel.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `docs/frontend/content_ops_frontend_contract.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Ingestion-File-Load-UI.md`

## Mechanism

- Add a file input button beside Inspect/Import.
- Read the selected file in-browser and normalize JSON arrays, JSON row
  objects, wrapped row payloads, or JSONL rows into the existing textarea.
- Set the source label to the loaded filename and reuse the current
  inspect/import actions.

## Intentional

- No backend API changes.
- No CSV parser in this slice.
- No drag-and-drop UI.
- No changes to import or inspect semantics.

## Deferred

- Browser CSV import.
- Drag-and-drop file loading.
- Versioned source-bundle schema support in the browser.

## Verification

- npm --prefix atlas-intel-ui ci
- npm --prefix atlas-intel-ui run build
- git diff --check
- bash scripts/local_pr_review.sh

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| New Run file loader UI and parsing | ~105 |
| Docs/coordination/plan | ~70 |
| **Total** | ~175 |
