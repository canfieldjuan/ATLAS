# PR-Content-Ops-Ingestion-Contract-File-Routes

## Why this slice exists

PRs #861, #865, and #867 moved selected Content Ops ingestion files onto
server-side multipart upload routes and proved the frontend adapter route
split. The frontend contract still describes loaded JSON/JSONL/CSV files as
going through the deprecated inline `/content-ops/ingestion/*` endpoints.

This slice updates the contract docs so future UI and API work follows the
production file route instead of reintroducing large browser JSON posts.

## Scope (this PR)

Ownership lane: content-ops/ingestion-contract-docs

1. Add the file upload ingestion routes to the frontend route contract.
2. Mark inline inspect/import as deprecated compatibility for pasted/manual
   rows.
3. Update the API adapter and New Run screen guidance so selected files route
   through multipart `/ingestion/files/*` endpoints.

### Files touched

- `plans/PR-Content-Ops-Ingestion-Contract-File-Routes.md`
- `docs/frontend/content_ops_frontend_contract.md`

## Mechanism

No production code changes. The contract now describes the same split the
adapter implements:

```text
selected File -> POST /content-ops/ingestion/files/*
pasted/manual rows -> deprecated POST /content-ops/ingestion/*
```

## Intentional

- Inline ingestion is still documented because manual pasted rows remain a
  compatibility path in the current UI.
- This slice does not remove inline backend routes or UI wrappers. Removing
  them needs a compatibility-window decision.
- No generated fixtures are updated because the wire response shapes are
  unchanged.

## Deferred

- Future PR: remove the inline UI/backend compatibility path after operators no
  longer need pasted/manual row ingestion.
- Future PR: expose upload/file limits in a catalog/config endpoint if the UI
  needs to render them before upload.
- Parked hardening: none.

## Verification

- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~58 |
| Frontend contract docs | ~20 |
| **Total** | **~78** |
