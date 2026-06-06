# PR-Content-Ops-Inline-Row-Label

## Why this slice exists

The Content Ops new-run page now supports real file uploads and keeps pasted
inline JSON rows only as a deprecated compatibility path. The upload limits
summary says inline JSON is deprecated, and #886 preflights the inline row cap,
but the actual inline rows textarea is unlabeled.

That makes the operator surface ambiguous: a selected file takes precedence, but
the old textarea still looks like an equal primary input. This slice labels the
inline field clearly and shows when file upload is the active source.

## Scope (this PR)

Ownership lane: content-ops/inline-row-label

1. Add a visible label for the inline rows JSON textarea.
2. Mark the inline field as deprecated in the field label.
3. Show that inline rows are ignored while an uploaded file is selected.
4. Keep inspect/import behavior unchanged.

### Files touched

- `plans/PR-Content-Ops-Inline-Row-Label.md`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`

## Mechanism

The UI wraps the existing inline rows textarea in a labeled block and renders a
small state note when `selectedIngestionFile` is present. The textarea remains
editable because editing it intentionally clears the selected file and returns
to the deprecated inline path.

## Intentional

- No API changes.
- No new validation; row-count preflight already landed in #886.
- No removal of inline compatibility in this slice.

## Deferred

- Future PR: remove inline compatibility after the operator compatibility
  window.
- Future PR: add richer upload/job status after the backend grows a job model.
- Parked hardening: none.

## Verification

- Passed: UI build: `npm run build`.
- Passed: UI lint: `npm run lint`.
- Passed: `git diff --check`.
- Passed: local PR review via `scripts/local_pr_review.sh`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~70 |
| UI label/state copy | ~20 |
| **Total** | **~90** |
