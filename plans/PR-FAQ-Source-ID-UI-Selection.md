# FAQ Source ID UI Selection

## Why this slice exists

#1116 made saved FAQ reports selectable by ID through
`inputs.source_faq_ids`, but the New Run UI still has no control that sends
those IDs. Operators would have to hand-edit JSON, which means the by-ID path
works technically but not as a usable product path.

This slice adds the thinnest UI bridge: show recent saved FAQ reports on the
Content Ops New Run screen and write checked report IDs into the existing
inputs JSON.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Vertical slice

1. Fetch recent `faq_markdown` drafts through the existing generated-assets API
   client.
2. Show a compact FAQ source selector when `blog_post` or `landing_page` is
   selected.
3. Write selected IDs to `inputs.source_faq_ids` and remove the key when the
   selection is empty.
4. Add a focused source-level UI test so the control and request key do not
   drift.

### Files touched

- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/scripts/content-ops-faq-source-selection.test.mjs`
- `atlas-intel-ui/package.json`
- `plans/PR-FAQ-Source-ID-UI-Selection.md`

## Mechanism

`ContentOpsNewRun` uses the existing `fetchGeneratedAssetDrafts(...)` wrapper
to load recent `faq_markdown` drafts. When a blog or landing-page output is
selected, the page renders those drafts as checkboxes. Toggling a checkbox calls
the same JSON-update pattern used by the existing landing-page and FAQ controls:
parse the current inputs JSON, update one key, serialize it back, and mark the
preview/plan stale.

The request key is `source_faq_ids`, matching #1116. The selector is additive:
it does not change raw JSON editing, generation services, or review/export
screens.

## Intentional

- This does not add search or pagination. Recent reports are enough to prove
  the by-ID path without creating a larger asset-picker component.
- This does not auto-select the FAQ report created by the current run. It only
  selects already-saved reports returned by the assets API.
- This keeps selected IDs inside the raw inputs JSON so preview, plan, execute,
  and the normalized request panel all show the same request shape.

## Deferred

- Future PR: richer saved-FAQ picker with search/status filters if operators
  need more than the recent list.
- Future PR: execute-route smoke that selects a persisted FAQ by ID against a
  real Postgres fixture.
- Parked hardening: none.

## Verification

Ran locally:

- Command: node --test atlas-intel-ui/scripts/content-ops-faq-source-selection.test.mjs
  - 3 passed
- Command: npm --prefix atlas-intel-ui run build
  - passed
- Command: npm --prefix atlas-intel-ui run lint
  - passed
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-source-id-ui-selection.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| New Run selector | ~220 |
| UI source test | ~30 |
| package script | ~5 |
| Plan doc | ~85 |
| **Total** | **~340** |
