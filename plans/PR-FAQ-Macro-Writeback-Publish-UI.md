# PR-FAQ-Macro-Writeback-Publish-UI

## Why this slice exists

The FAQ macro writeback lane now has tenant credentials, a publish service, and
a scoped backend route:
`POST /content-assets/faq_markdown/drafts/{draft_id}/publish-macros`. Operators
can approve generated FAQ Markdown drafts in the review queue, but they still
cannot trigger the Zendesk macro publish flow from the product UI. This slice
adds the thinnest UI path from an approved FAQ draft to the existing publish
route and surfaces the route summary without changing the backend contract.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Vertical slice

1. Add a typed frontend API wrapper for the FAQ macro publish route.
2. Add a FAQ-only publish action to Generated Asset Review rows.
3. Gate the button to approved FAQ drafts with a clear disabled hint for other
   FAQ statuses.
4. Surface success, skipped, failed, and pending-reconcile summary counts inline
   after the publish call and refresh the review list so a clean publish can
   show the updated draft status.
5. Add a focused Node test for the wrapper route and source-level UI wiring.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-UI.md` -- plan for this slice.
- `atlas-intel-ui/package.json` -- adds the focused test script.
- `atlas-intel-ui/scripts/content-ops-faq-macro-publish-ui.test.mjs` -- API wrapper and UI wiring checks.
- `atlas-intel-ui/src/api/contentOps.ts` -- typed publish summary and wrapper.
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` -- FAQ row publish action and summary rendering.

## Mechanism

The frontend API adapter adds `publishGeneratedFaqMacros(id)`, which posts an
empty JSON body to
`/api/v1/content-assets/faq_markdown/drafts/{id}/publish-macros` using the same
auth, fallback, and refresh plumbing as generated asset review actions.

`ContentOpsAssetsReview` keeps one publish result state keyed by draft id. FAQ
rows render a `Publish macros` button only on the FAQ Markdown tab. The button
is enabled for approved drafts with an id, disabled for drafts that still need
approval, and shows the existing busy spinner while the publish route is in
flight. The returned summary is rendered inline as a compact outcome banner so
operators can see whether macros published, updated, skipped, failed, or need
pending reconciliation.

## Intentional

- No backend changes in this slice. The route already exists and owns tenant
  scope, approval gating, provider wiring, and publish semantics.
- No live Zendesk preview or credential picker in the review row. Credential
  management landed separately; the publish route uses the tenant/provider
  selected by backend wiring.
- No broad drawer redesign. This is the narrow product trigger, not a full macro
  operations console.

## Deferred

- `PR-FAQ-Macro-Writeback-Pending-Reconcile`: add the backend reconcile command
  for pending Zendesk mapping recovery.
- `PR-FAQ-Macro-Writeback-Publish-History`: persist and display prior publish
  attempts beyond the latest route response if operators need an audit trail.

Parked hardening: none

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-faq-macro-publish -- 2 passed.
- npm --prefix atlas-intel-ui run build -- passed.
- npm --prefix atlas-intel-ui run lint -- passed.
- git diff --check -- passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-ui.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~82 |
| API wrapper | ~43 |
| Review UI | ~119 |
| Focused test | ~121 |
| Total | ~366 |
