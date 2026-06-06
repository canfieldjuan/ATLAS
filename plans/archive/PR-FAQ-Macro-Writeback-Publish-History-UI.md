# PR-FAQ-Macro-Writeback-Publish-History-UI

## Why this slice exists

PR-FAQ-Macro-Writeback-Publish-History-Read-Route added a tenant-scoped API
route for recent FAQ macro publish attempts, but operators still cannot see
that audit trail in the Generated Asset Review drawer. The publish button now
returns a one-request summary, while a refreshed page has no visible record of
prior skipped, failed, pending-reconcile, or successful attempts.

## Scope (this PR)

Ownership lane: content-ops/faq-macro-writeback
Slice phase: Product polish

1. Add typed frontend DTOs and a wrapper for the FAQ macro publish-attempts
   read route.
2. Fetch recent publish attempts when an FAQ Markdown draft detail drawer opens.
3. Render the attempt history in the drawer with status, counts, skipped items,
   provider result status, and timestamps.
4. Add focused Node coverage for the encoded route and source-level drawer
   wiring.

### Files touched

- `plans/PR-FAQ-Macro-Writeback-Publish-History-UI.md` -- slice plan.
- `atlas-intel-ui/src/api/contentOps.ts` -- publish-attempt DTOs and wrapper.
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` -- FAQ drawer history
  fetch/render state.
- `atlas-intel-ui/scripts/content-ops-faq-macro-publish-ui.test.mjs` --
  focused route and source wiring checks.

## Mechanism

The frontend API adapter adds:

```ts
fetchGeneratedFaqMacroPublishAttempts(id, { limit })
```

which calls:

```text
GET /api/v1/content-assets/faq_markdown/drafts/{id}/publish-macro-attempts?limit=5
```

using the existing authenticated generated-asset fetch plumbing. The detail
drawer receives a `publishHistory` state object from the page. When the drawer
opens for a FAQ Markdown draft with an id, the page fetches recent attempts and
passes loading/error/data state into the drawer. The drawer renders the history
only for FAQ drafts, so other generated asset types keep their existing detail
surface.

## Intentional

- No backend changes. The route already enforces tenant scope, FAQ-only access,
  draft existence, and display-safe DTOs.
- No polling or automatic refresh loop. Operators can close/reopen or use the
  existing page refresh after a publish.
- The drawer caps history at five attempts to keep the review surface compact.

## Deferred

- PR-FAQ-Macro-Writeback-Live-Smoke remains deferred until safe Zendesk sandbox
  credentials and test macro data are available.
- PR-FAQ-Macro-Writeback-Pending-Reconcile remains the future backend recovery
  command for pending Zendesk mapping repairs.
- Parked hardening: none.

## Verification

- `npm --prefix atlas-intel-ui run test:content-ops-faq-macro-publish` -- 3 passed.
- `npm --prefix atlas-intel-ui run build` -- passed.
- `npm --prefix atlas-intel-ui run lint` -- passed.
- `git diff --check` -- passed.
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Macro-Writeback-Publish-History-UI.md` -- passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Macro-Writeback-Publish-History-UI.md` -- passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-macro-writeback-publish-history-ui.md` -- passed.

## Estimated diff size

| Area | Estimate |
|---|---:|
| Plan | ~70 |
| API wrapper | ~45 |
| Review drawer UI | ~145 |
| Focused test | ~45 |
| Total | ~305 |
