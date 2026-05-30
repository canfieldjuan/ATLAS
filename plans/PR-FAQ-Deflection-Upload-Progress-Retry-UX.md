# PR-FAQ-Deflection-Upload-Progress-Retry-UX

## Why this slice exists

PR-FAQ-Deflection-Private-Blob-Persistence moved the FAQ deflection browser
upload to private Vercel Blob and explicitly deferred richer upload progress and
retry UX. The current page only flips the submit button between "Uploading CSV"
and "Creating report"; if the Blob upload or ATLAS submit fails, the buyer sees
an error but no explicit retry affordance or progress context.

This slice adds the narrow UX layer on top of the now-merged private Blob
contract: visible upload progress during Blob transfer and a retry state that
lets the buyer resubmit the same selected CSV after a transient failure.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Surface `@vercel/blob/client` upload progress from `onUploadProgress` in the
   FAQ deflection upload page.
2. Render an accessible progress bar and phase text while the CSV is uploading.
3. Change the failed submit state to an explicit retry affordance while keeping
   the selected CSV and form values intact.
4. Keep private Blob persistence, account binding, and server submit contracts
   unchanged.
5. Extend the focused upload-shell source tests to pin the progress/retry UX
   markers and ensure credentials still stay out of browser code.

### Files touched

- `plans/PR-FAQ-Deflection-Upload-Progress-Retry-UX.md`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The upload page already calls:

```ts
uploadBlob(pathname, file, { access: "private", ... })
```

This slice adds an `onUploadProgress` callback that updates
`SubmitState.status === "uploading"` with a bounded percentage. While that state
is active, the form renders a `role="progressbar"` element with stable data
markers for the hosted smoke/source tests.

If either the Blob upload or the subsequent server submit fails, the state moves
to `{ status: "error", message }`. The main submit button remains enabled when
the selected CSV and required fields are still valid, changes label to
`Retry upload`, and calls the same `startSubmit` path. No partial Blob reference
is trusted or reused from a failed attempt.

## Intentional

- Retry is manual and re-runs the full private Blob upload + server submit. The
  page does not cache a prior Blob pathname because failed submits can leave
  ambiguous server state.
- This slice does not add background polling, resumable multipart controls, or
  abandoned-blob cleanup.
- Progress is browser-only UI state. The server token, service JWT, and ATLAS
  envelope boundaries remain unchanged from #1195.

## Deferred

- Abandoned private Blob cleanup remains deferred until production observations
  show whether failed upload attempts create meaningful storage churn.
- Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` - passed, 13 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 12 checks.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 14 checks.
- `npm run build --prefix portfolio-ui` - passed after hydrating this fresh
  worktree with `npm install --prefix portfolio-ui`; sitemap generation skipped
  because `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` is not configured.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-upload-progress-retry-ux.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Upload page progress/retry UI | 50 |
| Focused tests | 8 |
| **Total** | **146** |
