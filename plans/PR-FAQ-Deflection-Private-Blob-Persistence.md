# PR-FAQ-Deflection-Private-Blob-Persistence

## Why this slice exists

PR-FAQ-Deflection-Portfolio-Live-Submit proved the hosted browser-to-ATLAS
handoff by forwarding the selected CSV through the portfolio server, but it
explicitly deferred private Vercel Blob persistence. That direct multipart path
keeps uploads under the Vercel Functions request body cap and does not preserve
the selected CSV before ATLAS submission.

This slice adds the durable private-Blob handoff that #1188 deferred: the
browser uploads the CSV to Vercel Blob with private access, then the portfolio
server reads that private object with its server token and forwards multipart
bytes to ATLAS. Rich progress/retry UX remains the next product slice.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Vertical slice

1. Add `@vercel/blob` to `portfolio-ui` and create a server upload-token route
   for private FAQ deflection CSV uploads.
2. Change the upload page to persist the selected CSV to private Blob before
   report submission.
3. Extend the portfolio submit endpoint to accept a server-only JSON Blob
   reference, read the private object, reconstruct the ATLAS multipart submit,
   and project the existing result payload.
4. Keep the service JWT, Blob read/write token, and ATLAS execute envelope out
   of browser responses.
5. Add focused tests for token-route validation, browser wiring, private Blob
   submit forwarding, and fail-closed config/error branches.

### Files touched

- `plans/PR-FAQ-Deflection-Private-Blob-Persistence.md`
- `portfolio-ui/package.json`
- `portfolio-ui/package-lock.json`
- `portfolio-ui/api/content-ops/deflection/upload.js`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The browser calls `upload(file.name, file, { access: "private",
handleUploadUrl: "/api/content-ops/deflection/upload" })` from
`@vercel/blob/client`. The upload token route uses `handleUpload(...)` and
allows only CSV-shaped content for a deflection-specific pathname prefix.

After Blob upload completes, the page posts JSON to the existing submit route:

```json
{
  "blob_url": "https://...",
  "blob_pathname": "faq-deflection/uploads/...",
  "support_platform": "zendesk",
  "company_name": "Acme Co.",
  "contact_email": "lead@example.com",
  "limit": 1000
}
```

The submit route keeps the existing raw multipart mode for local rollback, but
the new JSON mode validates the configured account binding, reads the private
Blob using the server `BLOB_READ_WRITE_TOKEN`, reconstructs a `FormData` body
with `csv_file` and the same metadata fields, and forwards that body to ATLAS
with the service JWT. The browser still receives only `{ ok, request_id,
account_id, result_path }`. The server routes lazy-import the Blob SDK only on
real Blob operations so the lean portfolio Node checks can import the modules
without an install step.

## Intentional

- This slice uses Vercel's private Blob client-upload flow rather than a
  server-side `put(...)`, because the direct browser-to-Blob transfer avoids the
  Vercel Functions request body cap that forced the temporary 4 MB portfolio
  limit.
- The existing multipart submit path remains as a rollback/local-test path.
  Private Blob JSON submit is the browser path.
- The Blob URL is treated as an opaque private object reference. It is not
  rendered as a download link and cannot expose content without the server
  token.
- Blob SDK imports are lazy server-side imports. The tested helper paths use
  injected fakes so dependency-light CI can collect the modules.
- Rich progress bars, resumable upload retry controls, and abandoned-upload
  cleanup are deferred to the next slice.

## Deferred

- Upload progress/retry UX is the next slice in this lane.
- Operational cleanup for orphaned private blobs after failed ATLAS submission
  remains deferred until the live persistence path has production observations.
- Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` - passed, 13 checks.
- `npm run test:deflection-result --prefix portfolio-ui` - passed, 11 checks.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` - passed, 14 checks.
- `npm run build --prefix portfolio-ui` - passed; sitemap generation skipped because
  `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` is not configured.
- Temporarily hid `portfolio-ui/node_modules`, then imported
  `submit.js` and `upload.js` with Node - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-private-blob-persistence.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 120 |
| Blob upload route | 118 |
| Submit endpoint private Blob path | 164 |
| Upload page wiring | 73 |
| Tests | 145 |
| Package metadata | 171 |
| **Total** | **791** |

This slice is above the soft cap because the token route, browser upload handoff,
server-side private Blob read, and contract tests need to land together for an
end-to-end private persistence path.
