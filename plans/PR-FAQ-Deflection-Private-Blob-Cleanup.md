# PR-FAQ-Deflection-Private-Blob-Cleanup

## Why this slice exists

PR-FAQ-Deflection-Private-Blob-Persistence moved large FAQ deflection CSV uploads
through private Vercel Blob, and PR-FAQ-Deflection-Upload-Progress-Retry-UX
made failed uploads retryable from the browser. The remaining production
hardening gap is server-side cleanup: after the portfolio API has consumed a
private Blob and attempted the ATLAS submit, the Blob object should be deleted
best-effort so successful and failed submit attempts do not leave stale private
CSV objects behind.

This slice is intentionally small because the browser upload, retry UX, and
ATLAS forwarding contracts are already merged.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Production hardening

1. Add a lazy private Blob delete helper to the FAQ deflection submit API.
2. Delete the validated private Blob pathname after the server has read the
   Blob and attempted the ATLAS submit.
3. Keep cleanup best-effort: delete failures must not change the already
   produced ATLAS submit result.
4. Extend the upload shell tests to prove cleanup on success, cleanup on ATLAS
   failure, and cleanup failure masking.

### Files touched

- `plans/PR-FAQ-Deflection-Private-Blob-Cleanup.md`
- `portfolio-ui/api/content-ops/deflection/submit.js`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

`readPrivateCsvBlob()` returns the validated Blob pathname along with the CSV
payload. `submitPrivateBlob()` forwards the CSV to ATLAS exactly as before,
then calls `cleanupPrivateCsvBlob()` with the validated pathname and Blob token:

```js
const result = await forwardSubmit(...);
await cleanupPrivateCsvBlob({ pathname: blob.pathname, token: blobToken, deleteBlobImpl });
return result;
```

`cleanupPrivateCsvBlob()` validates the pathname again, lazy-imports
`@vercel/blob`'s `del`, catches any delete error, and returns a cleanup status
only for tests. The route response continues to expose only the existing submit
success or failure payload.

## Intentional

- Cleanup runs after the ATLAS submit attempt, including ATLAS failure
  envelopes, because the private Blob has already been consumed into the server
  request path.
- Cleanup does not run when the Blob reference is invalid, unavailable, or too
  large; no ATLAS submit was attempted in those cases.
- Cleanup status is not returned to the browser. It is a secondary write and
  must not change the user-facing submit result.
- The Vercel Blob SDK remains lazy-imported so the route stays importable in
  local shell tests without requiring live Blob credentials or runtime wiring.

## Deferred

- Production observability for cleanup failures is deferred until the portfolio
  API has a shared logging/telemetry surface for Vercel serverless routes.

Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` -- 16 checks passed.
- `npm run test:deflection-result --prefix portfolio-ui` -- 12 checks passed.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` -- 14 checks passed.
- `npm run build --prefix portfolio-ui` -- passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Submit API cleanup | 35 |
| Focused tests | 126 |
| **Total** | **248** |

Actual diff is 3 files, +247 / -1. Under the 400 LOC soft cap.
