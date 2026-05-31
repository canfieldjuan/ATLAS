# PR-FAQ-Deflection-Submit-Live-Validation

## Why this slice exists

Issue #1161 now has the paid Stripe/artifact leg verified end-to-end, and the
latest remaining public-funnel gap is the submit leg: portfolio upload ->
private Blob -> multipart POST to the now-live ATLAS `/submit`. The portfolio
route already reads private Blob objects server-side and forwards multipart
bytes, so this slice should not reimplement that path. It adds the missing
operator live-smoke harness around the existing production helper so the team
can prove the deployed submit loop after #1200 without building a parallel
submit implementation.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Functional validation

1. Add a portfolio-side live submit smoke that calls the existing
   `submitPrivateBlob()` helper against a configured ATLAS host.
2. Support two validation modes: a real private Blob pathname with the Blob
   token, or a local CSV fixture injected through the same Blob-read seam.
3. Redact secrets from smoke output and return only request/result metadata.
4. Enroll the smoke command and source-level checks in the existing upload-shell
   test.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Live-Validation.md`
- `portfolio-ui/package.json`
- `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The new smoke imports `submitPrivateBlob()` from the portfolio submit route. If
`--blob-pathname` and `--blob-token` are provided, it lets the production helper
read and clean up the real private Blob. If `--csv-file` is provided instead,
the script injects a fake Blob reader backed by that local CSV while still
posting the reconstructed multipart `FormData` to live ATLAS:

```text
submitPrivateBlob({
  config: { baseUrl, token, accountId, timeoutMs },
  payload: { blob_pathname, support_platform, company_name, contact_email, limit },
  getBlobImpl: csvFile ? localFixtureBlobReader : getPrivateBlob,
})
```

This keeps the validation on the exact server helper used by the browser
handoff while avoiding a new PII-serving URL or signed-proxy surface.

## Intentional

- This does not change the customer browser flow or route response shape.
- The smoke is operator-run; it is not enrolled in automatic CI because it needs
  a deployed ATLAS host and service JWT.
- Local CSV fixture mode is explicitly a validation fallback. Production wiring
  still uses private Vercel Blob pathname + server token.
- The smoke prints no bearer token, Blob token, raw CSV body, or ATLAS response
  body.

## Deferred

- Running the smoke against the deployed portfolio/ATLAS environment remains an
  operator step after this PR lands and environment values are available.
- Parked hardening: none.

## Verification

- `npm run test:deflection-upload-shell --prefix portfolio-ui` -- 19 checks passed.
- `npm run smoke:deflection-submit-live --prefix portfolio-ui -- --preflight-only --base-url https://atlas.example.com --token test-token --account-id 2b2b950d-f64b-4852-bc30-f92a34cdf169 --json` -- passed preflight.
- `npm run test:deflection-result --prefix portfolio-ui` -- 12 checks passed.
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` -- 14 checks passed.
- `node --check portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs` -- passed.
- `npm run build --prefix portfolio-ui` -- passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 89 |
| Live smoke script | 214 |
| Package command | 1 |
| Focused source tests | 41 |
| **Total** | **345** |

Actual diff is 4 files, +345 / -0. Under the 400 LOC soft cap.
