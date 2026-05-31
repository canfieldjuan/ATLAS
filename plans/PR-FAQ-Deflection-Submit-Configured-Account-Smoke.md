# PR-FAQ-Deflection-Submit-Configured-Account-Smoke

## Why this slice exists

PR-FAQ-Deflection-Submit-Configured-Account removed the buyer-facing ATLAS
account field from the FAQ deflection upload form and bound private Blob submit
requests to the configured server account. The existing live submit smoke still
calls `submitPrivateBlob()` directly, so it proves the helper but not the
public portfolio submit route that now owns the configured-account contract.

This slice adds a route-handler mode to the existing smoke so an operator can
validate the real `/api/content-ops/deflection/submit` JSON/private-Blob path
without reintroducing a browser-supplied account id.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Functional validation

1. Extend the portfolio submit live smoke with `--route-handler` mode.
2. In route-handler mode, call the default portfolio submit API handler with
   JSON private-Blob payload and no `X-Atlas-Account-Id` header.
3. Require a real private Blob pathname/token for route-handler mode; local CSV
   fixture mode remains helper-only.
4. Redact the smoke output to request/result metadata only.
5. Add focused tests for the new validation mode and its preflight failures.

### Files touched

- `plans/PR-FAQ-Deflection-Submit-Configured-Account-Smoke.md`
- `portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

The current smoke keeps its default helper path:

```text
submitPrivateBlob(...) -> ATLAS multipart submit
```

The new `--route-handler` mode builds the same JSON body the browser sends
after private Blob upload:

```json
{
  "blob_pathname": "faq-deflection/uploads/...",
  "support_platform": "zendesk",
  "company_name": "Acme Co.",
  "contact_email": "lead@example.com",
  "limit": "1000"
}
```

It then calls the portfolio submit route handler with `Content-Type:
application/json`, the configured server env values, and no account header.
That exercises the server's configured `ATLAS_ACCOUNT_ID`, private Blob read,
ATLAS multipart forwarding, projection, and cleanup path.

## Intentional

- Route-handler mode requires a real private Blob pathname and token because
  the route handler intentionally owns the Blob SDK boundary; local CSV fixture
  mode stays on the helper seam where a fake reader can be injected.
- This does not change customer browser behavior or API responses.
- The smoke still prints no bearer token, Blob token, raw CSV content, or raw
  ATLAS response body.
- This does not touch Checkout or Stripe key behavior from #1206.

## Deferred

- Running route-handler mode against the deployed portfolio/ATLAS environment
  remains an operator step once a private Blob fixture is available.
- Parked hardening: none.

## Verification

- `node --check portfolio-ui/scripts/faq-deflection-submit-live-smoke.mjs`
- `node --check portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `npm run test:deflection-upload-shell --prefix portfolio-ui` (21 checks)
- `npm run test:deflection-result --prefix portfolio-ui` (15 checks)
- `npm run test:deflection-atlas-proxy --prefix portfolio-ui` (14 checks)
- `bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-submit-configured-account-smoke.md`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 94 |
| Smoke route-handler mode | 123 |
| Focused tests | 89 |
| **Total** | **306** |

Under the 400 LOC soft cap.
