# PR-Deflection-Evidence-Export-Download

## Why this slice exists

Epic #1588 defines the paid deflection report as separate customer surfaces:
the hosted result page is the concise dashboard, the PDF becomes curated and
shareable, and the complete evidence export is the uncapped audit/detail
surface. #1591 added `deflection_evidence.v1` to the paid artifact, but that
export is still only an internal artifact field. A paid customer cannot fetch
it from the hosted result page yet.

This slice closes that buyer-access gap before the PDF redesign. The root
cause is that the paid web surface has no route or affordance for the already
persisted `artifact.evidence_export`; this PR fixes that root on the portfolio
surface by adding a paid-only download endpoint and rendering a download link
only after ATLAS reports the artifact is unlocked.

This is slightly over the 400 LOC soft cap because the slice includes the new
download route, positive and fail-closed route tests, paid-page render tests,
and exact-name plan archive teardown for the two just-merged predecessor PRs.
Splitting the tests or archive moves out would leave either an unproven paid
boundary or stale in-flight plans in `plans/`.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a portfolio API endpoint that returns the unlocked
   `artifact.evidence_export` as a JSON attachment.
2. Render an evidence-export download link on the paid hosted result page only
   when the unlocked artifact includes a structurally valid export object.
3. Fail closed for locked, missing, malformed, or not-yet-unlocked artifacts;
   do not synthesize export content from Markdown or snapshot data.
4. Keep PDF/email delivery and backend artifact generation unchanged.
5. Archive the merged #1590 and #1591 plan docs by name only, and refresh the
   plan index as teardown housekeeping.
6. Extend the existing portfolio deflection tests for success, locked/missing
   failure, malformed-export rejection, and no credential/account-id leakage.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Deflection-Evidence-Export-Download.md`
- `plans/archive/PR-Deflection-Complete-Evidence-Export.md`
- `plans/archive/PR-Deflection-Paid-Result-Page-Consolidated-View.md`
- `portfolio-ui/api/content-ops/deflection/evidence-export.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

### Review Contract

Acceptance criteria:

- [ ] A GET route returns `artifact.evidence_export` as JSON only after
      `loadDeflectionReport` returns an unlocked artifact with an object export.
- [ ] Locked, missing, malformed, or snapshot-only reports do not render a
      download link and the download route returns a fail-closed public error.
- [ ] The download response uses attachment headers and `Cache-Control:
      no-store`.
- [ ] The hosted paid result page renders the export link only in the unlocked
      paid dashboard; locked pre-payment snapshots stay free-only.
- [ ] The browser link carries only `request_id`, not `account_id`, ATLAS
      service credentials, raw CSV content, or paid artifact JSON inline.
- [ ] PDF/email delivery and backend artifact generation remain untouched.
- [ ] #1590 and #1591 plans are archived by exact name only; no bulk archive
      sweep moves concurrent in-flight plans.

Affected surfaces: portfolio deflection result page, portfolio deflection API
routes, portfolio route tests, and plan archive housekeeping.

Risk areas: paid artifact leakage before unlock, accidentally making the
complete export public, browser credential/account-id leakage, malformed
artifact fail-open behavior, and off-lane PDF/delivery changes.

Reviewer rules triggered: R1, R2, R3, R5, R9, R10, R12, R14.

## Mechanism

Add a small portfolio route beside the existing report proxy:

```js
GET /api/content-ops/deflection/evidence-export?request_id=<id>
```

The route reuses `loadDeflectionReport`, so request validation, configured
account binding, ATLAS token usage, snapshot projection, and artifact unlock
status stay on the same proxy path as the hosted result page. It then checks
that `artifact_status === "unlocked"` and `artifact.evidence_export` is an
object with `schema_version === "deflection_evidence.v1"` before returning it
as `application/json` with a deterministic attachment filename and
`Cache-Control: no-store`. Everything else returns a small public JSON error
and no artifact payload.

`renderPaidArtifact` gets a narrow export-availability helper. When the paid
artifact is unlocked and the export is present, the paid report nav includes a
download link to the new route. The link uses only the public `request_id`.
There is no Markdown parsing and no attempt to rebuild evidence rows in the
portfolio layer; the export remains the structured artifact created by #1591.

## Intentional

- No PDF redesign in this PR. #1588 says the curated/shareable PDF comes after
  the complete export exists and is buyer-accessible.
- No backend artifact changes. #1591 already added the export to the paid
  artifact; this slice exposes that exact field after unlock.
- No account id in the browser link. The portfolio proxy already binds to the
  configured `ATLAS_ACCOUNT_ID`, and exposing `account_id` in customer URLs has
  been intentionally avoided by the result-page slices.
- No raw Markdown/CSV fallback. If `artifact.evidence_export` is absent or
  malformed, the route/link fail closed instead of inventing a partial export.

## Deferred

- Epic #1588 next slice: curated/shareable PDF with plain table of contents
  and a clear pointer to this complete evidence export.
- Epic #1588 later slice: structured `deflection.v1` paid report model and
  section registry.
- Existing parked hardening left parked: `portfolio-ui` npm audit
  vulnerabilities in `HARDENING.md`; dependency upgrades are not required for
  this route/button slice and would be off-scope.

Parked hardening: none.

## Verification

- `cd portfolio-ui && npm run test:deflection-atlas-proxy`
  - Result: 23 passed.
- `cd portfolio-ui && npm run test:deflection-result`
  - Result: 27 passed.
- `git diff --check`
  - Result: passed.
- Local PR review bundle.
  - Pending via push wrapper before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 4 |
| `plans/PR-Deflection-Evidence-Export-Download.md` | 149 |
| `plans/archive/PR-Deflection-Complete-Evidence-Export.md` | 0 |
| `plans/archive/PR-Deflection-Paid-Result-Page-Consolidated-View.md` | 0 |
| `portfolio-ui/api/content-ops/deflection/evidence-export.js` | 116 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 41 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 189 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 12 |
| **Total** | **511** |
