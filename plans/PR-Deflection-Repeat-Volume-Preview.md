# PR-Deflection-Repeat-Volume-Preview

## Why this slice exists

#1450 made the public FAQ-deflection upload run ATLAS inspect before private
Blob submit, and #1419's publishable-answer proof already landed through
#1424, #1430/#1444, and #1441. The remaining in-lane preview gap is from
#1440 Ask 2: a low-repeat-volume export can produce a technically valid but
thin report, so the pre-payment surfaces should show repeat-ticket volume and
warn when the export is light before the customer reaches the $1,500 checkout.

This is a follow-on to the deflection/clustering raw-data lane. It renders an
already-produced snapshot metric; it does not touch CSV parsing, report
generation, PDF/email delivery, Stripe, or the separate full-50k delivery
proof.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Functional validation

1. Project `snapshot.summary.repeat_ticket_count` through the existing
   portfolio snapshot projection.
2. Render the repeat-volume signal on the hosted locked result page next to
   the existing resolution-evidence lane.
3. Render the same signal on the React fallback result page for local/client
   snapshot continuity.
4. Keep the warning soft: light exports are not blocked because they can still
   be useful as gap-list diagnostics.

### Review Contract
- Acceptance criteria:
  - [ ] The portfolio snapshot proxy preserves `repeat_ticket_count` as a safe
        summary metric and still strips paid artifact fields.
  - [ ] Missing or non-finite `repeat_ticket_count` fails closed instead of
        synthesizing `0` and showing a false light-volume warning.
  - [ ] The hosted locked result page renders `repeat_ticket_count` from the
        snapshot summary before checkout, including a low-volume warning when
        it is small.
  - [ ] The React fallback result page renders the same repeat-volume cue from
        session-storage snapshots.
  - [ ] Existing resolution-evidence diagnostics and private-Blob submit
        gating remain unchanged.
- Affected surfaces: portfolio result proxy, portfolio result page, React
  fallback result page, CI-enrolled portfolio tests.
- Risk areas: API backcompat, PII/source leakage, frontend state coverage, CI
  enrollment.
- Reviewer rules triggered: R1, R2, R5, R9, R10, R12.

### Files touched

- `plans/PR-Deflection-Repeat-Volume-Preview.md`
- `portfolio-ui/api/content-ops/deflection/atlas-report.js`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`

## Mechanism

`faq_deflection_report.build_deflection_snapshot()` already includes
`summary.repeat_ticket_count`. The portfolio `atlas-report.js` projection
currently drops that safe summary metric, and the hosted result page renders
only questions, evidence-backed answers, and needs-proof counts.

This slice carries `repeat_ticket_count` through the existing safe snapshot
projection and renders a `Repeat-ticket volume` card before the Checkout CTA.
Counts under 10 get a soft "light on repeat volume" warning; higher counts get
the ready-state copy. The React fallback result page gets the same metric and
copy so local/client snapshot rendering stays aligned with the hosted API
route. Because the metric informs payment self-selection, both the hosted
projection and the React fallback parser require a finite value and reject
malformed snapshots instead of defaulting missing values to zero.

## Intentional

- This is a soft warning, not a hard gate. Low-volume uploads can still produce
  useful diagnostics, and blocking them would be a product-policy decision
  outside this narrow slice.
- The threshold is presentation-only. No new `ATLAS_*` config is introduced
  because no backend behavior changes.
- The slice uses the existing snapshot summary. It does not call an LLM/Ollama
  route, add a second clustering implementation, or recompute report quality in
  the browser.
- The full-50k intake -> snapshot -> email -> pay -> report-email proof from
  #1440 remains outside this session's lane.

## Deferred

- #1440 full-50k delivery proof remains with the delivery/payment lane.
- Operator-supplied production exports can tune the warning copy/threshold once
  enough live examples exist.
- Upload-inspect repeat-cluster diagnostics remain deferred; this slice uses
  the locked result-page snapshot because that is the pre-checkout surface.
- Closing stale tracker issues #1384/#1419 is operator/reviewer housekeeping,
  not part of this code slice.

Parked hardening: none.

## Verification

- `npm --prefix portfolio-ui run test:deflection-result` - passed, 19 checks.
- `npm --prefix portfolio-ui run test:deflection-atlas-proxy` - passed, 18 checks.
- `npm --prefix portfolio-ui ci` - passed; reports existing npm audit findings
  (1 moderate, 2 high), not changed in this slice.
- `npm --prefix portfolio-ui run build` - passed; Vite reports existing
  large-chunk and missing-sitemap-env warnings.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Repeat-Volume-Preview.md` | 119 |
| `portfolio-ui/api/content-ops/deflection/atlas-report.js` | 3 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 39 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 24 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 27 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 47 |
| **Total** | **259** |
