# PR-FAQ-Deflection-Intake-Copy-Alignment

## Why this slice exists

ATLAS #1367 calls out that the FAQ-deflection intake page still reads like a
file-upload utility even though the surrounding landing/result copy now sells a
deterministic, repeatable analysis system. The intake step is where prospects
hand over support-ticket data, so the copy needs to land the same trust angle
before upload: private handling, deterministic clustering, and no LLM analysis
of logs.

This is a product-polish slice only. The upload route, private Blob handoff,
submit payload, validation, result redirect, and paid-unlock flow are already
working and stay unchanged.

## Scope (this PR)

Ownership lane: deflection/intake-copy
Slice phase: Product polish

1. Reframe `FaqDeflectionUpload.tsx` hero/meta/helper copy from "CSV upload"
   utility language toward the deterministic FAQ-deflection outcome.
2. Replace implementation-note privacy copy with trust-language copy while
   preserving the existing private Blob/server-side mechanics.
3. Add the requested "100% Deterministic Engine" trust badge at the upload
   dropzone.
4. Keep form fields, file validation, Blob upload settings, submit endpoint,
   payload shape, progress state, and CTA behavior unchanged.
5. Extend the upload shell smoke test with source-level assertions for the new
   copy, badge, and preserved mechanical markers.

### Review Contract

- Acceptance criteria:
  - [ ] The intake H1/subhead frames the outcome and trust angle, not just
        "Support-ticket CSV upload".
  - [ ] The upload dropzone includes the exact "100% Deterministic Engine"
        badge copy requested in #1367.
  - [ ] Privacy/mechanics copy says data stays private, service tokens stay
        server-side, and logs are not sent to an LLM/generative AI for analysis.
  - [ ] Existing upload mechanics remain unchanged: CSV validation, private
        Blob upload, submit endpoint, JSON payload, progress UI, retry copy, and
        route wiring remain covered by the upload shell smoke test.
  - [ ] No checkout, result-page, ATLAS API, Stripe, or generator behavior is
        changed.
- Affected surfaces: `portfolio-ui` FAQ-deflection intake page and its source
  smoke test.
- Risk areas: frontend copy regression, accidentally changing upload semantics,
  overstating deterministic behavior beyond the intake analyzer.
- Reviewer rules triggered: R1, R2, R9, R12.

### Files touched

- `plans/PR-FAQ-Deflection-Intake-Copy-Alignment.md`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`

## Mechanism

The implementation keeps the component structure and state machine intact. It
only changes user-facing copy in three places:

1. SEO/hero framing on `FaqDeflectionUpload.tsx`.
2. Helper copy around account/privacy/upload mechanics.
3. A compact trust badge rendered inside the existing dropzone, directly under
   the file input.

The upload shell smoke test already reads the component source and checks route
wiring plus browser credential boundaries. This slice adds copy assertions
there so the deterministic/no-LLM trust language and exact badge text cannot
silently disappear.

## Intentional

- No shared badge component is introduced; the badge is a single, page-specific
  trust block in a one-page product-polish slice.
- No visual redesign beyond compact copy/badge placement; the existing
  `surface-*` and `primary-*` palette stays in use.
- No upload, validation, Blob, submit, checkout, result, Stripe, or ATLAS API
  logic changes.
- This does not resolve atlas-portfolio #198. That issue is a broader
  verify-before-delete backlog and remains filler after this intake slice.

## Deferred

- atlas-portfolio #198 dead-code/legacy-path verification remains separate and
  should not be mixed into this product-polish PR.

Parked hardening: none.

## Verification

- `npm --prefix portfolio-ui run test:deflection-upload-shell` - PASS; upload
  shell route/copy/mechanics assertions passed.
- `npm --prefix portfolio-ui run build` - PASS; Vite emitted the existing
  chunk-size and missing-sitemap-url warnings only.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/faq-deflection-intake-copy-alignment-pr-body.md`
  - PASS.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-FAQ-Deflection-Intake-Copy-Alignment.md` | 107 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 13 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | 39 |
| **Total** | **159** |
