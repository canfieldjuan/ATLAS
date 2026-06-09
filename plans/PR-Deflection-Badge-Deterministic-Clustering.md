# PR-Deflection-Badge-Deterministic-Clustering

## Why this slice exists

Issue #1367 tracks deterministic messaging for the FAQ deflection intake trust
badge. The existing subtext says "exact mathematical clustering," which is less
precise than the more defensible claims-doctrine wording. This slice swaps that
phrase to "deterministic clustering" and locks it with a source assertion.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-support-ticket-deflection
Slice phase: Product polish

1. Update the `portfolio-ui` intake trust-badge subtext to use
   "deterministic clustering" wording.
2. Update the upload-shell source assertion to require deterministic-clustering
   wording and reject the old exact-mathematical phrase.
3. Keep upload behavior, submit flow, and endpoint contracts unchanged.

### Review Contract

- Acceptance criteria:
  - [ ] Badge subtext on the FAQ deflection intake page uses
        "deterministic clustering" wording.
  - [ ] Upload-shell test asserts deterministic-clustering wording and rejects
        "exact mathematical clustering".
  - [ ] No mechanics change in intake upload/submit behavior.
- Affected surfaces: `portfolio-ui` intake copy and upload-shell source test.
- Risk areas: copy/test drift where wording changes without assertion updates.
- Reviewer rule IDs triggered: R1, R2, R9.

### Files touched

1. `plans/PR-Deflection-Badge-Deterministic-Clustering.md`
2. `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`
3. `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`

## Mechanism

Replace one phrase in the intake trust-badge paragraph and align the existing
source-level test assertion to the new phrase while adding a negative assertion
against the old wording.

## Intentional

- Copy/test-only update; no UI layout changes.
- No route, API, payload, or analytics changes in this slice.

## Deferred

- Any broader copy harmonization beyond this phrase swap remains out of scope.

Parked hardening: none.

## Verification

- `npm --prefix portfolio-ui run test:deflection-upload-shell` - PASS.
- `rg -n "deterministic clustering|exact mathematical clustering" portfolio-ui/src/pages/FaqDeflectionUpload.tsx portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
  - PASS; runtime + test now assert deterministic-clustering wording and reject
    exact-mathematical wording.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Badge-Deterministic-Clustering.md` | ~77 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | ~1 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | ~2 |
| Total | ~80 |
