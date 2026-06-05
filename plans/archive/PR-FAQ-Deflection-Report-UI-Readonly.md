# PR-FAQ-Deflection-Report-UI-Readonly

## Why this slice exists

The `faq_deflection_report` backend is now complete as a pipeline-only product
path: generation artifact, execute output, contract docs, bulk/concurrency
proofs, and CLI ergonomics have all landed. The next product gap is the Intel
UI: a user can execute the output only through the existing generic run form,
but the result is not rendered as the $1,500 deflection report deliverable.

This slice builds the thinnest end-to-end UI path: select/execute
`faq_deflection_report` through the existing Content Ops execute route and
render the returned report read-only from the verified frontend contract.
The diff is over the 400 LOC soft cap because the vertical slice includes the
typed contract parser, the read-only UI renderer, the focused fixture test, and
CI enrollment for that test so the check is not left as unenrolled local-only
coverage.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-ui

Slice phase: Vertical slice

1. Add typed UI/domain helpers for the `FAQDeflectionReportArtifact` contract
   from `docs/frontend/content_ops_faq_report_contract.md`.
2. Render completed `faq_deflection_report` execute results in
   `ContentOpsNewRun` with ranked opportunities, proven drafted answers, and
   no-proven-answer gaps as distinct sections.
3. Add a focused UI contract test using
   `docs/frontend/content_ops_faq_deflection_report_example.json`.
4. Enroll that focused test in `.github/workflows/atlas_intel_ui_checks.yml`.

### Files touched

| File | Purpose |
|---|---|
| `atlas-intel-ui/src/domain/contentOps/faqDeflectionReport.ts` | Adds typed parsing/splitting helpers for the report artifact. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the helper for screens/tests. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Renders the deflection report execute result read-only. |
| `atlas-intel-ui/scripts/content-ops-deflection-report-ui.test.mjs` | Verifies the split against the canonical example and the screen branch. |
| `atlas-intel-ui/package.json` | Adds the focused npm test script. |
| `.github/workflows/atlas_intel_ui_checks.yml` | Runs the focused test in the UI CI lane. |
| `HARDENING.md` | Parks pre-existing npm audit dependency findings observed during setup. |
| `plans/PR-FAQ-Deflection-Report-UI-Readonly.md` | Documents this slice contract. |

## Mechanism

The generic execute flow already sends selected catalog output IDs through
`executeContentOpsRun(toWireRequest(...))`. This PR keeps that path and adds a
renderer branch for `faq_deflection_report`.

The domain helper accepts a raw `Record<string, unknown>`, validates the minimum
artifact shape, and splits `faq_result.items` with the same rule as the backend:

```ts
item.answer_evidence_status === 'resolution_evidence'
```

Only proven items render in "Drafted Answers (proven solutions)". All other
items render in "No Proven Answer Yet", so the UI cannot style an unbacked gap
as a publishable answer.

## Intentional

- No rule-file, intent-rule, or documentation-term controls are added here.
  Operators can use the raw JSON input field until a follow-up control slice.
- No persistence or review flow is added. This is execute-result rendering only.
- The existing output selector remains catalog-driven; the UI does not hardcode
  a fake output card when the backend catalog omits `faq_deflection_report`.

## Deferred

- Parked hardening: `atlas-intel-ui npm audit vulnerabilities` in
  `HARDENING.md`; this was observed during `npm ci`, is not introduced by this
  slice, and would require dependency upgrade work outside the read-only report
  UI path.
- Future product-polish slice: add first-class controls for documentation terms,
  vocabulary-gap rules, and intent rules.
- Future product-polish slice: add richer Markdown rendering/export affordances
  after the read-only contract view is proven.

## Verification

- Command: npm ci
  - Result: passed; reported 6 existing audit findings, parked in `HARDENING.md`.
- Command: npm run test:content-ops-deflection-report-ui
  - Result: 3 passed.
- Command: npm run lint
  - Result: passed.
- Command: npm run build
  - Result: passed.
- Command: npm run test:landing-page-prerender
  - Result: 4 passed.
- Command: npm run verify:blog-geo
  - Result: passed; verified 14 blog pages.
- Command: npm run verify:landing-page-geo
  - Result: passed; skipped because no generated landing-page sitemap entries were present.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-UI-Readonly.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-UI-Readonly.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Review fix verification: npm run test:content-ops-deflection-report-ui
  - Result: 3 passed.
- Review fix verification: npm run lint
  - Result: passed.
- Review fix verification: npm run build
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Domain helper | 161 |
| UI renderer | 289 |
| Test + CI enrollment | 84 |
| Hardening entry | 11 |
| Plan doc | 121 |
| **Total** | **677** |
