# PR-Deflection-Full-Thread-Export-Guidance

## Why this slice exists

Issue #1419 split the publishable-answer launch gate into three parts:
pre-payment resolution-evidence diagnostics, a deterministic proof run with
resolution-bearing source rows, and customer-facing export guidance. The first
two are already on `main` via #1424/#1428/#1430; the remaining small gap is the
intake surface still says only "support-ticket CSV".

That wording is incomplete for buyers. A CSV with inbound questions can produce
cluster diagnostics and a gap list, but publishable answer drafts require full
ticket threads with agent replies or resolved notes. This slice makes that
expectation visible before upload so customers do not mistake a question-only
export for the publishable-answer lane.

## Scope (this PR)

Ownership lane: deflection/clustering-raw-data
Slice phase: Product polish

1. Update the portfolio FAQ-deflection upload page copy to ask for full ticket
   threads, agent replies, and resolved ticket notes.
2. Keep the existing resolution-evidence honesty: question-only exports still
   produce a locked preview/gap-list signal, not a promise of publishable
   answers.
3. Pin the guidance in the upload-shell test that is already enrolled in
   `.github/workflows/portfolio_ui_checks.yml`.
4. Do not change upload mechanics, private Blob handling, checkout, result
   rendering, PDF/email delivery, or deterministic report generation.

### Review Contract

- Acceptance criteria:
  - [ ] The upload page names full ticket threads and agent replies/resolved
        notes as the best input for publishable answers.
  - [ ] The copy still preserves the no-LLM deterministic trust story.
  - [ ] The copy explicitly warns that question-only exports can produce a
        gap list but not publishable answer drafts.
  - [ ] The upload-shell test asserts those phrases so the guidance cannot
        regress silently.
  - [ ] `.github/workflows/portfolio_ui_checks.yml` already runs the touched
        test script.
- Affected surfaces: portfolio upload-page text and its existing shell test.
- Risk areas: overpromising that every export yields publishable answers,
  drifting into paid delivery/PDF lanes, or adding untested marketing copy.
- Reviewer rules triggered: R1, R9, R12, R13.

### Files touched

- `plans/PR-Deflection-Full-Thread-Export-Guidance.md`
- `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionUpload.tsx`

## Mechanism

The upload page already carries the guarded flow copy and private Blob
explanation. This PR adds one compact guidance block near the CSV picker with
the source-shape rule:

- best input: full ticket threads with customer question and agent reply or
  resolved note columns;
- question-only input: still accepted for clustering/gap diagnostics, but it
  cannot produce publishable answers unless resolution evidence is present.

The existing `test:deflection-upload-shell` source-level test asserts the new
phrases and already runs in
`.github/workflows/portfolio_ui_checks.yml`, so no new CI enrollment is needed.

## Intentional

- This is product guidance only. The backend already computes and displays the
  resolution-evidence lane signal; this slice does not duplicate that logic.
- This does not block question-only uploads. They are valid for gap-list
  analysis and preview diagnostics, but the buyer should know what they will
  get before paying.
- This does not touch `atlas-portfolio`; this repo's `portfolio-ui` copy is
  the in-repo surface covered by the current CI contract.

## Deferred

- Operator-supplied production upload/payment/delivery proof remains outside
  this slice; it is a live-run activity, not copy guidance.
- Any matching copy change in the separate `canfieldjuan/atlas-portfolio`
  deployment repo remains outside this repo and should be applied there if the
  deployed source has diverged.

Parked hardening: none.

## Verification

- `npm --prefix portfolio-ui run test:deflection-upload-shell`
  - Result: passed; 20 upload/private-Blob shell assertions.
- `npm --prefix portfolio-ui run build`
  - Result: passed; Vite emitted the existing large-chunk warning and skipped
    sitemap generation because no site URL env was set.
- `scripts/local_pr_review.sh` with the current PR body file.
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Full-Thread-Export-Guidance.md` | 107 |
| `portfolio-ui/scripts/faq-deflection-upload-shell.test.mjs` | 5 |
| `portfolio-ui/src/pages/FaqDeflectionUpload.tsx` | 14 |
| **Total** | **126** |
