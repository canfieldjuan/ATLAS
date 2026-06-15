# PR-Deflection-Paid-Result-Page-Consolidated-View

## Why this slice exists

Epic #1588 locks the deflection report product shape: the hosted result page is
the concise decision dashboard, the PDF is the curated/shareable artifact, and
the future evidence export is the only uncapped completeness surface. Today the
paid hosted result page still renders the unlocked artifact Markdown as an
escaped `<pre>` block. That makes the browser surface hard to use and keeps the
old "one giant report blob serves every job" shape in front of the customer.

Root cause: `renderPaidArtifact` treats `artifact.markdown` as the browser
report model. This PR fixes the first customer-facing layer by rendering the
already-persisted structured paid artifact data (`summary` + `faq_result.items`)
into a compact decision view, without changing PDF delivery or the backend
artifact contract.

This slice is slightly over the 400 LOC soft cap because the existing hosted
result page is a single Vercel route that owns projection helpers, HTML, CSS,
and route-level tests. Splitting the renderer from its tests or shipping only
part of the dashboard would leave the paid browser surface halfway between the
old Markdown dump and the epic's decision-dashboard shape.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Product polish

1. Replace the unlocked paid-result `<pre class="report-markdown">` dump with a
   structured web dashboard rendered from `artifact.summary` and
   `artifact.faq_result.items`.
2. Show concise paid sections: operating summary, readiness badges, top ranked
   questions, publishable answers, no-proven-answer gaps, top customer wording
   / SEO phrases, and an explicit full-detail delivery note.
3. Fail closed when the unlocked artifact lacks structured `faq_result.items`;
   do not parse Markdown or invent paid copy from a malformed artifact.
4. Keep PDF/email delivery and backend artifact generation unchanged.
5. Archive merged #1585's plan doc as teardown housekeeping in this branch.
6. Extend the existing portfolio result-page tests so the paid web surface is
   structured, escaped, and no longer a raw Markdown `<pre>`.

### Files touched

- `plans/INDEX.md`
- `plans/PR-Deflection-Paid-Result-Page-Consolidated-View.md`
- `plans/archive/PR-Deflection-Report-SEO-Target-Cap.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

### Review Contract

Acceptance criteria:

- Unlocked paid result pages render structured sections instead of
  `<pre class="report-markdown">`.
- The dashboard uses only structured artifact fields; it does not parse
  Markdown headings or infer missing paid copy.
- Repeat-ticket workload and support-tax estimates consume the canonical
  `artifact.summary.repeat_ticket_count` instead of re-summing item counts.
- Customer-provided text from questions, answers, steps, and evidence is HTML
  escaped.
- Locked, missing, or malformed artifacts do not render paid sections.
- PDF/email delivery and the backend artifact route are unchanged.
- Existing result-page retry/checkout metadata behavior remains intact.
- #1585's plan is archived by name only; no bulk plan archive sweep.

Affected surfaces:

- Hosted portfolio result page for paid deflection reports.
- Portfolio result-page contract tests.
- Plan archive housekeeping.

Risk areas:

- Accidentally leaking paid artifact content before unlock.
- Accidentally rendering unescaped ticket text as HTML.
- Quietly inventing or parsing paid copy from Markdown instead of structured
  artifact data.
- Breaking checkout retry/unlock behavior while changing the paid report block.

Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

Add small local projection helpers inside the result-page route:

- validate/extract `artifact.faq_result.items` as the paid source collection;
- consume the artifact summary's canonical repeat-ticket count for workload and
  support-tax metrics;
- partition top publishable answers by `answer_evidence_status ===
  "resolution_evidence"`;
- partition no-proven-answer gaps from the remaining items;
- derive top customer wording / SEO phrases from `customer_wording` or
  `question`.

`renderPaidArtifact` will render a bounded dashboard over that structured data.
It will not render `artifact.markdown` into the browser. The full-detail
artifact remains available through the existing email/PDF path; this PR only
changes the hosted browser view.

## Intentional

- No PDF changes in this PR. The curated PDF is blocked on the future complete
  evidence export from epic #1588.
- No evidence export in this PR. This slice only fixes the web consumption
  surface.
- No new persisted report schema in this PR. The structured paid report model
  is a later strangler slice; this PR uses the existing artifact shape.
- No Markdown parsing. Parsing the current report text would preserve the blob
  dependency the epic is trying to unwind.

## Deferred

- Epic #1588 slice 2: complete evidence export + promise text update.
- Epic #1588 slice 3: curated/shareable PDF with plain TOC after export exists.
- Epic #1588 later slice: structured `deflection.v1` paid report sections and
  per-surface renderers.

Parked hardening: none.

## Verification

- Command: `cd portfolio-ui && npm run test:deflection-atlas-proxy`
  -- 18 passed.
- Command: `cd portfolio-ui && npm run test:deflection-result`
  -- 27 passed.
- Command: `python scripts/sync_pr_plan.py plans/PR-Deflection-Paid-Result-Page-Consolidated-View.md --check`
  -- passed.
- Command: `git diff --check`
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Paid-Result-Page-Consolidated-View.md` | 143 |
| `plans/archive/PR-Deflection-Report-SEO-Target-Cap.md` | 0 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 266 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 45 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 76 |
| **Total** | **531** |
