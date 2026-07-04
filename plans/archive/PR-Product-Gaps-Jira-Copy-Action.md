# PR-Product-Gaps-Jira-Copy-Action

## Why this slice exists

Issue #1846 asks the paid Product Gap card to provide a copy-to-Jira action.
#1854 added the structured `jira_template` payload and rendered it in a
`<details>` disclosure, but the buyer still has to manually reconstruct the
handoff from prose. Root cause: the S3 surface stopped at displaying the Jira
handoff fields and did not construct a copy-ready handoff artifact at the paid
card boundary. This PR fixes the root for the buyer surface by rendering a
bounded, escaped, copy-ready Jira handoff block from the already-hosted-safe
`jira_template` fields.

## Scope (this PR)

Ownership lane: deflection/product-gaps-report-shape
Slice phase: Product polish

1. Add a copy-ready Jira handoff affordance to unlocked paid Product Gap cards.
2. Keep the handoff built only from the hosted-safe `jira_template`/action item
   fields already admitted by the report-model contract.
3. Prove the unlocked card renders the handoff and the locked/free page still
   does not render paid product-gap fields.

### Files touched

- `plans/PR-Product-Gaps-Jira-Copy-Action.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

### Review Contract

- Acceptance criteria:
  - [ ] Unlocked paid Product Gap cards render a copy-ready Jira handoff payload.
  - [ ] The payload includes issue, owner lane, impact/repeat count, cost basis,
        evidence tier, customer vocabulary, and next action when present.
  - [ ] The handoff is constructed from existing hosted-safe fields only; no
        `source_ids`, `top_evidence`, raw quotes, or new backend fields are
        introduced.
  - [ ] Locked/free result pages still do not render product-gap/Jira handoff
        fields.
- Affected surfaces: paid hosted result page, result-page smoke tests, plan docs.
- Risk areas: privacy boundary, frontend rendering/escaping, scope creep into
  backend report contract.
- Reviewer rules triggered: R1, R2, R3, R9, R10, R12, R14.

## Mechanism

Add a small result-page helper that constructs a deterministic Jira handoff
string from the already parsed action item and nested `jira_template` object.
The paid card renders that string in a read-only control with a copy affordance
that uses the browser Clipboard API when available and falls back to selecting
the text. The existing HTML escaping path remains the boundary for rendered
text, and the helper deliberately does not read raw evidence collections.

## Intentional

- No backend contract change; #1854 already published the hosted-safe
  `jira_template` shape.
- No source ID or raw quote handoff in the browser card; the complete audit trail
  stays in the paid evidence export.
- No generic copy-component abstraction; this is one small paid-result affordance
  for the Product Gap card.

## Deferred

- #1847 remains the next QA/proof slice after this S3 affordance is complete.
- #1854 plan archiving remains deferred because another active PR is touching
  `plans/INDEX.md`; keep this product slice conflict-free.
- Monthly cost normalization and richer owner taxonomy remain deferred from
  #1854.

Parked hardening: none.

## Verification

- `node portfolio-ui/scripts/faq-deflection-result-page.test.mjs` - passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/product-gaps-jira-copy-pr-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Product-Gaps-Jira-Copy-Action.md` | 87 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 61 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 14 |
| **Total** | **162** |
