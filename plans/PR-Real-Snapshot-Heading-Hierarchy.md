# PR-Real-Snapshot-Heading-Hierarchy

## Why this slice exists

PR #1359 added real uploaded-ticket customer wording to the locked FAQ
deflection Snapshot and merged with LGTM. The reviewer left one non-blocking
product-polish NIT: the React render path nests the SEO targeting list as a
section heading with repeated questions beneath it, but the server-rendered API
HTML leaves "Help-desk SEO targeting list" at the same heading level as each
question. This slice aligns the server-rendered outline so both paths describe
the same hierarchy.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Promote the server-rendered API HTML "Help-desk SEO targeting list" heading
   to the same outline level as the React page's SEO-list section heading.
2. Keep the question titles nested beneath that SEO-list heading.
3. Extend the existing `portfolio-ui` deflection result test to assert the
   server HTML heading order.
4. Leave customer-wording data, escaping, caps, Checkout, paid-gate, artifact,
   generator, and projection behavior unchanged.

### Files touched

- `plans/PR-Real-Snapshot-Heading-Hierarchy.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`

## Mechanism

`renderSnapshot()` already renders the server HTML Snapshot as one section:

- `<h2 id="snapshot-title">Free snapshot</h2>`
- SEO targeting list heading/copy
- customer wording card
- repeated question articles

This PR changes only the SEO-list heading tag in the API HTML from `h3` to
`h2` and pins the resulting server-rendered heading sequence in the existing
deflection result test. The React page already has the desired outline, so it
is not modified.

## Intentional

- This does not change visible copy, customer wording limits, or data source
  behavior; it is a document-outline polish slice only.
- The React page is left untouched because the review called it correctly
  nested already.
- No new `portfolio-ui` `test:*` script is added; the existing enrolled
  `test:deflection-result` script is the right coverage surface.

## Deferred

- Existing `HARDENING.md` `atlas-intel-ui` npm audit vulnerabilities remain
  parked because they are outside `portfolio-ui/faq-deflection`.

Parked hardening: none.

## Verification

- `npm run test:deflection-result --prefix portfolio-ui` -- 17 checks passed.
- `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Real-Snapshot-Heading-Hierarchy.md` | 74 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 2 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 6 |
| **Total** | **82** |
