# PR-Real-Snapshot-Customer-Wording

## Why this slice exists

The real locked FAQ deflection Snapshot page already receives
`top_questions[].customer_wording`, but the current UI only exposes those
phrases inside each ranked row. The user asked for parity with the `/snapshot`
demo by adding a bounded "Customer wording" list/card that makes the actual
ticket phrases visible as a long-tail help-desk SEO target list without
inventing search terms or promising volume/ranking/traffic.

## Scope (this PR)

Ownership lane: portfolio-ui/faq-deflection
Slice phase: Product polish

1. Add a compact customer-wording card/list to the real locked Snapshot result
   page, derived only from `snapshot.top_questions[].customer_wording`, trimmed,
   filtered for non-empty phrases, and capped at five examples.
2. Keep the existing ranked-row customer wording unchanged and do not touch
   ATLAS schema/projection, Checkout, paid-gate, artifact, or generator code.
3. Extend the existing `portfolio-ui` result-page smoke/source assertions so the
   visible card label, derivation expression, accessible list marker, bounded
   rendering, and no-invented-terms fail-closed copy are covered.

### Review Contract
- Acceptance criteria:
  - [ ] The locked React Snapshot page renders a "Customer wording" card near
        the ranked question list when a real snapshot is available.
  - [ ] The grouped phrases are derived from
        `snapshot.top_questions.map((question) => question.customer_wording.trim())`,
        exclude empty phrases, and render at most five examples.
  - [ ] Empty wording/example state fails closed with copy that says no invented
        SEO terms are displayed.
  - [ ] Row-level customer wording in ranked questions remains unchanged.
  - [ ] The hosted HTML renderer used by the deployed rewrite keeps the same
        visible doctrine for real snapshot payloads.
- Affected surfaces: frontend, portfolio hosted result renderer, source-level
  smoke tests.
- Risk areas: frontend copy/accessibility, privacy boundary, CI enrollment.
- Reviewer rules triggered: R1, R2, R9, R12.

### Files touched

- `plans/PR-Real-Snapshot-Customer-Wording.md`
- `portfolio-ui/api/content-ops/deflection/result-page.js`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `portfolio-ui/src/pages/FaqDeflectionResult.tsx`

## Mechanism

`FaqDeflectionResult.tsx` derives `customerWordingExamples` with a `useMemo`
from the already-validated snapshot rows:

```tsx
snapshot.top_questions
  .map((question) => question.customer_wording.trim())
  .filter((phrase) => phrase.length > 0)
  .slice(0, 5)
```

The available-snapshot block renders a small "Customer wording" card before the
ranked rows. The card uses `aria-label="Customer wording examples"` on the list
and uses fail-closed copy instead of synthetic keywords if no real phrases are
present.

The hosted API HTML renderer mirrors the same source-only list because
`portfolio-ui/vercel.json` rewrites result URLs to
`api/content-ops/deflection/result-page.js`. The renderer keeps escaping all
phrases and leaves the existing row-level `item.customer_wording` output intact.

## Intentional

- No keyword volume, ranking, traffic, or savings claims are added. The wording
  is described only as observed ticket phrasing that can inform help-center
  titles and internal-search synonyms.
- No schema/projection changes are made because `customer_wording` is already
  present in the locked snapshot contract.
- The existing `test:deflection-result` script is extended instead of adding a
  new `test:*` script, so no portfolio workflow enrollment change is needed.

## Deferred

- The existing `HARDENING.md` item for `atlas-intel-ui` npm audit
  vulnerabilities is unrelated to `portfolio-ui/faq-deflection` and remains
  parked.

Parked hardening: none.

## Verification

- `npm run test:deflection-result --prefix portfolio-ui` - 17 checks passed.
- `npm run build --prefix portfolio-ui` - passed after hydrating local
  `portfolio-ui/node_modules` with `npm install --prefix portfolio-ui`; Vite
  emitted the existing large-chunk advisory and skipped sitemap generation
  because no `PORTFOLIO_SITE_URL` / `VITE_SITE_URL` was set.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed advisory review
  before commit.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-real-snapshot-customer-wording-body.md`
  - passed before push.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Real-Snapshot-Customer-Wording.md` | 110 |
| `portfolio-ui/api/content-ops/deflection/result-page.js` | 35 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 64 |
| `portfolio-ui/src/pages/FaqDeflectionResult.tsx` | 49 |
| **Total** | **258** |
