# PR-Content-Ops-FAQ-Catalog-Driven-Inputs

## Why this slice exists

PR-Content-Ops-FAQ-Vocabulary-Gap-Input-Contracts published the hosted FAQ
vocabulary-gap input contracts through `GET /content-ops/control-surfaces`, but
the Content Ops new-run screen still renders those FAQ labels and placeholders
from hardcoded UI literals.

That leaves one last small drift point between the backend catalog and the UI.
This slice consumes the catalog contracts in the existing FAQ controls.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-ui-controls

1. Resolve the FAQ documentation-term and vocabulary-gap-rule contracts from the
   fetched catalog when FAQ Markdown is selected.
2. Render FAQ field labels and placeholders from those contracts, falling back
   to the existing literals only if the catalog entry is absent.
3. Preserve the existing text-area behavior that writes
   `faq_documentation_terms` and `faq_vocabulary_gap_rules` into inputs JSON.
4. Add focused coverage proving catalog labels/placeholders win over fallback
   literals.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Catalog-Driven-Inputs.md` | Plan doc for this UI contract-consumption slice. |
| `atlas-intel-ui/package.json` | Adds the focused Node test command. |
| `atlas-intel-ui/scripts/content-ops-input-display.test.mjs` | Verifies catalog display metadata precedence. |
| `atlas-intel-ui/src/domain/contentOps/inputDisplay.ts` | Provides the small display-metadata resolver used by the page. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the display resolver through the domain barrel. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Reads FAQ labels/placeholders from catalog input contracts. |

## Mechanism

`ContentOpsNewRun` already has the normalized `catalog.inputContracts` map and
already conditionally renders FAQ controls when `faq_markdown` is selected. This
slice adds two small contract lookups:

```ts
const faqDocumentationTermsContract =
  catalog.inputContracts[FAQ_DOCUMENTATION_TERMS_INPUT]
const faqVocabularyGapRulesContract =
  catalog.inputContracts[FAQ_VOCABULARY_GAP_RULES_INPUT]
```

The FAQ textareas continue to use the same draft-value and update helpers. Only
display metadata moves from literals to the catalog.

The precedence rule lives in a tiny pure helper so it can be covered by the
existing Node `--test` pattern without introducing a frontend test framework:

```ts
inputContractDisplay(contract, fallback)
```

## Intentional

- Fallback labels/placeholders remain in the UI so older or partial backend
  catalogs still render usable controls.
- No generic form renderer is introduced. The slice is the thinnest end-to-end
  use of the new FAQ contracts, not a control-surface refactor.
- No additional validation is added for `nested_string_list`; submit-time
  backend validation remains the source of truth.
- No Vitest/JSDOM setup is introduced for this one assertion. The test covers
  the display precedence rule directly.

## Deferred

- A generic catalog-driven renderer for grouped input contracts is larger than
  this slice and should be scoped separately if repeated input groups keep
  growing.
- Rich client-side validation for nested FAQ vocabulary rules remains a future
  hardening item.
- Current `HARDENING.md` entries were scanned; they are landing-page repair
  items and do not touch this FAQ UI lane.

## Verification

- `npm run test:content-ops-input-display` from `atlas-intel-ui/` - passed,
  2 tests.
- `npm run lint` from `atlas-intel-ui/` - passed.
- `npm run build` from `atlas-intel-ui/` - passed; frontend contract fixtures
  compiled and Vite generated sitemap/prerender output.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main --allow-dirty` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| FAQ UI contract lookups/rendering | ~35 |
| Display helper and export | ~25 |
| Focused Node test and package script | ~50 |
| **Total** | ~200 |
