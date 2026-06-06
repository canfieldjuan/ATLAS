# PR-Content-Ops-FAQ-Vocabulary-Gap-UI-Inputs

## Why this slice exists

PR-Content-Ops-FAQ-Vocabulary-Gap-Execute-Config wired
`faq_documentation_terms` and `faq_vocabulary_gap_rules` through hosted
execution, but the Content Ops run form still requires operators to hand-edit
the raw inputs JSON to exercise vocabulary-gap detection.

This slice adds the thinnest hosted UI path for entering those two FAQ inputs
and sending them through the existing preview/plan/execute request flow.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-ui-inputs

1. Show a FAQ vocabulary-gap inputs panel only when `faq_markdown` is selected.
2. Let operators enter documentation terms as a newline/comma-separated list.
3. Let operators enter vocabulary-gap rules as one rule per line using
   `customer term, documentation term`.
4. Serialize non-empty controls into the existing inputs JSON under the backend
   keys `faq_documentation_terms` and `faq_vocabulary_gap_rules`.
5. Keep invalid raw inputs JSON as the existing blocking state for per-output
   controls.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-UI-Inputs.md` | Plan doc for this hosted UI input slice. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Adds FAQ-only controls that write documentation terms and vocabulary-gap rules into the request inputs JSON. |

## Mechanism

The existing Content Ops form already treats `inputsJson` as the canonical
request state and has landing-page helper controls that patch that JSON. This
slice follows the same pattern for FAQ Markdown:

```ts
faq panel draft -> update FAQ keys in inputsJson -> buildDomainRequest()
```

The documentation-term field uses the existing string-list parser. The custom
rule field parses each non-empty line into a string array, preserving the
backend's structured `string[][]` contract from PR-Content-Ops-FAQ-
Vocabulary-Gap-Execute-Config. Backend validation still enforces at least two
terms per rule.

## Intentional

- UI-only slice. No backend generator, dispatcher, API, persistence, or CLI
  changes.
- No separate React state for FAQ inputs; the inputs JSON remains canonical.
- No file upload for documentation terms or rule files in this slice.
- Invalid vocabulary-rule line shape is allowed to surface through the backend
  400 path; this slice keeps the UI thin and only serializes structured arrays.

## Deferred

- Per-file documentation term upload remains a separate product slice.
- Rich validation or inline per-line error messages for FAQ rules remain
  hardening unless the current slice cannot submit the real flow.
- Current `HARDENING.md` entries were scanned; they are landing-page repair
  items and do not touch this FAQ UI lane.

## Verification

- `npm run lint` from `atlas-intel-ui/` - passed.
- `npm run build` from `atlas-intel-ui/` - passed; Vite built
  `ContentOpsNewRun` and generated sitemap/prerender output.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| FAQ UI controls | ~90 |
| FAQ input helpers | ~80 |
| **Total** | ~245 |
