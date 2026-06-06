# PR-Content-Ops-FAQ-Vocabulary-Gap-Execute-Summary

## Why this slice exists

Hosted FAQ runs can now accept vocabulary-gap inputs and generate mappings, but
the execute result panel still hides that signal inside the raw JSON details.
Operators need immediate confirmation that a hosted run found vocabulary gaps
without opening the asset review drawer or expanding the full result payload.

This slice adds the thinnest execute-result summary for FAQ Markdown
vocabulary-gap output.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-execute-summary

1. Render a dedicated FAQ Markdown execution summary when an executed step has
   output `faq_markdown`.
2. Show generated item count, source row count, ticket source count, warning
   count, saved ids, and total vocabulary-gap mapping count.
3. Show up to three vocabulary-gap mappings with customer term, documentation
   term, source count, zero-result count, and opportunity score.
4. Keep the raw JSON result details unchanged.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Execute-Summary.md` | Plan doc for this execute-result visibility slice. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Adds the FAQ execution summary and small result-parsing helpers. |

## Mechanism

The existing `ExecutionStepSummary` switch renders output-specific summaries
before the raw JSON details. This slice adds a `faq_markdown` branch:

```tsx
faq_markdown result.items[].term_mappings -> summary chips + top mappings
```

The parser is defensive against unknown result shapes and uses the same
record/list helper style already used by the execute panel.

## Intentional

- UI-only slice. No backend generator, API, persistence, CLI, or asset review
  changes.
- No new chart, drawer, or table abstraction; this is inline execute feedback.
- Mapping list is capped at three entries to keep the execute panel compact.
- Raw JSON remains available for full inspection.

## Deferred

- Full sortable/searchable vocabulary-gap table remains a later product slice.
- Inline links from execute results to saved FAQ drafts remain separate.
- Current `HARDENING.md` entries were scanned; they are landing-page repair
  items and do not touch this FAQ execute-summary lane.

## Verification

- `npm run lint` from `atlas-intel-ui/` - passed.
- `npm run build` from `atlas-intel-ui/` - initially failed because the first
  implementation reused an asset-review-only `recordList` helper in
  `ContentOpsNewRun`; fixed by adding a local `recordArray` helper, then
  reran and passed.
- `git diff --check` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| FAQ execute summary UI | ~80 |
| Result parsing helpers | ~60 |
| **Total** | ~215 |
