# PR-Content-Ops-FAQ-Search-Volume-Weighting

## Why this slice exists

The FAQ generator now ranks FAQ opportunities and vocabulary gaps by frequency
and failure risk, but frequency still treats every source row as one occurrence.
That is correct for raw ticket exports, but not for aggregated search-log or
analytics exports where one row can represent many searches.

This slice lets pre-aggregated customer inputs carry explicit volume fields so a
query seen 200 times ranks above a one-off ticket without requiring callers to
duplicate rows.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add deterministic source-weight extraction for common aggregate fields such
   as `source_weight`, `search_count`, `query_count`, `request_count`,
   `occurrences`, and `volume`.
2. Use weighted frequency for FAQ item opportunity scoring and vocabulary-gap
   mapping scoring.
3. Preserve distinct source counts in `ticket_count`, `source_ids`, and
   rendered source coverage checks.
4. Surface weighted frequency in compact CLI item diagnostics.
5. Add focused tests for weighted FAQ ranking, weighted vocabulary-gap
   diagnostics, and unchanged one-row-per-source behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Search-Volume-Weighting.md` | Plan doc for this weighting slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds source-weight extraction and weighted opportunity scoring. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds weighted frequency to compact item diagnostics. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers weighted ranking and result diagnostics. |

## Mechanism

When evidence rows are admitted into FAQ grouping, the builder copies a
validated `source_weight` onto the internal row. The weight comes from the first
positive integer-like value found on the evidence or parent opportunity under
known aggregate field names. Missing, invalid, zero, and negative values fall
back to 1.

Opportunity scoring then sums one weight per distinct source key. If multiple
evidence rows share the same source id, the largest weight for that source is
used so duplicate evidence does not multiply the same aggregate row. Existing
distinct-source outputs remain unchanged: `ticket_count`, `source_ids`,
rendered coverage checks, and source labels still count source identities, not
weighted volume.

## Intentional

- No new source filtering or output-check gate.
- Raw ticket exports keep the same scores because missing weights default to 1.
- `ticket_count` remains a distinct-source count; weighted volume is reflected
  through `frequency`, `weighted_frequency`, and `opportunity_score`.
- `results_count` is not treated as volume because in search logs it means
  returned-result count, including zero-result detection.

## Deferred

- A later hosted upload slice can label these accepted aggregate fields in the
  UI.
- A later search-log slice can preserve separate impression/click metrics if
  product wants more than one volume dimension.
- A later glossary slice can apply weighted volume to standalone vocabulary-gap
  reports if those become a separate artifact.

## Verification

- Focused search-volume weighting pytest for builder and CLI behavior - passed, 4 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed, 79 tests.
- Py compile for affected Python files - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Git whitespace check - passed.
- Local PR review wrapper against origin/main - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 91 |
| FAQ generator | 41 |
| CLI diagnostics | 1 |
| Tests | 102 |
| **Total** | **~235** |
