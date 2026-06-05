# PR-Content-Ops-FAQ-Weighted-Input-Volume-Diagnostics

## Why this slice exists

PR-Content-Ops-FAQ-Source-Mix-Diagnostics shows which source channels the FAQ
CLI recognized, but it still reports only row counts. Search exports and other
aggregated customer-language inputs can represent hundreds or thousands of
queries through fields such as `search_count`, `query_count`, or
`source_weight`.

This slice adds weighted input-volume diagnostics so proof runs can show both
physical row count and represented customer-demand volume without changing FAQ
generation behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Expose the generator's source-weight resolution as a public helper for CLI
   diagnostics.
2. Add total weighted input volume to the FAQ CLI source-mix diagnostics.
3. Add weighted input volume by source type and source channel.
4. Add focused CLI regression coverage for weighted mixed-source result
   diagnostics.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Weighted-Input-Volume-Diagnostics.md` | Plan doc for this diagnostics slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Exposes the canonical source-weight helper. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds weighted input-volume values to source-mix diagnostics. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers weighted input-volume diagnostics. |

## Mechanism

The FAQ generator already resolves row weight from canonical fields such as
`source_weight`, `search_count`, `query_count`, `request_count`, `occurrences`,
`volume`, and `frequency`. This PR exposes that helper and has the CLI call it
for each normalized opportunity plus its evidence rows.

Source-mix diagnostics remain compact and additive. They keep the existing counts and add
sorted weighted totals: one overall value, one by source type, and one by source
channel. Repeated source ids use the maximum seen weight for that source key,
matching the generator's "one represented source, largest aggregate count"
approach.

## Intentional

- CLI result JSON only. Markdown and generated FAQ items are unchanged.
- No new accepted input fields. The diagnostics use the generator's existing
  source-weight contract.
- No per-topic volume breakdown in this slice; this is input-level visibility.

## Deferred

- Hosted UI display for weighted source-mix diagnostics.
- Per-topic source-channel and weighted-volume breakdowns.
- Scale-run summary tables that combine output checks, source mix, and item
  score distribution.

## Verification

- Focused source-mix/weighted FAQ CLI pytest - passed, 1 test.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  130 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| Generator helper exposure | ~10 |
| CLI diagnostics | ~60 |
| Tests | ~25 |
| **Total** | ~180 |
