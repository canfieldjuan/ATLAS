# PR-Content-Ops-FAQ-Weighted-Aggregation-Helper

## Why this slice exists

PR-Content-Ops-FAQ-Item-Source-Mix-Diagnostics added the last planned
source-mix diagnostic surface, and the reviewer correctly noted that the
max-per-source weighted aggregation now appears in multiple places. Leaving that
logic duplicated invites drift between item ranking, per-item weighted source
mix, and upload-level weighted source mix.

This slice consolidates that aggregation after the diagnostics surfaces have
settled, without changing generated Markdown or result JSON shape.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add one shared weighted-source aggregation helper in the FAQ generator.
2. Route item weighted frequency through the helper.
3. Route item source-type weighted volume through the helper.
4. Route CLI source-mix weighted input volume through the helper.
5. Re-run focused regression coverage that locks output parity for the existing
   weighted diagnostics.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Weighted-Aggregation-Helper.md` | Plan doc for this refactor slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds the shared weighted aggregation helper and uses it for item ranking/metadata. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Uses the shared helper for upload-level source-mix weighted volume. |
| `tests/test_extracted_ticket_faq_markdown.py` | Adds direct coverage for the public helper's default weight aliases. |

## Mechanism

The new helper accepts normalized rows plus a group-key function. For each
group/source-key pair it keeps the maximum represented source weight, then sums
those maximums by group. The default source-key and weight behavior matches the
existing generator logic; CLI diagnostics can pass their existing source-key and
weight callbacks for opportunity-level rows.

Using the same helper preserves current behavior while making the
max-per-source rule a single implementation.

## Intentional

- Refactor only. No generated Markdown, ranking, or result JSON field names
  change.
- The helper supports custom source-key and weight callbacks because CLI
  upload-level diagnostics aggregate opportunities plus evidence rows, while
  generator item metadata aggregates already-grouped evidence rows.
- The per-type breakdown can still count a rare multi-type source key once per
  type; the previous PR documented that behavior and this helper preserves it.

## Deferred

- Hosted UI display for item-level source-mix diagnostics.
- Per-item zero-result source counts by channel.
- Scale-run summary tables that combine item source mix with output checks.

## Verification

- Reviewer update: focused weighted/source-mix FAQ CLI pytest - passed, 2 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  131 tests.
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
| Generator helper refactor | ~45 |
| CLI helper reuse | ~25 |
| Tests | ~15 |
| **Total** | ~170 |
