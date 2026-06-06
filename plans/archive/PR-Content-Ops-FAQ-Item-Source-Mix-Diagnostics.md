# PR-Content-Ops-FAQ-Item-Source-Mix-Diagnostics

## Why this slice exists

The FAQ CLI now reports whole-upload source mix and weighted represented volume.
Operators can see that a run included tickets, search logs, chats, sales inputs,
and aggregate search demand, but they still cannot tell which source channels
contributed to each generated FAQ opportunity.

This slice adds compact per-item source-mix diagnostics so a ranked FAQ
opportunity can show whether it came from tickets alone, search demand, sales
objections, or a blend of channels.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add per-item source-type counts from the generator's grouped evidence rows.
2. Add per-item weighted source volume by source type.
3. Include per-item source-channel counts and weighted source-channel volume in
   the CLI compact item summaries.
4. Add focused CLI regression coverage for item-level mixed-source diagnostics.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Item-Source-Mix-Diagnostics.md` | Plan doc for this item diagnostics slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds per-item source-type and weighted-volume metadata. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds per-item source-channel diagnostics to compact result JSON. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers item-level mixed-source diagnostics. |

## Mechanism

The generator already builds each FAQ item from grouped normalized source rows.
This PR summarizes those rows before rendering: distinct source keys by
source type become source-type counts, and the existing source-weight value
on each row is summed by source type using the same max-per-source approach as
weighted frequency.

The CLI item summary keeps the raw source-type maps and derives channel maps
from the existing source-channel classifier. The Markdown body stays unchanged.

## Intentional

- No generation or ranking changes. This only adds metadata to item diagnostics.
- No new source channels. The CLI uses the same source-channel classifier as
  whole-upload source-mix diagnostics.
- No hosted UI display yet; this remains compact result JSON.

## Deferred

- Hosted UI display for item-level source-mix diagnostics.
- Per-item zero-result source counts by channel.
- Scale-run summary tables that combine item source mix with output checks.

## Verification

- Focused item/source-mix FAQ CLI pytest - passed, 2 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  130 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Local PR review against origin/main - passed after renaming the new helper to
  avoid a same-name private-helper caller-hint warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Generator item metadata | ~55 |
| CLI item diagnostics | ~40 |
| Tests | ~35 |
| **Total** | ~210 |
