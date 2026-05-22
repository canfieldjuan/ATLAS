# PR-Content-Ops-FAQ-Source-Mix-Diagnostics

## Why this slice exists

The FAQ generator now accepts support tickets, chat-like conversations, search
logs, sales objections, and public complaint-style rows. The compact result JSON
already exposes generated FAQ items and vocabulary-gap mappings, but it does not
summarize the input mix. That makes larger uploads harder to audit because an
operator cannot quickly confirm which demand channels were recognized.

This slice adds lightweight source-mix diagnostics to the FAQ CLI result output
so SMB and mid-market proof runs can show that multiple customer-language
sources were processed without inspecting the Markdown body.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add source-type counts to the FAQ CLI compact result JSON.
2. Add higher-level source-channel counts for support tickets, chats/search
   logs, sales inputs, complaints, and other rows.
3. Add a zero-result search source count so search-log demand is visible even
   when it is not the top FAQ item.
4. Add focused CLI regression coverage for mixed-source result diagnostics.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Source-Mix-Diagnostics.md` | Plan doc for this result-diagnostics slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Exposes the canonical zero-result search row helper for CLI diagnostics. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds source-mix diagnostics to compact result JSON. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers mixed-source diagnostics in CLI result output. |

## Mechanism

The CLI already has the normalized `loaded.opportunities` before generation.
`_result_payload` now receives those opportunities and builds a small
diagnostics block from their normalized `source_type`, source id, evidence rows,
and zero-result metadata.

Zero-result search counting uses the same public helper as the FAQ generator's
failure-risk scoring, so the diagnostic cannot drift from the scoring rule.

The diagnostics stay compact: sorted `source_type_counts`, sorted
`source_channel_counts`, and a distinct zero-result search source count. The
Markdown body and generator ranking stay unchanged.

## Intentional

- CLI result JSON only. The library result object stays focused on generated FAQ
  content and item metadata.
- No source ingestion changes. This slice reports what the existing adapter
  already normalizes.
- No weighted-volume rollup in this slice. Weighted frequency already appears
  on generated items and term mappings; this adds input visibility only.

## Deferred

- Hosted UI display for source-mix diagnostics.
- Per-topic source-channel breakdowns.
- Weighted input-volume diagnostics across all source rows.

## Verification

- Focused source-mix FAQ CLI pytest - passed, 2 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  130 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Local PR review against origin/main - passed; cross-layer caller hints for
  generic `main` and an unrelated same-name `_result_payload` helper were
  inspected and no external caller impact was found.
- Reviewer update: focused source-mix/zero-result FAQ pytest passed, 3 tests;
  full FAQ pytest passed, 130 tests; py compile, git whitespace check, extracted
  validation, extracted reasoning guard, extracted standalone audit, and
  extracted ASCII check passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Generator helper exposure | ~10 |
| CLI diagnostics | ~95 |
| Tests | ~65 |
| **Total** | ~250 |
