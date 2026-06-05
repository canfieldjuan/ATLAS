# PR-Content-Ops-FAQ-Vocabulary-Gap-Ranking

## Why this slice exists

PR-Content-Ops-FAQ-Vocabulary-Gap-Term-Mapping added deterministic term-mapping
suggestions, but each mapping only said which customer term differs from a
documentation term. Operators still need to know which vocabulary gap is most
urgent, especially when zero-result searches and tickets point at the same
wording mismatch.

This slice adds frequency and failure-risk context to each term mapping and
sorts CLI mapping diagnostics by impact so the artifact can prioritize the
phrasing gaps most likely to reduce support load or zero-result searches.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add per-mapping frequency, failure-risk, zero-result source count, and
   opportunity score metadata.
2. Render compact impact context in the Markdown vocabulary-gap bullets.
3. Sort CLI term-mapping diagnostics by mapping opportunity score, frequency,
   zero-result count, and FAQ rank.
4. Add focused tests for zero-result ranking and unchanged no-gap behavior.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Ranking.md` | Plan doc for this ranking slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds mapping impact metadata and Markdown impact text. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Sorts compact term-mapping diagnostics by impact and exposes impact fields. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers ranking metadata, CLI ordering, and no-gap stability. |

## Mechanism

The FAQ item already computes frequency, failure-risk signals, and opportunity
score from the item rows. Vocabulary mappings are generated from the same row
set, so this slice reuses that row context instead of adding a new scoring
model.

Each mapping records:

- `source_id_count`
- `zero_result_source_count`
- `failure_risk_score`
- `failure_risk_signals`
- `opportunity_score`

The Markdown renderer appends a short "Seen in ..." sentence to each mapping
bullet. The CLI result JSON keeps the compact mapping summaries but sorts them
by `opportunity_score`, then frequency, then zero-result count, then FAQ rank.

## Intentional

- No change to FAQ item ranking, grouping, output checks, or source filtering.
- No new scoring model; mapping impact mirrors the existing FAQ opportunity
  score for the rows that produced the mapping.
- No long source-id lists in result JSON. The compact result keeps first source
  id plus counts only.

## Deferred

- A later slice can add a dedicated vocabulary-gap artifact if operators need a
  standalone report separate from FAQ Markdown.
- A later search-log slice can weight mappings by search volume when uploaded
  search exports include impression or count fields.
- A later glossary slice can make the synonym table data-driven.

## Verification

- Focused vocabulary-gap ranking pytest for builder and CLI behavior - passed, 5 tests.
- Full FAQ pytest for tests/test_extracted_ticket_faq_markdown.py - passed, 76 tests.
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
| Plan | 90 |
| FAQ generator | 33 |
| CLI diagnostics | 17 |
| Tests | 94 |
| **Total** | **~235** |
