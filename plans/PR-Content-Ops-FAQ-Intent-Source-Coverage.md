# PR-Content-Ops-FAQ-Intent-Source-Coverage

## Why this slice exists

The FAQ product wedge promises intent-to-FAQ mapping from real queries across
search logs, chat transcripts, tickets, and sales objections. Search logs and
support tickets are already accepted by the FAQ generator, but source rows that
normalize as transcripts, sales calls, meetings, chats, or sales objections can
be filtered out by the default FAQ source-type allowlist.

That makes the feature look narrower than the source adapter already supports.
This slice closes the first-class source coverage gap without changing the FAQ
Markdown shape or adding vocabulary-gap detection.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add FAQ default source coverage for transcript, chat, sales-call, meeting,
   and sales-objection evidence.
2. Teach the source adapter common chat and sales-objection row labels so
   bundled uploads normalize to FAQ-eligible source types.
3. Add focused tests proving chat transcript and sales-objection inputs flow
   through the service and cluster into FAQ items.
4. Keep review/document source rows excluded from default FAQ generation.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Intent-Source-Coverage.md` | Plan doc for this FAQ source coverage slice. |
| `extracted_content_pipeline/campaign_source_adapters.py` | Recognize common chat and sales-objection bundle, id, text, and type labels. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Expand the default FAQ source-type allowlist to source types emitted for chats, transcripts, calls, meetings, and objections. |
| `tests/test_extracted_campaign_source_adapters.py` | Cover bundled chat and sales-objection source normalization. |
| `tests/test_extracted_ticket_faq_markdown.py` | Cover FAQ service ingestion for chat transcript and sales-objection sources. |

## Mechanism

TicketFAQMarkdownService.generate already normalizes arbitrary source material
through the campaign source adapter before calling the FAQ Markdown builder. The
generator then filters evidence rows by DEFAULT_TICKET_SOURCE_TYPES.

This slice keeps that flow and broadens the default allowlist to include the
customer-support and buyer-intent source types the adapter can emit for
conversation-shaped evidence. Source-material bundle expansion is centralized in
the adapter so the FAQ service does not need its own copy of bundle key names.
The adapter receives small alias additions so common bundle labels such as
`chats`, `chat_transcripts`, `sales_objections`, and row fields such as
`chat_id`, `objection_id`, and `objection_text` normalize without callers
needing to force `source_type` manually.

## Intentional

- No change to ranking, output checks, Markdown rendering, persistence, or the
  result JSON artifact.
- No vocabulary-gap detection in this slice; this is source coverage for the
  already-partial intent-to-FAQ mapping work.
- Review, survey, CRM, contract, renewal, and subscription evidence remain
  outside the default FAQ source filter unless a caller explicitly passes
  `source_types=()`.

## Deferred

- A later vocabulary-gap slice can compare customer terms against existing
  documentation terms and emit synonym/heading suggestions.
- A later intent-ranking slice can expose frequency x failure-risk summaries in
  a dedicated product artifact instead of only Markdown items and metadata.
- A later hosted upload slice can surface these accepted source families in the
  UI copy and examples.

## Verification

- Py compile for affected Python files - passed.
- Focused source-coverage pytest - passed, 9 tests.
- `pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_ticket_faq_markdown.py -q` - passed, 138 tests.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 89 |
| Source adapter aliases/helper | 37 |
| FAQ source allowlist/service routing | 44 |
| Tests | 162 |
| **Total** | **~335** |
