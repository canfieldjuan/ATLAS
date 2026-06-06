# PR-Content-Ops-FAQ-Search-Log-Sources

## Why this slice exists

The intent-to-FAQ path can now rank FAQ opportunities by frequency and failure risk, but it still only admits support-ticket-like source types by default. The SMB/mid-market wedge needs real customer query language from search logs, especially zero-result searches, because those queries are direct FAQ demand signals even when no ticket was opened.

This slice makes search-log rows first-class input to the existing FAQ generator without building vocabulary-gap detection yet.

## Scope (this PR)

1. Add source adapter aliases for search-log bundles, query text, query ids, and common search-result fields.
2. Infer search-log source type from query-shaped rows.
3. Allow FAQ Markdown generation to use search-log/search-query evidence by default.
4. Add focused tests that prove search queries generate FAQ items and zero-result search rows survive bundle/CSV normalization.

### Files touched

| File | Intent |
|---|---|
| `extracted_content_pipeline/campaign_source_adapters.py` | Add search-log row-list, id, text, title, and source-type aliases. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Include search-log source types in the default FAQ evidence filter. |
| `tests/test_extracted_campaign_source_adapters.py` | Cover search-log normalization and nested bundle extraction. |
| `tests/test_extracted_ticket_faq_markdown.py` | Cover search-log evidence flowing into generated FAQ items. |
| `plans/PR-Content-Ops-FAQ-Search-Log-Sources.md` | Plan contract for this slice. |

## Mechanism

The adapter treats rows with fields such as `search_query`, `query`, `query_id`, or `zero_result_query_id` as search-log evidence. Query text is copied into the normalized evidence `text`, source ids are preserved, and common context fields such as `results_count`, `zero_results`, and `page_url` remain on the opportunity for downstream diagnostics.

`DEFAULT_TICKET_SOURCE_TYPES` then admits `search_log` and `search_query`, letting the existing FAQ grouping, scoring, and output checks consume customer search demand without a separate runtime path.

## Intentional

- This PR does not add docs corpus comparison or term-mapping suggestions. Search logs are the source input needed before vocabulary-gap detection can be useful.
- Search-query rows use the existing deterministic FAQ grouping and scoring. No new classifier or provider dependency is introduced.
- Search result counts are preserved as metadata, but this slice does not change the failure-risk formula to weight zero-result searches yet.

## Deferred

- `PR-Content-Ops-FAQ-Zero-Result-Risk`: optionally add zero-result count weighting to opportunity scoring.
- `PR-Content-Ops-FAQ-Vocabulary-Gaps`: compare search/customer terms against documentation terms and emit term-mapping suggestions.
- `PR-Content-Ops-FAQ-Sales-Objection-Sources`: add explicit sales-objection source aliases.

## Verification

Local verification:

```bash
pytest tests/test_extracted_campaign_source_adapters.py tests/test_extracted_ticket_faq_markdown.py -q
# 132 passed

bash scripts/run_extracted_pipeline_checks.sh
# extracted_reasoning_core: 295 passed
# extracted_content_pipeline: 1615 passed, 1 warning
```

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_source_adapters.py` | 31 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 7 |
| `tests/test_extracted_campaign_source_adapters.py` | 74 |
| `tests/test_extracted_ticket_faq_markdown.py` | 35 |
| `plans/PR-Content-Ops-FAQ-Search-Log-Sources.md` | 66 |
| **Total** | **213** |
