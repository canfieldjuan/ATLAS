# PR-Content-Ops-FAQ-Zero-Result-Risk

## Why this slice exists

PR-Content-Ops-FAQ-Search-Log-Sources made search-log rows first-class FAQ evidence, but zero-result searches are still scored the same as any other query unless their text happens to match an existing failure keyword. For an FAQ product wedge, a zero-result search is a direct failure signal: the customer asked in their own words and the help surface returned nothing useful.

This slice closes that gap by making zero-result search rows contribute to FAQ opportunity risk.

## Scope (this PR)

1. Preserve search-result metadata from opportunities/evidence on grouped FAQ rows.
2. Add a deterministic `zero_result_search` failure-risk signal for search-log rows with zero-result metadata.
3. Let zero-result search signals affect existing `opportunity_score = frequency * (1 + failure_risk_score)` ranking.
4. Add focused regression tests for metadata preservation, signal output, and ranking.

### Files touched

| File | Intent |
|---|---|
| `extracted_content_pipeline/ticket_faq_markdown.py` | Carry search metadata into scoring and add the zero-result risk signal. |
| `tests/test_extracted_ticket_faq_markdown.py` | Cover zero-result scoring/ranking behavior. |
| `plans/PR-Content-Ops-FAQ-Zero-Result-Risk.md` | Plan contract for this slice. |

## Mechanism

The existing grouping loop builds compact row dictionaries for each FAQ topic. This PR adds source type and search-result fields to those grouped rows, using evidence values first and opportunity values as fallback. The failure-risk signal pass then appends `zero_result_search` when any grouped row is search-log/search-query evidence with zero-result metadata such as `zero_results=true`, `results_count=0`, or `result_count=0`.

The scoring formula remains unchanged:

```python
opportunity_score = frequency * (1 + failure_risk_score)
```

Zero-result searches increase `failure_risk_score` through the new signal.

## Intentional

- This is still deterministic and provider-free.
- Zero-result searches count once per FAQ topic as a signal category, not once per row. Frequency already captures repeated searches.
- This does not yet produce vocabulary-gap term mappings; it only makes zero-result search demand visible in the intent ranking.

## Deferred

- `PR-Content-Ops-FAQ-Vocabulary-Gaps`: compare customer/search terms against documentation terms and emit term-mapping suggestions.
- `PR-Content-Ops-FAQ-Sales-Objection-Sources`: add explicit sales-objection source aliases.

## Verification

Local verification:

```bash
pytest tests/test_extracted_ticket_faq_markdown.py -q
# 67 passed

bash scripts/run_extracted_pipeline_checks.sh
# extracted_reasoning_core: 295 passed
# extracted_content_pipeline: 1616 passed, 1 warning
```

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 46 |
| `tests/test_extracted_ticket_faq_markdown.py` | 67 |
| `plans/PR-Content-Ops-FAQ-Zero-Result-Risk.md` | 67 |
| **Total** | **180** |
