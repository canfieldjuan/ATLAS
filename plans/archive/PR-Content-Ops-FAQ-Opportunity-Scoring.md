# PR-Content-Ops-FAQ-Opportunity-Scoring

## Why this slice exists

The FAQ generator already ingests support-ticket-like evidence and clusters rows into FAQ topics, but the current output is only frequency-ranked. The product wedge needs explicit intent-to-FAQ opportunity mapping: each generated FAQ opportunity should expose frequency, failure-risk signals, and an opportunity score so operators can tell which customer intents should be fixed first.

This is not blocked by vocabulary-gap detection. Vocabulary gaps need a docs/search corpus later; this slice strengthens the existing ticket FAQ path first so later term mapping can attach to the same opportunity summary.

## Scope (this PR)

1. Add deterministic failure-risk scoring to each generated FAQ item.
2. Rank FAQ items by opportunity score, using source frequency and risk as the primary sort.
3. Expose compact scoring fields in the CLI result JSON item summaries.
4. Add focused regression tests for scoring, ranking, and CLI diagnostics.

### Files touched

| File | Intent |
|---|---|
| `extracted_content_pipeline/ticket_faq_markdown.py` | Compute risk signals and opportunity score per FAQ item and use that score for ordering. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Include compact scoring fields in result-output item summaries. |
| `tests/test_extracted_ticket_faq_markdown.py` | Cover opportunity-score ranking and result JSON diagnostics. |
| `plans/PR-Content-Ops-FAQ-Opportunity-Scoring.md` | Plan contract for this slice. |

## Mechanism

The builder keeps the existing grouping and rendering pipeline. After groups are formed, each group is converted to a scored group summary:

```python
frequency = distinct source count for the topic
failure_risk_score = deterministic count of blocked / failed / wrong / unable-style language
opportunity_score = frequency * (1 + failure_risk_score)
```

The generated FAQ item keeps the existing fields and adds:

- `frequency`
- `failure_risk_score`
- `failure_risk_signals`
- `opportunity_score`

The CLI result JSON mirrors only those compact fields; it does not duplicate Markdown body text.

## Intentional

- The risk score is deterministic keyword-based, not an LLM classifier. That keeps this slice small, testable, and provider-free.
- The Markdown body stays focused on customer-facing FAQ content. Score numbers live in item metadata and CLI result JSON, while item order intentionally follows opportunity score.
- Vocabulary-gap detection is deferred because it requires docs/search-log inputs that are separate from the current ticket FAQ generator.

## Deferred

- `PR-Content-Ops-FAQ-Search-Log-Sources`: add first-class search-log and zero-result query source rows.
- `PR-Content-Ops-FAQ-Vocabulary-Gaps`: compare customer language against documentation terms and emit term-mapping suggestions.
- `PR-Content-Ops-FAQ-Sales-Objection-Sources`: add explicit sales-objection source aliases once the opportunity summary can consume them.

## Verification

Local verification:

```bash
pytest tests/test_extracted_ticket_faq_markdown.py -q
# 65 passed

bash scripts/run_extracted_pipeline_checks.sh
# extracted_reasoning_core: 295 passed
# extracted_content_pipeline: 1599 passed, 1 warning

bash scripts/local_pr_review.sh origin/main
# passed
```

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 56 |
| `scripts/build_extracted_ticket_faq_markdown.py` | 4 |
| `tests/test_extracted_ticket_faq_markdown.py` | 156 |
| `plans/PR-Content-Ops-FAQ-Opportunity-Scoring.md` | 80 |
| **Total** | **296** |
