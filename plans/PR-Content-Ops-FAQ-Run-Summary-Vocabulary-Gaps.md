# PR-Content-Ops-FAQ-Run-Summary-Vocabulary-Gaps

## Why this slice exists

The FAQ generator now detects vocabulary gaps and writes full term-mapping
diagnostics, but the compact run summary used for large-upload triage does not
say whether any mappings were found. Operators running 500 or 1,000 rows can
see upload density, output-check status, and score shape at a glance, but still
have to dig into nested diagnostics to answer whether customer language differed
from documentation language.

This slice adds a small vocabulary-gap rollup to the existing FAQ run summary so
the real CLI and scale-smoke artifacts expose the signal without adding a new
generation path.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add compact vocabulary-gap counts and top customer terms to
   `diagnostics.run_summary`.
2. Keep the detailed `diagnostics.term_mappings` list unchanged.
3. Refresh the checked-in scale-smoke examples so they match the new guarded
   run-summary schema.
4. Extend focused tests for CLI output and checked-in examples.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Run-Summary-Vocabulary-Gaps.md` | Plan doc for this visibility slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds vocabulary-gap rollup to compact FAQ run summary. |
| `extracted_content_pipeline/examples/faq_scale_density_limited_summary.json` | Refreshes static example run-summary shape. |
| `extracted_content_pipeline/examples/faq_scale_output_check_failure_summary.json` | Refreshes static example run-summary shape. |
| `extracted_content_pipeline/README.md` | Notes that run-summary now includes vocabulary-gap visibility. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers vocabulary-gap summary values in CLI result JSON. |
| `tests/test_smoke_content_ops_faq_scale_run.py` | Pins refreshed example vocabulary-gap summary values. |

## Mechanism

`_result_payload` already builds sorted compact term-mapping diagnostics. This
slice computes those summaries once, passes them into `_run_summary`, and adds a
small nested `vocabulary_gaps` block:

```json
{
  "term_mapping_count": 2,
  "mapped_topic_count": 2,
  "zero_result_mapping_count": 1,
  "max_opportunity_score": 78,
  "top_customer_terms": ["export", "bill"]
}
```

The run summary stays compact and deterministic. It does not duplicate full
mapping bodies or suggestions; those remain in `diagnostics.term_mappings`.

## Intentional

- CLI/result visibility only. FAQ generation, ranking, Markdown body text, and
  output-check rules do not change.
- `top_customer_terms` is capped to three values because this summary is for
  triage, not a full report.
- The static examples remain representative artifacts; the schema guard added
  in the prior slice ensures their shape tracks the runtime summary.

## Deferred

- Hosted UI display for the vocabulary-gap run-summary block remains separate.
- CSV rule-file import and explicit documentation-term format overrides remain
  separate CLI slices.
- Parked hardening for this slice: none. The hardening tracker has no entries
  for this lane yet.

## Verification

- Focused FAQ CLI and scale-smoke pytest - passed, 4 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` and
  `tests/test_smoke_content_ops_faq_scale_run.py` - passed, 149 tests.
- JSON parse check for both refreshed scale-smoke examples - passed.
- Py compile for affected Python files - passed.
- CLI demo with packaged support tickets, documentation terms, and custom rules
  - passed. The emitted `diagnostics.run_summary.vocabulary_gaps` was:
  `term_mapping_count=2`, `mapped_topic_count=1`, `max_opportunity_score=4`,
  `top_customer_terms=["export", "reporting"]`.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.
- Full extracted pipeline checks - passed, 1,755 tests, 1 existing torch/pynvml
  warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~85 |
| CLI run-summary payload | ~35 |
| Example JSON | ~30 |
| README | ~5 |
| Tests | ~35 |
| **Total** | ~190 |
