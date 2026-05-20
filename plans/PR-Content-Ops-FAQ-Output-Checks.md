# Content Ops FAQ Output Checks

## Why this slice exists

The FAQ Markdown builder now supports ticket CSV ingestion, date windows,
deterministic intent clustering, ticket volume counts, customer-worded
questions, and generated-asset review. The remaining quality gap is the
machine-readable `output_checks` contract: `uses_user_vocabulary` currently
passes for any non-empty topic, and `condensed` passes even when every ticket
becomes its own FAQ item.

That makes the three FAQ output checks too easy to satisfy. This slice makes
the checks honest without changing rendering, storage, or generated-asset API
shape.

## Scope (this PR)

1. Tighten `uses_user_vocabulary` so it reflects customer-worded FAQ questions.
2. Tighten `condensed` so multi-ticket inputs must collapse below the ticket
   count instead of merely staying equal to it.
3. Preserve `has_action_items` as the existing action-step check.
4. Add regression coverage for passing and failing output-check states.
5. Replace the merged stale FAQ customer-wording in-flight row with this active
   slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Output-Checks.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale merged FAQ row with this active slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Tighten FAQ output-check predicates. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for honest output checks. |
| `tests/test_extracted_content_ops_execution_smoke.py` | Keep the FAQ execution smoke aligned with the stricter output checks. |

## Mechanism

`_output_checks(...)` remains the single output-quality summary point. It will
continue returning the same three keys so callers do not need a response-shape
migration:

- `uses_user_vocabulary`: true only when every rendered item used
  `question_source="customer_wording"`.
- `condensed`: true only when at least one FAQ item is generated and either the
  input has one ticket source or multiple ticket sources collapse into fewer
  FAQ items than source tickets.
- `has_action_items`: unchanged, true only when every item has action steps.

## Intentional

- No new output-check keys. The existing API shape stays stable.
- No changes to Markdown rendering. This is telemetry/contract hardening only.
- No LLM or semantic clustering. Deterministic intent clustering remains the
  current source of condensation.

## Deferred

- Semantic clustering beyond keyword rules remains deferred until a real host
  fixture proves deterministic clustering is insufficient.
- Per-item edit/publish UX remains separate from Markdown output checks.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_runs_faq_markdown_json - 36 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1516 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Output-Checks.md` | +77 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +5 / -3 |
| `tests/test_extracted_ticket_faq_markdown.py` | +11 / -4 |
| `tests/test_extracted_content_ops_execution_smoke.py` | +1 / -1 |
| Total | ~106 |
