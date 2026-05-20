# Content Ops FAQ Output Checks

## Why this slice exists

The FAQ Markdown builder now supports ticket CSV ingestion, date windows,
deterministic intent clustering, ticket volume counts, customer-worded
questions, and generated-asset review. The remaining quality gap is the
machine-readable `output_checks` contract: `uses_user_vocabulary` currently
passes for any non-empty topic, and `condensed` passes even when every ticket
becomes its own FAQ item.

That makes the three FAQ output checks too easy to satisfy. This slice makes
the checks honest without changing storage or generated-asset API shape.

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
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | Align rendered review labels with the stricter ticket-source denominator. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for honest output checks. |
| `tests/test_extracted_content_ops_execution_smoke.py` | Keep the FAQ execution smoke aligned with the stricter output checks. |
| `scripts/smoke_extracted_content_ops_execution.py` | Keep the documented FAQ smoke default aligned with the stricter output checks. |

## Mechanism

`_output_checks(...)` remains the single output-quality summary point. It will
continue returning the same three keys so callers do not need a response-shape
migration.

All three checks return `False` when the builder generates no FAQ items (empty
`items` sequence). When items are present:

- `uses_user_vocabulary`: true only when every rendered item used
  `question_source="customer_wording"`.
- `condensed`: true when the input has at most one ticket source, or when
  multiple ticket sources collapse into strictly fewer FAQ items than source
  tickets.
- `has_action_items`: true only when every item has at least one action step.

The ticket-source denominator is based on distinct source keys, not evidence
row count, so multiple snippets from one ticket do not look like multi-ticket
condensation. Rows without a source id use a row-level fallback source key
while evidence-level fallback keys remain available for de-duping repeated
snippets.

`condensed` also requires the rendered FAQ items to cover all ticket sources.
That keeps `max_items` truncation from looking like true condensation when many
unrelated ticket issues are simply dropped from the final Markdown.

The question extractor treats narrow first-person customer issue statements
such as "I cannot reset my password" as customer wording by converting them
into action-oriented FAQ questions using the same terms. This keeps declarative
support tickets from failing `uses_user_vocabulary` just because the customer
did not write the ticket as a question.

## Intentional

- No new output-check keys. The existing API shape stays stable.
- The Markdown summary label changes from evidence rows to ticket sources so
  the rendered copy matches the stricter distinct-source denominator.
- The asset review UI label changes from ticket rows to ticket sources for the
  same reason; the stored field name remains `ticket_source_count`.
- No LLM or semantic clustering. Deterministic intent clustering remains the
  current source of condensation.

## Deferred

- Semantic clustering beyond keyword rules remains deferred until a real host
  fixture proves deterministic clustering is insufficient.
- Per-item edit/publish UX remains separate from Markdown output checks.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_runs_faq_markdown_json - 40 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py scripts/smoke_extracted_content_ops_execution.py - passed
- python scripts/smoke_extracted_content_ops_execution.py --outputs faq_markdown --source-type support_ticket --source-title "login reset" --json >/tmp/faq_smoke.json && python -m json.tool /tmp/faq_smoke.json >/dev/null - passed
- npx eslint src/pages/ContentOpsAssetsReview.tsx (from atlas-intel-ui) - passed
- npm run build (from atlas-intel-ui, after npm ci) - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1520 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Output-Checks.md` | +102 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +69 / -9 |
| `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx` | +2 / -2 |
| `scripts/smoke_extracted_content_ops_execution.py` | +1 / -1 |
| `tests/test_extracted_content_ops_execution_smoke.py` | +0 / -2 |
| `tests/test_extracted_ticket_faq_markdown.py` | +96 / -7 |
| Total | ~295 |
