# PR-Content-Ops-FAQ-Scale-Run-Summary-Diagnostics

## Why this slice exists

The FAQ CLI now reports upload source mix, weighted represented volume, per-item
source mix, output-check details, and compact item diagnostics. Those fields are
useful individually, but a large run still requires stitching together several
JSON paths to answer the operator's first question: did this upload produce a
healthy FAQ run, and what volume and score shape did it cover?

This slice adds a compact run-summary block that combines output-check status,
source-mix volume, generated count, and item score distribution without changing
FAQ generation behavior.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

1. Add a `diagnostics.run_summary` block to the FAQ CLI compact result JSON.
2. Include output-check pass/fail counts and failed-check names.
3. Include upload volume summary from existing source-mix diagnostics.
4. Include item opportunity-score distribution for large-run triage.
5. Add focused CLI regression coverage for success and fail-closed result
   summaries.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Scale-Run-Summary-Diagnostics.md` | Plan doc for this run-summary diagnostics slice. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Adds compact run-summary diagnostics derived from existing result fields. |
| `tests/test_extracted_ticket_faq_markdown.py` | Covers run-summary diagnostics for passing and failing output-check runs. |

## Mechanism

`_result_payload` will compute source-mix diagnostics once, then pass that
dictionary plus the generated item summaries, output checks, failed checks, and
warnings into a new private `_run_summary` helper.

The summary is intentionally compact and structured: status, generated/source
counts, weighted source volume, source-channel counts, output-check counts,
failed-check names, warning count, and item score distribution. Score
distribution uses existing item `opportunity_score` values and reports min,
max, average, and band counts so a 1,000-row run can be triaged without opening
the Markdown body.

## Intentional

- CLI result JSON only. Markdown, ranking, source adapters, and library result
  objects remain unchanged.
- The summary duplicates only small aggregate values already present elsewhere
  in diagnostics; it does not duplicate item bodies, answers, steps, or the
  Markdown artifact.
- Score bands are broad triage buckets, not product scoring semantics.

## Deferred

- Hosted UI display for the run-summary block remains separate.
- Scale-smoke wrapper display can consume this block in a follow-up without
  changing the FAQ CLI schema again.

## Verification

- Focused run-summary/source-mix/output-check FAQ CLI pytest - passed, 3 tests.
- Full FAQ pytest for `tests/test_extracted_ticket_faq_markdown.py` - passed,
  131 tests.
- Py compile for affected Python files - passed.
- Git whitespace check - passed.
- Extracted manifest/import validation script - passed.
- Extracted reasoning import guard - passed.
- Extracted standalone audit - passed, 0 Atlas runtime import findings.
- Extracted ASCII Python check - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| CLI run-summary diagnostics | ~90 |
| Tests | ~55 |
| **Total** | ~220 |
