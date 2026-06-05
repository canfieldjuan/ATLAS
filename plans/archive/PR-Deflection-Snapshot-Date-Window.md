## Why this slice exists

The portfolio #196 redesign now sizes Support Tax from real repeat-ticket counts, but its remaining period-honesty gap is that the results page cannot normalize the projection to the upload's actual source date window. ATLAS already parses support-ticket date fields for explicit `window_days` filtering, but the deflection artifact and free snapshot do not preserve a safe aggregate date span for the frontend to consume. This slice exposes only fail-closed aggregate date-window fields, so later portfolio copy can say "at the rate in your data" without assuming a monthly batch.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-support-ticket-deflection

Slice phase: Production hardening

1. Preserve aggregate source-date coverage on generated FAQ deflection items without exposing source IDs in the free snapshot.
2. Add optional `source_date_start`, `source_date_end`, and `source_window_days` fields to `DeflectionSnapshot.summary` only when every contributing ticket source has a parseable date.
3. Keep date-window fields absent when source dates are missing or partial, so frontend normalized projections fail closed.
4. Update the frontend contract docs, generated examples, and focused tests for the new optional summary fields.

### Files touched

- `plans/PR-Deflection-Snapshot-Date-Window.md` - plan doc for this slice.
- `extracted_content_pipeline/ticket_faq_markdown.py` - carry per-item aggregate source-date coverage from contributing rows.
- `extracted_content_pipeline/faq_deflection_report.py` - expose complete date-window fields in report and snapshot summaries.
- `docs/frontend/content_ops_faq_report_contract.md` - document the optional fail-closed snapshot summary fields.
- `docs/frontend/content_ops_faq_deflection_checkout_contract.md` - tell the portfolio to use date-window fields only when present.
- `docs/frontend/content_ops_faq_deflection_report_example.json` - refreshed producer example.
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json` - refreshed snapshot example.
- `tests/test_content_ops_deflection_report.py` - focused summary/snapshot date-window tests.
- `tests/test_content_ops_faq_report_contract_docs.py` - contract-shape assertions for the optional fields.
- `tests/test_extracted_content_ops_live_execute_harness.py` - execute-path snapshot expectation.

## Mechanism

`ticket_faq_markdown` already knows how to parse source dates from the accepted support-ticket aliases. This PR carries an ISO `source_date` on each normalized row that enters a FAQ item, then records an item-level `source_date_span` summary with start, end, inclusive day count, dated source count, and missing source count.

`faq_deflection_report` folds item-level spans into report and snapshot summaries. The public snapshot only gets `source_date_start`, `source_date_end`, and `source_window_days` when the aggregate span is complete. If any contributing source lacks a parseable date, those fields are omitted instead of partially normalizing the upload.

## Intentional

- The free snapshot still strips Markdown, source IDs, evidence, answers outside the bounded teaser, and locked question text.
- Date fields are aggregate and optional. Missing/partial dates do not produce fallback windows or synthetic assumptions.
- This does not change grouping, ranking, answer drafting, paid unlock, Stripe, or portfolio rendering.

## Deferred

- Portfolio consumption is deferred to the follow-up atlas-portfolio slice after this contract lands and deploys.
- Live production verification is deferred until ATLAS main is redeployed and a fresh report is generated from a dated CSV.

Parked hardening: none.

## Verification

- Command: pytest tests/test_content_ops_deflection_report.py -q - passed, 32 tests.
- Command: pytest tests/test_content_ops_faq_report_contract_docs.py -q - passed, 5 tests.
- Command: pytest tests/test_extracted_content_ops_live_execute_harness.py -q - passed, 13 tests.
- Command: bash scripts/validate_extracted_content_pipeline.sh - passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt - passed.
- Command: bash scripts/check_ascii_python.sh - passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh - passed, 2980 passed / 10 skipped.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas-pr-bodies/deflection-snapshot-date-window.md - passed.

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Deflection-Snapshot-Date-Window.md` | ~65 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | ~60 |
| `extracted_content_pipeline/faq_deflection_report.py` | ~65 |
| `docs/frontend/content_ops_faq_report_contract.md` | ~20 |
| `docs/frontend/content_ops_faq_deflection_checkout_contract.md` | ~15 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | ~25 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | ~15 |
| `tests/test_content_ops_deflection_report.py` | ~70 |
| `tests/test_content_ops_faq_report_contract_docs.py` | ~20 |
| `tests/test_extracted_content_ops_live_execute_harness.py` | ~10 |
| Total | ~365 |
