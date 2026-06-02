# PR-Deflection-Uncap-Paid-Report

## Why this slice exists

Issue #1265 identifies a product-truth gap in the paid FAQ deflection report: the offer says the paid report lists every recurring question, but the report build still uses the generic FAQ Markdown `max_items` cap. When distinct questions exceed that cap, the tail is collapsed into a single overflow item, and the free snapshot inherits capped totals. This slice makes the paid report uncapped so the full artifact and snapshot summary counts reflect the true ranked question set.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection
Slice phase: Production hardening

1. Add a real unlimited mode to the FAQ Markdown grouping cap without losing existing finite-cap overflow behavior.
2. Make the customer-facing deflection report use unlimited mode for the paid artifact.
3. Align the local deflection report CLI with the same uncapped behavior so verification does not show stale capped output.
4. Preserve the free snapshot projection: top questions remain governed by `deflection_snapshot_top_n`, and paid answer/evidence fields remain absent from the snapshot.
5. Add regression coverage for uncapped generation, truthful snapshot counts, script parity, and finite-cap compatibility.

### Files touched

- `plans/PR-Deflection-Uncap-Paid-Report.md` - Plan doc for this slice.
- `.github/workflows/atlas_content_ops_deflection_report_checks.yml` - Dedicated atlas workflow enrollment for the atlas_brain-importing harness test.
- `extracted_content_pipeline/ticket_faq_markdown.py` - Add unlimited max-items normalization while preserving finite caps.
- `extracted_content_pipeline/faq_deflection_report.py` - Force the paid deflection artifact through uncapped FAQ Markdown generation.
- `scripts/build_content_ops_deflection_report.py` - Force the local deflection report CLI through uncapped FAQ Markdown generation.
- `tests/test_extracted_ticket_faq_markdown.py` - Pin unlimited sentinel and finite-cap compatibility.
- `tests/test_extracted_content_ops_live_execute_harness.py` - Prove the paid artifact is uncapped while the free snapshot remains top-N.
- `tests/test_content_ops_deflection_report.py` - Prove the CLI cannot verify the deflection report with a stale finite cap.

## Mechanism

`build_ticket_faq_markdown(..., max_items=0)` becomes the explicit unlimited sentinel. Positive caps keep the existing overflow condensation behavior, and negative values remain invalid. `TicketFAQMarkdownService` normalizes explicit/configured values through the same helper so a configured `0` or `None` can mean uncapped.

`FAQDeflectionReportService.generate()` passes the unlimited sentinel to the FAQ Markdown service instead of forwarding the execution/request cap. That keeps upload row limits separate from paid-report display limits: row ingestion may still be bounded, but every distinct recurring question found in the ingested rows is listed in the paid artifact.

`scripts/build_content_ops_deflection_report.py` uses the same unlimited sentinel for local artifacts. Its legacy `--max-items` flag is retained for compatibility and recorded in result JSON as `requested_max_items`, but it no longer caps the deflection report display.

The snapshot already slices the artifact items by `top_n` and strips paid fields. Once the artifact is uncapped, the deflection report summary remains truthful because it counts the full item tuple.

## Intentional

- Generic finite-cap FAQ Markdown output is retained for non-deflection callers and for existing tests that prove overflow condensation.
- The deflection report intentionally ignores `max_items` as a display cap. In this product lane, request limits govern input volume, not how many paid report questions are shown.
- The deflection CLI keeps `--max-items` as a compatibility flag but records it as requested-only because capped local artifacts are now misleading for this product.
- No separate `questions_found` summary field is added because the uncapped paid artifact makes `generated`, drafted, and no-proven counts truthful.

## Deferred

- Frontend copy or layout changes for larger paid reports are deferred to the atlas-portfolio snapshot/results work tracked separately.
- Live redeploy and customer re-verification are operational follow-ups after this PR merges.

Parked hardening: none.

## Verification

- Command: pytest tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_skips_non_ticket_sources_and_validates_limits tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_treats_zero_max_items_as_unlimited tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_condenses_overflow_sources_instead_of_truncating tests/test_extracted_content_ops_live_execute_harness.py::test_deflection_report_execute_uncaps_paid_artifact_and_keeps_snapshot_top_n tests/test_content_ops_deflection_report.py::test_deflection_report_cli_ignores_legacy_max_items_cap -q -- 5 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- passed; 2961 passed, 10 skipped, 1 warning in the extracted content pipeline pytest suite, plus extracted reasoning core checks passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/deflection-uncap-paid-report-pr-body.md -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Workflow enrollment | 52 |
| Product code | 56 |
| Plan doc | 72 |
| Tests | 218 |
| **Total** | **398** |
