# PR-Deflection-Paid-Report-Reframe

## Why this slice exists

The live free snapshot now sells the upgraded deflection story: Support Tax cost
framing, source-window run-rate, locked-row FOMO, a teaser answer, and a
help-desk SEO targeting list. The paid artifact is functionally correct, but it
still opens as a clinical generated-report summary and dumps large inline
source-ID walls in the reading flow.

That creates a promise-delivery mismatch for buyers: they unlock a report after
seeing cost and SEO framing, then receive a paid deliverable that buries the
same value in neutral tables. This slice updates the paid Markdown renderer so
the full artifact matches what the snapshot sells while keeping the artifact's
traceability and fail-closed evidence doctrine intact.

This is expected to land slightly over the 400 LOC soft budget because the
customer-facing renderer, contract doc, generated example payload, and focused
regression tests need to move together. Splitting those would leave either the
docs stale or the live paid-report mismatch under-tested.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-paid-report
Slice phase: Product polish

1. Reframe the paid report opening around Support Tax using only raw measured
   ticket counts and the existing Gartner `$13.50` assisted-contact benchmark.
2. Add a named "Your Help-Desk SEO Targeting List" section from existing
   customer wording and vocabulary-gap data, with no search-volume, ranking, or
   traffic promise.
3. Add per-question estimated support cost to the ranked table using
   `ticket_count * 13.50`.
4. Rename the drafted-answer section around publishable help-center copy.
5. Replace inline full source-ID walls with compact backing summaries in the
   drafted and no-proven sections.
6. Preserve full per-question source IDs in the Evidence Appendix alongside
   evidence quotes.

### Files touched

- `plans/PR-Deflection-Paid-Report-Reframe.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `docs/frontend/content_ops_faq_report_contract.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_content_ops_execution.py`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_mcp_content_ops_deflection_readonly.py`

## Mechanism

`render_deflection_report(...)` continues to render from
`TicketFAQMarkdownResult` only. The report-level Support Tax block derives:

```text
repeat ticket count = sum(item.ticket_count, fallback source_ids count)
batch cost = repeat ticket count * 13.50
annualized cost = batch cost / source_window_days * 365
```

The annualized line appears only when `source_date_start`, `source_date_end`,
and `source_window_days` passed the existing completeness checks. If that
window is unavailable, the report uses the snapshot's unknown-window posture:
the uploaded batch cost is shown directly, and any 12-month number is framed as
conditional monthly pace rather than inferred reporting-period truth.

The reading sections call a shared source-summary helper such as
`Backed by 35 resolved tickets (ticket-1, ticket-2, ticket-3, +32 more).`
The full `source_ids` sequence remains available in the Evidence Appendix for
auditability.

## Intentional

- This is a Markdown renderer change only. It does not alter grouping,
  ranking, answer generation, paid unlock, Stripe, storage, or snapshot shape.
- The cost estimate uses assisted-contact cost, not claimed savings. It is
  labeled as an estimate and tells the buyer to adjust for their own loaded
  support cost.
- The SEO section frames customer wording as help-center headings, internal
  search synonyms, and FAQ wording inputs. It intentionally avoids keyword
  volume, Google rank, organic traffic, or guaranteed deflection claims.
- Full source IDs are relocated, not removed, so every paid answer remains
  traceable to real uploaded tickets.

## Deferred

- Rich paid-report visual styling belongs to the portfolio renderer after the
  Markdown contract is corrected.
- Custom buyer cost inputs in paid Markdown are deferred because the ATLAS
  artifact is static; the artifact uses the shared `$13.50` benchmark and names
  the adjustment path.
- Parked hardening: none. `HARDENING.md` has no current entry touching this
  ownership lane or renderer.

## Verification

- `python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_extracted_content_ops_execution.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py` - passed.
- `pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_extracted_content_ops_execution.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` - passed, 100 passed, 1 warning.
- `pytest tests/test_mcp_content_ops_deflection_readonly.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_content_ops_deflection_report.py -q` - passed, 48 passed, 1 warning.
- `pytest tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_mcp_content_ops_deflection_readonly.py -q` - passed, 80 passed, 1 warning.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, extracted_reasoning_core 295 passed; extracted_content_pipeline 2990 passed, 10 skipped, 1 warning.
- `python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/saas-deflection-large-sample.csv --source-format csv --output /tmp/deflection-paid-reframe-420.md --summary-output /tmp/deflection-paid-reframe-420-summary.json --result-output /tmp/deflection-paid-reframe-420-result.json --json` - passed, 420 sources, 18 generated, 18 drafted, source window 2026-04-06 to 2026-05-29.
- `bash scripts/local_pr_review.sh` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~110 |
| Paid report renderer helpers | ~230 |
| Contract docs/example | ~10 |
| Focused renderer/contract/caller tests | ~160 |
| **Total** | **~510** |
