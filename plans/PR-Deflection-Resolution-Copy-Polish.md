## Why this slice exists

The large SaaS-only validation proved the paid report can rank recurring
questions and produce resolution-backed drafts, and
`PR-Deflection-Question-Evidence-Scoping` fixed the unsafe cross-resolution
bleed. The remaining buyer-visible issue is copy quality: resolution-backed
steps expose internal scaffolding such as "Use the uploaded resolution
evidence:" instead of clean help-center prose. The full paid report page and
macro writeback both render those steps directly, so this polish belongs in the
backend draft generator.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Product polish

1. Remove internal evidence-scaffold copy from resolution-backed FAQ steps.
2. Preserve fail-closed behavior for review-needed drafts; only
   `resolution_evidence` steps are polished.
3. Update report and macro-writeback expectations so buyer-visible surfaces no
   longer contain "Use the uploaded resolution evidence:".
4. Refresh the frontend contract examples so docs fixtures match the polished
   paid-report payload.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_example.json`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_extracted_ticket_faq_macro_writeback.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_output_ingestion.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Resolution-Copy-Polish.md`

## Mechanism

The resolution step builder in
`extracted_content_pipeline/ticket_faq_markdown.py` already extracts the first
sentence from each trusted `resolution_text`. This slice stops wrapping that
trusted excerpt in internal provenance wording. A resolved step becomes:

```text
Open Reports, choose Export, then select CSV
```

instead of:

```text
Use the uploaded resolution evidence: Open Reports, choose Export, then select CSV
```

The evidence status, source IDs, resolution source count, and review-needed
fallback stay unchanged.

The frontend example payloads are regenerated from the producer fixtures after
the generator change, and their contract tests now assert that the public JSON
examples do not contain the internal scaffold phrase.

## Intentional

- This does not rewrite or synthesize new product instructions. It only removes
  wrapper text around already-scoped `resolution_text` excerpts.
- This does not change the `answer` field yet. The highest-friction buyer
  surface is the ordered step list used by the paid page, Markdown report, and
  macro writeback. A later slice can improve answer summaries separately.
- This does not change overflow/yield behavior from #1235.

## Deferred

- Follow-up slice: improve `answer` summaries so the paid card paragraph no
  longer starts with "Customers mention:".
- Follow-up slice: add the stricter artifact/eval gate that fails if an item
  cites evidence outside its evidence scope.
- Considered hardening: the existing `HARDENING.md` FAQ UI dependency-audit
  entry is unrelated to this backend copy surface and remains parked.
- Parked hardening: none.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py -k "resolution_evidence" -q` -- 2 passed, 149 deselected.
- `pytest tests/test_extracted_ticket_faq_macro_writeback.py::test_macro_writeback_preview_uses_real_faq_generator_resolution_steps tests/test_content_ops_deflection_report.py::test_deflection_report_partitions_proven_and_unproven_answers tests/test_extracted_ticket_faq_output_ingestion.py::test_source_material_to_source_rows_accepts_ticket_faq_result_dict tests/test_content_ops_faq_report_contract_docs.py -q` -- 8 passed.
- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_ticket_faq_macro_writeback.py tests/test_content_ops_deflection_report.py tests/test_extracted_ticket_faq_output_ingestion.py tests/test_content_ops_faq_report_contract_docs.py -q` -- 189 passed.
- `python scripts/build_content_ops_deflection_report.py /home/juan-canfield/Desktop/saas-deflection-large-sample.csv --source-format csv --max-items 8 --result-output /tmp/deflection-large-saas-copy-polish.json --summary-output /tmp/deflection-large-saas-copy-polish-summary.json --require-output-checks --json` -- passed; generated 8 opportunities, 7 proven drafts, 1 review-needed bucket.
- `rg -n "Use the uploaded resolution evidence" /tmp/deflection-large-saas-copy-polish.json /tmp/deflection-large-saas-copy-polish-summary.json` -- no matches.
- `bash scripts/validate_extracted_content_pipeline.sh` -- passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -- passed.
- `bash scripts/check_ascii_python.sh` -- passed.
- `ATLAS_API_BASE_URL= ATLAS_B2B_JWT= ATLAS_TOKEN= ATLAS_ACCOUNT_ID= ATLAS_FAQ_SEARCH_ACCOUNT_ID= ATLAS_DEFLECTION_SUBMIT_BLOB_URL= ATLAS_DEFLECTION_SUBMIT_CSV_FILE= ATLAS_DEFLECTION_COMPANY_NAME= ATLAS_DEFLECTION_CONTACT_EMAIL= ATLAS_DEFLECTION_SUPPORT_PLATFORM= ATLAS_DEFLECTION_PORTFOLIO_RESULT_URL= ATLAS_DEFLECTION_REQUEST_ID= bash scripts/run_extracted_pipeline_checks.sh` -- local checkout caveat: 1 unrelated SaaS demo preflight subprocess test failed because it reloads this repo's `.env`; 2886 passed, 10 skipped.
- `bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file .git/pr-deflection-resolution-copy-polish-body.md` -- passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | ~5 |
| Tests | ~30 |
| Frontend JSON examples | ~312 |
| Plan doc | ~108 |
| **Total** | **~452** |

Over the 400 LOC soft cap because the contract examples are regenerated JSON
fixtures. The behavioral diff is intentionally small: one generator line plus
focused expectations that keep buyer-visible payloads free of internal
scaffold copy.
