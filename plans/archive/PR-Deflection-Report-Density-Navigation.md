# PR-Deflection-Report-Density-Navigation

## Why this slice exists

Issue #1579 flags the buyer-facing paid deflection report as too dense to
review: the same ranked question is rendered across multiple sections, the
no-proven-answer boilerplate repeats per question, and the complete evidence
appendix adds another full per-question pass. After #1574 proved the paid
result page renders under real hosted conditions, report readability is the
next product-quality blocker.

The safest upstream fix is in the deterministic report renderer, not in the PDF
or portfolio presentation layer. This slice preserves the paid-report
completeness promise while reducing repetition: each ranked question gets one
canonical detail block containing status, publishable copy or gap guidance,
vocabulary mappings, and complete evidence.

This slice is expected to exceed the 400-line soft cap because the renderer
contract is producer-bound: frontend contract examples and committed validation
proof excerpts must be regenerated in the same PR so docs/tests keep matching
real output instead of stale hand-edited shapes.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Product polish

1. Replace the separate publishable-answer, no-proven-answer, vocabulary-gap,
   and evidence-appendix per-question passes with one canonical
   `Question Details and Evidence` section.
2. Keep the ranked table and SEO targeting list as scannable indexes, but make
   the full detail/evidence content appear once per question.
3. Update the Support Tax promise language so it still promises complete
   evidence, but describes where that evidence now lives.
4. Add regression coverage proving complete source IDs/quotes are preserved and
   repeated no-proven boilerplate is not multiplied per question.
5. Archive this session's just-merged #1578 and #1574 plan docs while this
   branch is already touching plans.

### Files touched

- `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md`
- `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md`
- `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Report-Density-Navigation.md`
- `plans/archive/PR-Deflection-Portfolio-Paid-Result-Live-Proof.md`
- `plans/archive/PR-Deflection-Request-ID-Redaction.md`
- `scripts/evaluate_zendesk_product_proof_corpus.py`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_deflection_resolution_live_proof.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_evaluate_zendesk_product_proof_corpus.py`
- `tests/test_extracted_content_ops_execution.py`

### Review Contract

Acceptance criteria:

- Every ranked question still appears in the paid report.
- Full source IDs and evidence quotes remain present for each question; this PR
  must not cap or drop the evidence appendix content.
- Publishable answer text and steps still render for
  `answer_evidence_status == resolution_evidence`.
- Draft-needs-review items still render clear no-proven-answer guidance, but
  the repeated explanatory boilerplate appears once at the section level.
- The report no longer renders separate per-question passes for publishable
  answers, no-proven answers, vocabulary gaps, and evidence appendix.
- Merged plan docs for #1578 and #1574 are moved to `plans/archive/` and the
  plan index is refreshed.

Affected surfaces:

- `extracted_content_pipeline/faq_deflection_report.py`
- `tests/test_content_ops_deflection_report.py`
- `plans/**`

Risk areas:

- Accidentally dropping source IDs or evidence quotes while moving the appendix.
- Weakening the buyer-facing completeness promise.
- Breaking downstream PDF/report rendering by introducing HTML-only navigation.
- Free snapshot leakage is not in scope; this slice touches only paid Markdown.

Reviewer rules triggered: R1, R2, R6, R10, R13, R14.

## Mechanism

`render_deflection_report` will keep the top-level summary/index sections:

1. `Support Tax Confirmation`
2. `Your Help-Desk SEO Targeting List`
3. `Ranked Question Opportunities`
4. `Resolution Outcome Diagnostics` when present

Then it will render one canonical `Question Details and Evidence` section over
the original ranked `items` tuple. For each item, the block emits:

- the answer status and support-cost/ticket-count context;
- publishable answer + steps when resolution evidence exists;
- concise no-proven-answer guidance when it does not;
- any vocabulary mappings for that question; and
- the complete source-ID list and every evidence quote.

This removes the downstream symptom of four repeated per-question sections
without moving evidence out of the paid report or relying on portfolio/PDF
rendering changes.

## Intentional

- No evidence quote/source-ID caps in this slice. Capping or moving full
  evidence into an attachment changes the buyer-facing completeness promise and
  needs operator sign-off from #1579.
- No HTML anchors or browser-only navigation. The current portfolio page renders
  paid Markdown inside a `<pre>`, and the PDF renderer is plain Markdown-to-PDF;
  HTML anchors would not reliably help both surfaces.
- The SEO targeting list remains uncapped. It is still an issue for very large
  reports, but capping that list is a separate product-shape decision.

## Deferred

- #1579 follow-up: decide whether to cap inline evidence/SEO or move complete
  evidence to an attached CSV/JSONL while preserving or changing the paid-report
  promise.
- #1579 follow-up: add true clickable browser navigation once the hosted
  portfolio report renders Markdown as structured HTML instead of escaped
  `<pre>` text.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py -q -- 41
  passed.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_extracted_content_ops_execution.py tests/test_content_ops_deflection_resolution_live_proof.py tests/test_content_ops_deflection_report.py -q
  -- 127 passed, 1 warning.
- Command: python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_pdf_renderer.py tests/test_extracted_content_deflection_submit.py -q
  -- 79 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_resolution_live_proof.py
  -- passed.
- Command: git diff --check -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- 4298 passed, 10 skipped,
  1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_resolution_evidence_live_proof_2026-06-09.md` | 8 |
| `docs/extraction/validation/fixtures/deflection_resolution_evidence_live_proof_20260609/report.md` | 88 |
| `docs/extraction/validation/fixtures/deflection_zendesk_product_proof_eval_20260614/report_excerpt.md` | 46 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 2 |
| `docs/frontend/content_ops_faq_report_contract.md` | 9 |
| `extracted_content_pipeline/faq_deflection_report.py` | 197 |
| `plans/INDEX.md` | 4 |
| `plans/PR-Deflection-Report-Density-Navigation.md` | 175 |
| `plans/archive/PR-Deflection-Portfolio-Paid-Result-Live-Proof.md` | 0 |
| `plans/archive/PR-Deflection-Request-ID-Redaction.md` | 0 |
| `scripts/evaluate_zendesk_product_proof_corpus.py` | 18 |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | 6 |
| `tests/test_content_ops_deflection_report.py` | 83 |
| `tests/test_content_ops_deflection_resolution_live_proof.py` | 25 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 5 |
| `tests/test_evaluate_zendesk_product_proof_corpus.py` | 31 |
| `tests/test_extracted_content_ops_execution.py` | 13 |
| **Total** | **710** |
