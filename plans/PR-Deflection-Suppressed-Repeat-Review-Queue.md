# PR-Deflection-Suppressed-Repeat-Review-Queue

## Why this slice exists

atlas-portfolio#324 asks the deflection report to stop making repeated
questions disappear silently when they are expensive but not safe to turn into a
publishable answer. The ATLAS child tracker is canfieldjuan/ATLAS#1829. The
existing report already surfaces the two valuable ticket-derived action lanes:

- unresolved repeated questions with no proven answer (`top_unresolved_repeats`);
- proven answers that still recur with pain (`already_covered_still_recurring`).

The remaining gap is the audit trail for rows that are intentionally kept out of
repeat accounting because the cluster is too sparse, has no question text, or
does not have enough source support. Without that paid-only review queue, a high
cost-looking but low-confidence row can be excluded from the customer-facing
action sections without saying why.

Root cause: the action classifier already labels these rows as `Low confidence`,
but the paid structured report has no dedicated section that preserves the
suppression reason for buyer/operator review.

The synced diff is over the 400-LOC soft cap because the slice updates the
backend contract, producer-backed frontend example, generated frontend
contracts, smoke fixtures, the plan doc, and direct runtime/projection
regression tests together. Splitting those would create a contract/docs drift
window for the exact behavior this PR is meant to prove.

## Scope (this PR)

Ownership lane: deflection/report-model
Slice phase: Functional validation

1. Add a `suppressed_repeat_review_queue` paid structured-report section that
   captures low-confidence action rows with deterministic reason codes.
2. Keep `top_unresolved_repeats`, `drafted_resolutions`, and
   `already_covered_still_recurring` semantics unchanged.
3. Update the backend-owned frontend contract docs/example so portfolio can
   consume the new section in a later slice.
4. Add focused tests proving sparse/one-off hidden rows get review reasons while
   unresolved and recurring-but-covered rows remain surfaced in their existing
   lanes.

### Review Contract
- Acceptance criteria:
  - [ ] Low-confidence rows excluded from `top_unresolved_repeats` appear in
        `suppressed_repeat_review_queue.items`.
  - [ ] Each suppressed queue row carries a deterministic
        `suppression_reason` and human-readable `suppression_reason_label`.
  - [ ] Reason codes cover `missing_question`, `too_low_volume`,
        and `insufficient_source_support`.
  - [ ] `already_covered_still_recurring` rows are not suppressed or hidden.
  - [ ] The Snapshot projection remains unchanged and does not expose this
        paid audit queue.
  - [ ] Contract docs and example JSON match the producer output.
- Affected surfaces: extracted content pipeline report model, frontend contract
  docs/example, report model tests, hosted smoke fixture.
- Risk areas: backcompat, data truthfulness, frontend contract drift.
- Reviewer rules triggered: R1, R2, R5, R10, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Suppressed-Repeat-Review-Queue.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_generate_deflection_frontend_contract_types.py`
- `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`
- `tests/test_smoke_content_ops_deflection_pdf_export_validators.py`

## Mechanism

The existing report-model builder already builds action rows and labels
hidden/noisy candidates as `Low confidence`. This slice adds a separate
suppressed-repeat review data pass over the same ranked FAQ items. It builds the
action row, filters only rows whose status is `Low confidence`, derives a reason
from the original ticket aggregate, and returns a bounded paid section:

- `items`: action-row fields plus `suppression_reason` and
  `suppression_reason_label`;
- `total_item_count`: count before the default display cap;
- `default_limit`: the same bounded backlog cap used by the broader paid backlog;
- `reason_counts`: deterministic count by reason code.

Reason derivation is deterministic and ticket-only:

- no normalized question text -> `missing_question`;
- fewer than two tickets -> `too_low_volume`;
- fewer than two supporting sources -> `insufficient_source_support`;
An unexpected low-confidence row that does not match those predicates raises
rather than silently emitting an unsupported reason code.

The section is paid-only (`web` + `export`) and intentionally absent from the
free Snapshot projection. No help-center corpus is consulted, so the slice does
not claim or implement `already_deflected`.

## Intentional

- No published-help-center dedupe in this slice. That would require a new
  help-center/documentation input and is deferred until the product needs live
  FAQ corpus comparison.
- No portfolio renderer in this PR. This slice makes ATLAS emit and document the
  contract first; portfolio can render the new section from the backend-owned
  model in the child follow-up.
- No result-page display cap for the suppressed review queue yet. Adding that
  cap would make the hosted QA harness require a renderer that this slice
  deliberately defers.
- No PDF surface claim yet. The queue is exposed for paid web/export consumers;
  PDF renderer wiring can follow once the product decides how prominent this
  audit queue should be in a shareable PDF.

## Deferred

- atlas-portfolio child slice: render `suppressed_repeat_review_queue` and add
  audience framing for content vs product/ops lanes.
- Future ATLAS slice, if sold/needed: ingest a published help-center corpus and
  add true `already_deflected` dedupe.
- Future PDF polish: decide whether the suppressed review queue belongs in the
  curated shareable PDF, then update the PDF renderer and QA scorecard.

Parked hardening: none.

## Verification

- `pytest` for the selected report-model behaviors in `tests/test_content_ops_deflection_report.py` - 6 passed, 158 deselected.
- `pytest` for the selected contract-doc behaviors in `tests/test_content_ops_faq_report_contract_docs.py` - 2 passed, 3 deselected.
- `python` `scripts/generate_deflection_snapshot_example.py` `--check` - current.
- `pytest` for `tests/test_content_ops_deflection_report.py` and `tests/test_content_ops_faq_report_contract_docs.py` - 169 passed.
- `python` `scripts/generate_deflection_frontend_contract_types.py` `--check` - current for all four frontend contract outputs.
- `pytest` for `tests/test_generate_deflection_frontend_contract_types.py`, `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py`, and `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` - 40 passed.
- `npm` `run` `test:deflection-full-report-qa-hosted-smoke` from `portfolio-ui` - passed.
- `bash` `scripts/validate_extracted_content_pipeline.sh` - passed.
- `python` `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` `extracted_content_pipeline` - clean.
- `python` `scripts/audit_extracted_standalone.py` `--fail-on-debt` - 0 findings.
- `bash` `scripts/check_ascii_python.sh` - passed.
- `bash` `extracted/_shared/scripts/sync_extracted.sh` `extracted_content_pipeline` - refreshed mapped files; no unrelated diff added.
- Pending before push: `bash scripts/push_pr.sh <pr-body-file> -u origin HEAD`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 23 |
| `docs/frontend/content_ops_faq_report_contract.md` | 7 |
| `extracted_content_pipeline/faq_deflection_report.py` | 120 |
| `plans/PR-Deflection-Suppressed-Repeat-Review-Queue.md` | 159 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 18 |
| `portfolio-ui/scripts/faq-deflection-full-report-qa-hosted-smoke.test.mjs` | 10 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 77 |
| `scripts/generate_deflection_frontend_contract_types.py` | 2 |
| `tests/test_content_ops_deflection_report.py` | 161 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 4 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 8 |
| `tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py` | 9 |
| `tests/test_smoke_content_ops_deflection_pdf_export_validators.py` | 9 |
| **Total** | **607** |
