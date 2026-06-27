# PR-Deflection-Owner-Category-Contract

## Why this slice exists

atlas-portfolio#384 asks for owner-cost accountability without changing the
meaning of the existing `owner_lane` field. The portfolio page can now roll up
cost by the lane it receives, but ATLAS still only emits `owner_lane` as the
routeable topic/area (`Auth / Product UX`, `Billing`, `Reporting`, etc.). That
is useful, but it is not the higher-level category the buyer needs for "content
vs product/support" accountability.

Root cause: action rows have a deterministic `status`/`fix_type` split, but the
report contract does not expose the category that follows from that split. If
portfolio repurposes `owner_lane`, the contract changes semantics with no shape
change; if portfolio invents categories locally, it fakes product truth.

This slice fixes the root for the first category layer by adding an optional,
hosted-safe `owner_category` field derived from the existing action status. It
does not claim policy detection yet.

## Scope (this PR)

Ownership lane: deflection/report-model
Slice phase: Vertical slice

1. Add `owner_category` to deflection action rows and Jira handoff data without
   removing or changing `owner_lane`.
2. Derive `owner_category` deterministically from existing action status:
   publishable/missing/review/low-confidence rows route to content/support
   enablement; recurring-after-answer rows route to product/support experience.
3. Keep the new field optional for legacy stored `deflection.v1` rows and
   hosted-safe for paid consumers.
4. Regenerate frontend report-model contracts from the backend contract.
5. Add tests for category derivation, contract projection, generated artifacts,
   and legacy backfill.

### Review Contract

- Acceptance criteria:
  - `owner_lane` values and semantics remain unchanged.
  - `owner_category` is present on newly generated action rows, included in
    `jira_template`, and admitted to hosted paid consumers.
  - Stored legacy reports that lack `owner_category` are normalized with a
    deterministic fallback instead of surfacing `undefined`.
  - No policy category is emitted unless a deterministic policy rule exists in
    this PR.
- Affected surfaces:
  - deflection report producer and generated report-model contracts.
  - ATLAS `portfolio-ui` proxy contract artifacts.
- Risk areas:
  - contract back-compat for persisted reports;
  - generated artifact drift;
  - accidental `owner_lane` semantic change.
- Reviewer rules triggered: R1, R8, R10, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Owner-Category-Contract.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The producer computes `status = _action_status(item)` before building each
action row. A new helper maps that status to `owner_category`:

- `Already covered but still recurring` -> `Product / Support Experience`
- `Draft ready`, `Needs answer`, `Needs review`, and `Low confidence` ->
  `Content / Support Enablement`
- unknown statuses -> `Review`

That category is emitted beside `owner_lane` and copied into
`jira_template`. The report-model collection metadata adds the field to
projected fields, hosted-safe fields, and optional context fields so older
stored rows remain compatible. The stored-report normalization/backfill path
gets the same deterministic fallback used by the producer.

Generated contract artifacts are refreshed with
`scripts/generate_deflection_frontend_contract_types.py`, so
`portfolio-ui/src/types/deflectionReportModel.ts` and
`portfolio-ui/api/content-ops/deflection/report-model-contract.js` match the
producer contract.

## Intentional

- No `owner_lane` repurpose. It remains the routeable topic/area.
- No new `cost_by_owner` section in ATLAS. Portfolio already computes the
  rollup as a view from rows, which keeps this additive.
- No policy category in this slice. The current data has no deterministic rule
  strong enough to separate policy failures from missing-content or
  recurring-after-answer rows.
- The category labels are broad and buyer-facing; person/team routing remains
  out of scope.

## Deferred

- Policy as a distinct category remains deferred until a deterministic policy
  signal or operator taxonomy exists.
- Person/team routing remains deferred until an assignee/group export field or
  operator owner map is accepted.
- Portfolio consumption of `owner_category` is a follow-up after this backend
  contract lands and the portfolio contract is regenerated from merged ATLAS
  main.

Parked hardening: none.

## Verification

- `scripts/generate_deflection_frontend_contract_types.py` via Python - passed,
  regenerated the report-model TS/API contract artifacts.
- `scripts/generate_deflection_snapshot_example.py` via Python - passed, refreshed
  the canonical frontend report example with `owner_category`.
- `python -m pytest tests/test_content_ops_deflection_report.py -q` - passed,
  173 tests.
- `python -m pytest tests/test_generate_deflection_frontend_contract_types.py -q`
  - passed, 23 tests.
- `python -m pytest tests/test_smoke_content_ops_deflection_hosted_qa_scorecard.py -q`
  - passed, 3 tests.
- `python -m pytest tests/test_smoke_content_ops_deflection_pdf_export_validators.py -q`
  - passed, 18 tests.
- `scripts/validate_extracted_content_pipeline.sh` via Bash - passed.
- `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` via Python
  - passed.
- `scripts/audit_extracted_standalone.py` via Python - passed.
- `scripts/check_ascii_python.sh` via Bash - passed.
- `extracted/_shared/scripts/sync_extracted.sh` via Bash
  - passed; diff unchanged after sync.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 42 |
| `extracted_content_pipeline/deflection_report_access.py` | 22 |
| `extracted_content_pipeline/faq_deflection_report.py` | 17 |
| `plans/PR-Deflection-Owner-Category-Contract.md` | 148 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 48 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 60 |
| `scripts/generate_deflection_frontend_contract_types.py` | 1 |
| `tests/test_content_ops_deflection_report.py` | 18 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 12 |
| **Total** | **368** |
