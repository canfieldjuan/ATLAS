# PR-Deflection-Suppressed-Review-Key

## Why this slice exists

The next #324 follow-up is a real reviewer override workflow for suppressed repeat rows. Portfolio can display suppressed rows after atlas-portfolio#378, but it cannot persist reviewer decisions safely yet because the hosted-safe payload strips `repeat_key` and `cluster_id`, and keying overrides by rank or question text would be brittle and would store customer wording unnecessarily.

Root cause: the paid report model has no hosted-safe stable identifier for suppressed review rows. This change fixes the root prerequisite by adding a deterministic `review_key` to suppressed repeat review items and publishing that field in the hosted-consumer-safe allowlist.

## Scope (this PR)

Ownership lane: deflection/report-review-overrides
Slice phase: Vertical slice

1. Add a stable, non-raw `review_key` to each `suppressed_repeat_review_queue.items[]` row.
2. Keep `repeat_key`, `cluster_id`, source IDs, and evidence quotes paid/export-only; hosted consumers receive only `review_key` plus the existing safe row fields and suppression reason labels.
3. Regenerate the Portfolio contract artifacts so atlas-portfolio can consume the key in the next UI/API slice.
4. Add focused tests proving `review_key` is present, deterministic, hosted-safe, and does not expose raw repeat identity fields.

### Review Contract

- Acceptance criteria:
  - [ ] `suppressed_repeat_review_queue.items[]` includes `review_key` for every emitted row.
  - [ ] `review_key` is stable across repeated builds of the same input and does not change when row rank/order changes.
  - [ ] Hosted-safe projection includes `review_key` but still excludes `repeat_key`, `cluster_id`, `top_evidence`, source IDs, and evidence quotes.
  - [ ] Generated `portfolio-ui` TypeScript and API contract artifacts expose the new field.
- Affected surfaces: report model contract / hosted projection / generated frontend artifacts / tests.
- Risk areas: backcompat, privacy, contract drift.
- Reviewer rules triggered: R1, R2, R5, R10, R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Suppressed-Review-Key.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The suppressed review queue builds from action items that already carry deterministic internal identity (`repeat_key` / `cluster_id`). This PR derives `review_key` from the section name, raw repeat identity, and suppression reason using SHA-256, then exposes a prefixed short digest such as `review_abcd...`. The key is deterministic for the same suppressed finding but does not reveal the raw repeat identity or question text.

Only the suppressed review item schema gets the key. The hosted-safe allowlist for that collection includes `review_key`; the shared action-item hosted allowlist remains unchanged so other report sections do not accidentally expose raw or new identifiers.

## Intentional

- `repeat_key` and `cluster_id` remain paid/export-only. They are already hashes, but exposing a new purpose-specific review key keeps the browser contract narrow and prevents Portfolio from depending on internal report identity fields.
- This does not add an override endpoint or persisted decision store. It only creates the safe row key needed for that next Portfolio slice.

## Deferred

- Portfolio follow-up: persist reviewer decisions keyed by `(request_id, review_key)` and render reviewed/promoted/suppressed states.
- Future ATLAS follow-up: accept reviewer decisions back into the report generation flow if we want overrides to regenerate exports, not just annotate the hosted report.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py -k "suppressed_repeat_review_queue or suppressed_review_key or projection_separates_paid_and_hosted_action_fields"` - passed, 3 selected.
- `python -m pytest tests/test_generate_deflection_frontend_contract_types.py -k "report_model"` - passed, 11 selected.
- `python scripts/generate_deflection_frontend_contract_types.py --check` - passed.
- `ATLAS_CURRENT_PR_BODY_FILE=/tmp/deflection-suppressed-review-key-pr-body.md bash scripts/local_pr_review.sh` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 20 |
| `plans/PR-Deflection-Suppressed-Review-Key.md` | 76 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 4 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 5 |
| `scripts/generate_deflection_frontend_contract_types.py` | 1 |
| `tests/test_content_ops_deflection_report.py` | 58 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 10 |
| **Total** | **174** |
