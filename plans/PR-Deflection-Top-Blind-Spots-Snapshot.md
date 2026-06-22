# PR-Deflection-Top-Blind-Spots-Snapshot

## Why this slice exists

Issue #1799 found a demo-only deliverable gap: the portfolio landing/results
surfaces render `top_blind_spots`, but the backend free deflection snapshot
never emits that key for real reports. Real snapshots therefore silently omit a
section the product advertises.

Root cause: the backend snapshot contract has no allowlisted projection for
blind spots even though the paid report already produces the underlying
unresolved repeat rows. This PR fixes the root for the current backend shape by
projecting `top_blind_spots` through the existing `snapshot_safe_fields`
allowlist spine. The broader codegen/derivation hardening noted in #1799 stays
deferred.

## Scope (this PR)

Ownership lane: issue-1799/deflection-top-blind-spots-snapshot
Slice phase: Vertical slice

1. Register a snapshot-safe projection on `top_unresolved_repeats.items` for
   `rank`, `question`, and `ticket_count`.
2. Emit `top_blind_spots` from real backend snapshots as an array of
   `{rank, question, ticket_count}` rows. Empty results emit `[]`.
3. Treat the product predicate as the paid report's existing
   `top_unresolved_repeats`: high-volume unresolved repeated questions
   (`Needs answer` / `Needs review`, ticket_count >= 2), preserving the report's
   current ordering/limits.
4. Update the snapshot drift guard, exact `snapshot_safe_fields` assertions, and
   frontend contract doc in the same slice.

### Review Contract

Acceptance criteria:
- `DeflectionSnapshot.as_dict()` always includes `top_blind_spots`.
- `top_blind_spots` rows are allowlist-constructed from
  `top_unresolved_repeats.items` and expose only `rank`, `question`, and
  `ticket_count`.
- Paid-only action row fields such as status, recommended action, source IDs,
  evidence, representative phrasing, support cost, and identity fields never
  reach the free snapshot.
- Existing top question, locked question, and teaser behavior remains unchanged.
- The contract doc names `top_blind_spots` and the unpaid rendering guidance
  includes it.

Affected surfaces:
- `extracted_content_pipeline.faq_deflection_report`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_deflection_snapshot_report_drift.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `docs/frontend/content_ops_faq_report_contract.md`

Risk areas:
- Privacy/paywall boundary: this is a new free snapshot field.
- Drift between the paid unresolved repeat section and the free snapshot.
- Portfolio consumes the field without a frontend change, so backend shape must
  exactly match the existing parser.

Reviewer rules triggered: R1, R2, R3, R8, R10, R13, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Top-Blind-Spots-Snapshot.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`
- `tests/test_deflection_snapshot_report_drift.py`

## Mechanism

- Add `top_blind_spots` to `DeflectionSnapshot`.
- Add snapshot-safe fields to the `top_unresolved_repeats` section definition:
  `items.rank`, `items.question`, and `items.ticket_count`.
- In report-model snapshots, read the allowlisted projected
  `top_unresolved_repeats.items` rows and construct the exact frontend-ready row
  shape. Legacy non-`report_model` snapshots emit `[]` because they do not have
  the paid report section registry available.
- Extend the drift guard to compare snapshot blind spots with the paid
  `top_unresolved_repeats` section and to prove injected paid fields are caught
  by the shared forbidden-field detector.

## Intentional

- No atlas-portfolio change in this slice. The issue verified the frontend
  parser already expects `{rank, question, ticket_count}` from
  `o.top_blind_spots`.
- No new blind-spot classifier. The product definition reuses the existing
  `top_unresolved_repeats` paid section so report, PDF, email, and snapshot
  language stay aligned.
- Legacy artifacts without a supported `report_model` emit an empty
  `top_blind_spots` list rather than guessing from old FAQ item fields.

## Deferred

- Structural hardening from #1799: derive/codegen the snapshot/report/frontend
  shape instead of maintaining parallel hand-synced contracts.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_deflection_snapshot_report_drift.py -q -- 160 passed.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_snapshot_example_matches_producer_shape -q -- 1 passed.
- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q -- 5 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py tests/test_deflection_snapshot_report_drift.py tests/test_content_ops_faq_report_contract_docs.py -q -- 165 passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline -- passed.
- Command: python -m json.tool docs/frontend/content_ops_faq_deflection_snapshot_example.json >/dev/null && python -m py_compile extracted_content_pipeline/faq_deflection_report.py -- passed.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Top-Blind-Spots-Snapshot.md --check -- passed.
- Pending before push: Atlas push wrapper local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 6 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 7 |
| `docs/frontend/content_ops_faq_report_contract.md` | 13 |
| `extracted_content_pipeline/faq_deflection_report.py` | 50 |
| `plans/PR-Deflection-Top-Blind-Spots-Snapshot.md` | 131 |
| `tests/test_content_ops_deflection_report.py` | 19 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 5 |
| `tests/test_deflection_snapshot_report_drift.py` | 38 |
| **Total** | **269** |
