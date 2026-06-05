# PR-FAQ-Report-Contract-Producer-Fidelity

## Why this slice exists

PR #988 documented the frontend-facing generated FAQ report shape and shipped a
compact JSON example for landing-page/demo work. Review accepted the contract
as accurate, but flagged that the doc test only proved the example matched a
hardcoded test mirror, not the actual FAQ producer.

This slice closes that drift gap while the contract is new. If
`build_ticket_faq_markdown` adds, removes, or renames an item field, the
frontend contract example should fail until the doc/example updates in the same
change.

## Scope (this PR)

Ownership lane: content-ops/faq-generator
Slice phase: Functional validation

1. Derive the expected generated FAQ item key set from
   `build_ticket_faq_markdown`.
2. Assert the checked-in frontend JSON example uses the same item field surface
   as the producer.
3. Keep the existing semantic checks for output checks, source IDs, question
   source, and answer evidence status.

### Files touched

| File | Change |
|---|---|
| `plans/PR-FAQ-Report-Contract-Producer-Fidelity.md` | Plan contract for this validation slice. |
| `tests/test_content_ops_faq_report_contract_docs.py` | Tie the example item field surface to the real FAQ producer. |

## Mechanism

The contract-doc test builds a small deterministic FAQ result with
`build_ticket_faq_markdown`, reads the first generated item, and compares its
keys to every item in `docs/frontend/content_ops_faq_report_example.json`.

The synthetic example values remain checked in for frontend use; only the field
surface is producer-derived so future producer drift fails closed.

## Intentional

- No generated FAQ runtime behavior changes.
- No documentation text or JSON example value changes unless the producer field
  surface is already different.
- This does not generate the frontend example dynamically; the static artifact
  remains easy for other sessions to consume.

## Deferred

- Formal OpenAPI/schema export remains deferred until the hosted API schema is
  generated from FastAPI.
- Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q - 2 passed.
- python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Report-Contract-Producer-Fidelity.md - passed.
- git diff --check - passed.
- Local PR review bundle - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | 70 |
| Test update | 38 |
| **Total** | **~108** |
