# PR-FAQ-Deflection-Snapshot-Contract

## Why this slice exists

The paid-gated deflection report now returns a free
`DeflectionSnapshot` before payment, but `docs/frontend/` only documents the
full paid report artifact. The results-page builder needs the same
byte-faithful handoff for the free snapshot so it can render real top questions
without accidentally typing against the paid `faq_result` shape.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating

Slice phase: Product polish

1. Document the `DeflectionSnapshot` TypeScript shape in the existing FAQ
   frontend contract.
2. Add a compact JSON example generated from the same producer fixture as the
   full deflection report example.
3. Extend the contract test so the example matches the producer shape and does
   not expose paid answer/evidence fields.

### Files touched

| File | Purpose |
|---|---|
| `docs/frontend/content_ops_faq_report_contract.md` | Adds the free snapshot contract and rendering guidance. |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | Adds a byte-faithful free snapshot example. |
| `tests/test_content_ops_faq_report_contract_docs.py` | Locks the snapshot example against the producer helper. |
| `plans/PR-FAQ-Deflection-Snapshot-Contract.md` | Documents the slice contract and verification. |

## Mechanism

The snapshot example is produced by:

```python
build_deflection_snapshot(_producer_deflection_report_payload(), top_n=2).as_dict()
```

That is the same projection used by the hosted free results route. The contract
test compares keys against the producer payload and asserts that the JSON does
not contain paid fields such as `markdown`, `faq_result`, `steps`,
`evidence_quotes`, or `source_ids`.

## Intentional

- This slice only documents the free snapshot contract. It does not change the
  hosted paid gate or frontend rendering code.
- The example uses top 2 for compact docs, while the hosted default remains top
  5 through `DEFAULT_DEFLECTION_SNAPSHOT_TOP_N`.

## Deferred

- Future slice: portfolio/Intel UI can type the free results page against this
  snapshot example.
- Parked hardening considered: none.

## Verification

- Command: pytest tests/test_content_ops_faq_report_contract_docs.py tests/test_content_ops_deflection_report.py::test_deflection_snapshot_strips_answers_evidence_and_sources -q
  - Result: 5 passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Snapshot-Contract.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Snapshot-Contract.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Contract doc | 47 |
| Snapshot example | 21 |
| Contract tests | 36 |
| Plan doc | 78 |
| **Total** | **182** |
