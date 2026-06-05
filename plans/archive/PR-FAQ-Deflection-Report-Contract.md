# PR-FAQ-Deflection-Report-Contract

## Why this slice exists

PR-FAQ-Deflection-Report-Output made `faq_deflection_report` a real Content Ops
execute output, but the frontend/demo handoff doc still only describes
`faq_markdown`. The next consumer needs a stable contract for the customer-facing
deflection report artifact shape without reverse-engineering the Python result.

This slice documents the new output shape and adds a checked example so the docs
stay aligned with the producer.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Product polish

1. Extend the FAQ report frontend contract with the `faq_deflection_report`
   execute result shape.
2. Add a compact deflection report example JSON generated from the same support
   ticket style as the existing FAQ example.
3. Extend the contract-doc test to verify the example keys match the real
   deflection report artifact dictionary shape.

### Files touched

| File | Purpose |
|---|---|
| `docs/frontend/content_ops_faq_report_contract.md` | Documents the deflection report execute result and rendering guidance. |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | Adds a compact frontend/demo example artifact. |
| `tests/test_content_ops_faq_report_contract_docs.py` | Verifies the new example matches producer shape and the doc links it. |
| `plans/PR-FAQ-Deflection-Report-Contract.md` | Documents this slice contract. |

## Mechanism

The docs name the deflection report artifact dictionary output as the canonical
producer contract for `faq_deflection_report`. The example carries the
top-level `markdown`, `summary`, and `faq_result` keys that the execute route
returns inside `steps[0].result`.

The test builds a real FAQ result, wraps it with
`build_deflection_report_artifact(...)`, and compares the checked-in example's
top-level keys, summary keys, and nested FAQ result keys against the producer.

## Intentional

- This is docs/contract only; no runtime behavior changes.
- The example is compact and synthetic, but labeled as an example artifact, not
  customer proof.
- Keep the existing `content_ops_faq_report_example.json` unchanged so current
  `faq_markdown` consumers are not disrupted.

## Deferred

- Future PR: update Intel UI controls after the backend and handoff contract are
  both landed.
- Future PR: add a real anonymized customer example when a design-partner export
  exists.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q
  - Result: 3 passed.
- Command: python -m py_compile tests/test_content_ops_faq_report_contract_docs.py
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Contract.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Contract.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-contract.md
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Contract doc | 55 |
| Example JSON | 157 |
| Tests | 77 |
| Plan doc | 85 |
| **Total** | **371** |
