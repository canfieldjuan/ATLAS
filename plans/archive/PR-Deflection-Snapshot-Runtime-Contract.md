# PR-Deflection-Snapshot-Runtime-Contract

## Why this slice exists

The #1804 sequencing review called out the remaining high-value gap in the
snapshot chain: the contract now documents `projected_fields`, but the test
still checks hardcoded expected lists instead of the real runtime output from
`build_deflection_snapshot`. A projection implementation change can drift away
from the contract while the contract test stays green.

Root cause: the contract assertion is anchored to duplicated literals rather
than to the runtime projection boundary. This PR fixes the root for
`projected_fields` by building a representative snapshot and comparing the
actual emitted field sets to the contract.

## Scope (this PR)

Ownership lane: deflection/full-report-actionability
Slice phase: Production hardening

1. Add a runtime-backed contract test for snapshot projection fields.
2. Keep the existing registry-derived assertions for source section metadata and
   snapshot-safe allowlists.
3. Prove summary, top questions, locked questions, top blind spots, teaser full
   answer, and teaser previews match the contract's projected field lists.

### Review Contract

- Acceptance criteria:
  - [ ] The representative runtime snapshot contains rows for every contract
        field that declares `projected_fields`, including locked questions and
        teaser preview rows.
  - [ ] The test compares actual snapshot key sets to
        `deflection_report_model_contract_shape()["snapshot_projection"]`.
  - [ ] The old hardcoded expected lists are not the only proof for runtime
        projected field shape.
- Affected surfaces: deflection report contract tests.
- Risk areas: false-green snapshot contract, frontend type generation
  preconditions, free snapshot privacy boundary.
- Reviewer rules triggered: R1, R2, R10, R14. Boundary-probe required because
  this is a contract checker.

### Files touched

- `plans/PR-Deflection-Snapshot-Runtime-Contract.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

The test uses the existing structured deflection report fixture, adds one extra
scoped resolved repeat with complete source-date metadata, and calls
`build_deflection_snapshot` with `top_n=1`. That forces the runtime path to emit
a top question, locked question, top blind spot, teaser full answer, and teaser
preview. The assertion indexes the contract by field name and compares actual
snapshot key sets to the contract's projected field lists.

## Intentional

- This slice changes test enforcement only. The runtime projection already emits
  the intended shape; the defect is the contract test's proof source.
- It reuses the deterministic structured report fixture instead of creating a
  new fixture corpus.

## Deferred

- Generate frontend/portfolio types from the enforced contract in the next
  slice.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projected_fields_match_runtime_output -q -- 2 passed.
- python -m py_compile tests/test_content_ops_deflection_report.py -- passed.
- python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_model_contract_shape_requires_version_bump tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projection_contract_is_registry_derived tests/test_content_ops_deflection_report.py::test_deflection_snapshot_projected_fields_match_runtime_output tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example -q -- 4 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- passed.
- bash scripts/check_ascii_python.sh -- passed.
- git diff --check -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Snapshot-Runtime-Contract.md` | 86 |
| `tests/test_content_ops_deflection_report.py` | 59 |
| **Total** | **145** |
