# PR-Deflection-Report-Runtime-Contract

## Why this slice exists

#1805's paid report-model contract arc now has an ATLAS-owned
`report_projection` artifact, but #1815 intentionally stopped before proving
that contract against the runtime producer. The #1815 reviewer called this out
as the hard gate before report-model codegen: `projected_fields` and the new
section `presence` metadata are still hand-written claims until
the build_deflection_report_artifact output is reconciled against them.

Root cause: the report projection contract and the runtime
`DeflectionStructuredReport` producer are currently parallel shapes. The
contract can drift by adding/removing a field, mislabeling a conditional section
as required, or modeling nested rows incorrectly without any test comparing the
contract to a real emitted artifact. This slice fixes that root at the ATLAS
producer/contract boundary by adding runtime parity tests over real report
artifacts. It does not generate TypeScript yet; codegen is deferred until this
runtime proof is load-bearing.

This slice is over the usual 400 LOC soft budget because the parity checker has
to cover top-level fields, conditional presence, collections, nested objects,
nested collections, and negative drift probes in one coherent test harness. If
those proofs are split, slice 3 could still generate from a partially verified
contract.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Add a runtime parity test that builds a real paid deflection report artifact
   and asserts each emitted section's top-level `data` keys match that
   section's `report_projection.projected_fields`, allowing only fields marked
   `optional_projected_fields` to be absent.
2. Assert collection item contracts from real emitted rows/items, including
   action-section `items`, `csat_signal` nested objects, `top_evidence` nested
   collections, ranked/question/detail rows, outcome diagnostic rows, and string
   phrase collections.
3. Add omit-case coverage for conditional section presence: no `source_label`
   means `source_file` is absent, no rendered outcome diagnostics means
   `outcome_diagnostics` is absent, and a fixture with both signals proves they
   are present with the declared fields.
4. Add negative drift probes that mutate a real report model to prove missing
   required fields, unexpected extra fields, and wrong conditional presence are
   detected by the parity checker.

### Review Contract
- Acceptance criteria:
  - [ ] The test suite compares `report_projection` to
   build_deflection_report_artifact output, not to another hand-written
        mirror.
  - [ ] Required section presence and conditional section absence/presence are
        both verified with real artifacts.
  - [ ] Top-level section data, collection row/item fields, nested object
        fields, and nested collection item fields are all covered.
  - [ ] At least one negative fixture proves each checker branch fails closed
        for missing fields, extra fields, and presence drift.
  - [ ] No frontend codegen or portfolio consumer changes are included.
- Affected surfaces: paid deflection report model contract tests; extracted
  content pipeline contract metadata if a real mismatch is found.
- Risk areas: schema/contract drift, future frontend codegen correctness,
  report payload privacy boundary.
- Reviewer rules triggered: R1, R2, R10, R14; boundary-probe required because
  this adds a contract checker.

### Files touched

- `plans/PR-Deflection-Report-Runtime-Contract.md`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

Add small test-local helpers in `tests/test_content_ops_deflection_report.py`
that:

1. read `deflection_report_model_contract_shape()["report_projection"]`;
2. build real build_deflection_report_artifact payloads;
3. walk emitted sections by `id`;
4. compare actual `data` key sets to declared projected fields, with optional
   fields allowed to be absent;
5. recursively compare declared collection item fields, nested object fields,
   nested collection item fields, and record-field mappings; and
6. return human-readable drift messages so negative tests can assert the
   failure mode.

The positive test uses a rich fixture with `source_label` and outcome
diagnostics so every section can be checked. A second sparse fixture omits
`source_label` and outcome diagnostics to prove the two conditional sections
are legitimately absent while required sections remain present.

## Intentional

- This is test-enforced runtime parity, not a production runtime validator. The
  goal is to make CI fail before a drifted contract can feed report-model
  codegen; adding request-time validation would add cost to report generation
  without changing the emitted payload.
- The checker lives beside the existing report tests rather than in a new
  script because the codegen step does not exist yet. A standalone
  `scripts/check_*` gate can consume this same contract after slice 3 publishes
  generated artifacts.

## Deferred

- Slice 3: generate the report-model frontend artifact from the ATLAS-owned
  report projection after this runtime enforcement lands.
- Cross-repo portfolio/proxy consumption remains deferred until ATLAS publishes
  that generated artifact.

Parked hardening: none.

## Verification

- python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_projection_fields_match_runtime_output tests/test_content_ops_deflection_report.py::test_deflection_report_projection_conditional_sections_match_runtime_output tests/test_content_ops_deflection_report.py::test_deflection_report_projection_checker_fails_on_field_drift tests/test_content_ops_deflection_report.py::test_deflection_report_projection_checker_fails_on_nested_drift tests/test_content_ops_deflection_report.py::test_deflection_report_projection_checker_fails_on_presence_drift -q
  - 5 passed.
- python -m pytest tests/test_content_ops_deflection_report.py -q
  - 160 passed.
- python -m py_compile extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py
  - Passed.
- git diff --check
  - Passed.
- bash scripts/validate_extracted_content_pipeline.sh
  - Passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt
  - Passed.
- bash scripts/check_ascii_python.sh
  - Passed.
- Pending before push: bash scripts/push_pr.sh /tmp/atlas-pr-body-deflection-report-runtime-contract.md

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Report-Runtime-Contract.md` | 138 |
| `tests/test_content_ops_deflection_report.py` | 361 |
| **Total** | **499** |
