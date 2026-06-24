# PR-Deflection-Report-Frontend-Artifact

## Why this slice exists

#1805 is the remaining report-contract work after the snapshot contract arc and
the paid report contract enforcement landed. #1815 published the ATLAS-owned
paid `report_projection` contract, and #1817 proved it against real
`build_deflection_report_artifact()` output. The next missing piece is the
ATLAS-owned frontend artifact that future in-repo and cross-repo consumers can
generate from instead of hand-authoring paid report-model types.

Root cause: the paid `DeflectionStructuredReport` frontend shape is still
consumer-inferred. Snapshot types already come from
`deflection_report_model_contract_shape()["snapshot_projection"]`, but the paid
report model does not have an equivalent generated TypeScript artifact. This
slice fixes the root for the ATLAS artifact layer by extending the existing
frontend contract generator to emit paid report-model TypeScript from the now
runtime-enforced `report_projection`.

This slice does not wire portfolio consumers to the generated types yet. It
publishes the source artifact and CI drift gate that consumer slices need.

This slice is expected to exceed the 400 LOC soft cap because the generated
artifact intentionally enumerates every paid report section and nested field.
Splitting the generator from the committed artifact would leave `--check`
unable to prove the source-of-truth output in the same PR.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Extend `scripts/generate_deflection_frontend_contract_types.py` so it emits
   a committed paid report-model TypeScript artifact from `report_projection`.
2. Add the generated `portfolio-ui/src/types/deflectionReportModel.ts` file with
   section IDs, model/section envelope fields, per-section data field tuples,
   section data types, collection item types, nested object/collection item
   types, and the exported `DeflectionStructuredReport` shape.
3. Extend the generator's `--check` mode and CI path enrollment so stale
   report-model output fails the existing deflection report workflow.
4. Add generator tests proving report fields, conditional sections, nested
   collections, record fields, and fail-closed unknown report fields.

### Review Contract
- Acceptance criteria:
  - [ ] The paid report-model TS artifact is generated from
        `report_projection`, not hand-written.
  - [ ] `--check` fails when the committed report-model artifact is stale.
  - [ ] The generated artifact represents section IDs, report/section envelope
        fields, required-vs-optional data fields, conditional section presence,
        collection item fields, nested object fields, nested collection item
        fields, and record fields.
  - [ ] Unknown report-projection fields fail closed rather than becoming
        `any`.
  - [ ] No portfolio page/parser or atlas-portfolio cross-repo consumption is
        included.
- Affected surfaces: generator script, generated in-repo portfolio-ui TS
  artifact, deflection report CI path enrollment, generator tests.
- Risk areas: schema/contract drift, future frontend codegen correctness,
  generated artifact staleness.
- Reviewer rules triggered: R1, R2, R10, R12, R14; boundary-probe required
  because this extends a contract generator/checker.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_report_checks.yml`
- `plans/PR-Deflection-Report-Frontend-Artifact.md`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

Reuse the existing frontend contract generator rather than adding a second
script:

1. keep the current snapshot output unchanged;
2. read `report_projection.sections`;
3. validate each projected/optional/nested/record field against explicit
   TypeScript type mappings;
4. render stable const tuples for section IDs and per-section projected fields;
5. render data and collection item types from the contract metadata; and
6. include the new output in `main(... --check)` so CI compares the committed
   artifact to freshly generated output.

## Intentional

- The generated artifact covers the full paid report model, including paid-only
  fields. Hosted-safe subset construction remains a later consumer slice; this
  PR publishes the source contract types without changing runtime payloads.
- The artifact is added to in-repo `portfolio-ui` only. `atlas-portfolio` is a
  separate repo and should consume this after ATLAS owns and checks the
  generated artifact.

## Deferred

- Slice 4: consume the generated report-model artifact in in-repo
  `portfolio-ui` and/or publish the cross-repo artifact for `atlas-portfolio`.
- Hosted-safe payload construction from `hosted_consumer_safe_fields` remains a
  consumer-boundary slice, not this artifact-publication slice.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_generate_deflection_frontend_contract_types.py -q`
  - 11 passed.
- `python scripts/generate_deflection_frontend_contract_types.py --check`
  - all three generated artifacts current.
- Python py_compile for `scripts/generate_deflection_frontend_contract_types.py`
  and `tests/test_generate_deflection_frontend_contract_types.py`
  - passed.
- `git diff --check`
  - passed.
- `scripts/validate_extracted_content_pipeline.sh`
  - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - passed.
- `scripts/check_ascii_python.sh`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_report_checks.yml` | 2 |
| `plans/PR-Deflection-Report-Frontend-Artifact.md` | 133 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 621 |
| `scripts/generate_deflection_frontend_contract_types.py` | 486 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 127 |
| **Total** | **1369** |
