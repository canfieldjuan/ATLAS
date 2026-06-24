# PR-Deflection-Report-Hosted-Shape-Metadata

## Why this slice exists

#1831 made the hosted paid report projection safe and added a fail-closed
parity test, but it intentionally left one maintenance root behind: field-shape
knowledge still lives in code/tests instead of the generated ATLAS-owned report
model contract.

Root cause: the backend report projection contract already owns which fields are
hosted-safe and which fields are nested objects, nested collections, records, or
scalars. The generated frontend artifacts only expose field-name constants, so
`report.js` and the parity fixture must infer shapes locally. #1831 made that
inference fail closed, but future hosted-safe shape additions still require
editing runtime and test classifiers by hand.

This change fixes the root by publishing generated hosted-safe field-shape
metadata from the ATLAS projection contract and making both the public hosted
projection and its parity fixture consume that one generated source.

Review follow-up: the shape contract also covers nullable object-shaped fields.
`source_date_window` is hosted-safe and required-nullable, so the public
projection must preserve `null` instead of treating it as an invalid object
value. This slice now locks that boundary in the proxy parity test.

This slice is expected to exceed the 400 LOC soft cap because the root fix has
one small source contract change plus regenerated JavaScript and TypeScript
artifacts. Splitting the generated outputs from the generator/runtime change
would leave the repo in a stale-artifact state and defeat the contract gate this
slice is adding.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Extend the report projection metadata so hosted-safe nested object,
   nested collection, and record shapes are represented in the backend
   contract.
2. Generate a `DEFLECTION_REPORT_HOSTED_FIELD_SHAPES` artifact into both
   frontend contract outputs.
3. Replace local runtime/test shape maps in `portfolio-ui` with the generated
   shape artifact.
4. Keep #1831's fail-closed behavior: unknown/unclassified shapes do not pass
   as scalars.
5. Preserve legitimate `null` values for hosted-safe nullable object fields.

### Review Contract

Acceptance criteria:

1. `source_date_window`, `term_mappings`, `status_mix`, `status_counts`, and
   `reason_counts` shapes come from generated metadata, not local maps.
2. `publicHostedReportModel` projects scalar, scalar-array, record, object, and
   object-array fields from `DEFLECTION_REPORT_HOSTED_FIELD_SHAPES`.
3. The parity fixture builds expected data from the same generated shape map.
4. Generator validation rejects hosted-safe non-scalar fields without shape
   metadata.
5. Generated TypeScript types still represent record and nested shapes
   correctly.
6. Nullable hosted-safe object fields survive as `null` instead of being
   dropped.

Affected surfaces:

- Backend report projection contract metadata.
- Frontend contract generator output.
- Public paid report JSON proxy projection.
- Public paid report JSON proxy parity test.

Risk areas:

- Generator drift between Python contract metadata and JavaScript/TypeScript
  artifacts.
- Accidentally broadening public projection to arbitrary objects instead of
  allowlist-shaped fields.
- Generated artifact churn in `portfolio-ui`.

- Reviewer rules triggered: R1, R2, R5, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Report-Hosted-Shape-Metadata.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/api/content-ops/deflection/report.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The backend projection metadata adds explicit nested field/collection entries
for hosted-safe non-scalar fields that previously had only local runtime
knowledge. The frontend contract generator walks each section and nested owner,
derives each hosted-safe field's shape (`scalar`, `scalar_array`, `record`,
`object`, or `object_array`), and emits a path-keyed
`DEFLECTION_REPORT_HOSTED_FIELD_SHAPES` map.

`report.js` then projects by owner path instead of by generated field-name
constant plus local shape sets. For each generated shape it clones only the
corresponding safe structure: scalars, scalar arrays, scalar-valued records,
allowlisted nested objects, and allowlisted object arrays. Object-shaped fields
also preserve `null` when the generated contract admits the field and the
producer emits a legitimate nullable value.

The parity fixture uses the same generated shape map to build test data and to
assert every hosted-safe path survives. That keeps runtime projection and test
coverage anchored to one generated contract artifact instead of parallel
hand-maintained classifiers.

## Intentional

- This slice keeps the generated shape artifact path-keyed instead of emitting
  a richer schema tree. The path map is enough to remove the dual-map drift and
  keeps the public projection small.
- Unknown shape values remain fail-closed in the fixture and runtime: they are
  not treated as scalars.
- Non-null non-object values for object-shaped fields still fail closed; only a
  legitimate nullable `null` value is preserved.
- This does not change the hosted public payload contract beyond preserving the
  fields already marked hosted-safe by the ATLAS report projection contract.

## Deferred

- Cross-repo `atlas-portfolio` consumption of the generated report-model
  artifact remains a separate consumer slice after the ATLAS artifact is stable.

Parked hardening: none.

## Verification

- Passed: generator write command, python scripts/generate_deflection_frontend_contract_types.py.
- Passed: generator check command, python scripts/generate_deflection_frontend_contract_types.py --check.
- Passed: Python compile check for scripts/generate_deflection_frontend_contract_types.py.
- Passed: generator pytest command for tests/test_generate_deflection_frontend_contract_types.py - 23 tests.
- Passed: deflection report pytest command for tests/test_content_ops_deflection_report.py - 167 tests.
- Passed: portfolio-ui deflection proxy test script - 29 tests.
- Passed: extracted_content_pipeline manifest validation script.
- Passed: extracted_content_pipeline Atlas reasoning import audit.
- Passed: extracted standalone audit with fail-on-debt.
- Passed: extracted_content_pipeline ASCII Python check.
- Pending before push: push wrapper local review.
  (runs the local PR review bundle through the managed push path).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 45 |
| `plans/PR-Deflection-Report-Hosted-Shape-Metadata.md` | 161 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 241 |
| `portfolio-ui/api/content-ops/deflection/report.js` | 89 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 274 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 260 |
| `scripts/generate_deflection_frontend_contract_types.py` | 204 |
| `tests/test_content_ops_deflection_report.py` | 72 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 96 |
| **Total** | **1442** |
