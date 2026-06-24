# PR-Deflection-Report-Hosted-Safe-Artifact

## Why this slice exists

#1805's report-contract arc has now landed the paid report contract, runtime
enforcement, generated paid frontend artifact, and in-repo paid consumer. The
remaining boundary gap named by the prior slices is hosted-safe construction:
the backend `report_projection` contract already carries
`hosted_consumer_safe_fields`, but the generated frontend artifacts currently
drop that metadata. A consumer cannot construct hosted/token page payloads from
the contract until ATLAS publishes those allowlists.

Root cause: the generator validated hosted-safe metadata inside
`report_projection` but only emitted paid runtime fields and
`snapshot_safe_fields`. This fixes the root for the ATLAS-owned artifact by
publishing hosted-safe allowlist constants from the backend contract; it does
not yet consume those constants in a hosted result page.

Diff-size note: this lands slightly above the 400 LOC soft cap because half of
the diff is generated constants plus the negative fixtures required for the
privacy-boundary checker. The hand-written implementation remains narrow.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Emit generated `*_HOSTED_CONSUMER_SAFE_FIELDS` constants for report sections,
   object collections, nested objects, and nested collections in both the
   TypeScript and JS report-model artifacts.
2. Keep runtime paid model types unchanged: `hosted_consumer_safe_fields` is
   contract metadata, not a field on `DeflectionStructuredReport` sections.
3. Add generator tests proving hosted-safe metadata is emitted, and fails closed
   when hosted-safe metadata references a field outside the paid projection.

### Review Contract

- Acceptance criteria: generated artifacts publish hosted-safe allowlists for
  top-level sections, action `items`, nested `csat_signal`, and raw
  `top_evidence` as an empty hosted-safe allowlist.
- Affected surfaces: `scripts/generate_deflection_frontend_contract_types.py`,
  `portfolio-ui/src/types/deflectionReportModel.ts`,
  `portfolio-ui/api/content-ops/deflection/report-model-contract.js`, and the
  generator tests.
- Risk areas: do not widen the runtime paid report model shape; do not add
  hosted-safe metadata to free snapshot output; fail closed on invalid
  hosted-safe fields.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  security/privacy boundary, R6 generated artifact drift, R9 checker failure
  branches, R13 class fix, R14 codebase verification.

### Files touched

- `plans/PR-Deflection-Report-Hosted-Safe-Artifact.md`
- `portfolio-ui/api/content-ops/deflection/report-model-contract.js`
- `portfolio-ui/src/types/deflectionReportModel.ts`
- `scripts/generate_deflection_frontend_contract_types.py`
- `tests/test_generate_deflection_frontend_contract_types.py`

## Mechanism

The generator already reads `report_projection` and validates the section,
collection, nested-object, and nested-collection metadata. This slice teaches
that metadata pass to validate `hosted_consumer_safe_fields` as subsets of the
same object's projected fields, then renders stable constants in the generated
TS/JS artifacts.

The runtime `DeflectionStructuredReport` types remain paid-shape types. Hosted
allowlists are emitted only as sidecar constants for future consumers to use
when allowlist-constructing hosted result-page payloads.

## Intentional

- No hosted result-page consumer change in this PR. Publishing the ATLAS-owned
  artifact first keeps the next consumer slice small and prevents repeating the
  atlas-portfolio "waiting on ATLAS artifact" problem.
- No `hosted_consumer_safe_fields` property is added to
  `DeflectionReportSection`. The runtime report model does not carry that
  metadata; adding it to the type would make valid paid report output look
  invalid.

## Deferred

- Hosted result-page construction must consume these generated allowlists in a
  follow-up slice before any hosted/free paid report payload uses the paid
  model. That slice should allowlist-construct from the constants rather than
  validate-and-pass raw report rows.

Parked hardening: none.

## Verification

- Focused generator pytest: `python -m pytest`, target
  `tests/test_generate_deflection_frontend_contract_types.py` -- 19 passed.
- Generator drift check: `python`, target
  `scripts/generate_deflection_frontend_contract_types.py`, with `--check` --
  all four generated frontend contract outputs are current.
- Local PR review bundle -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Report-Hosted-Safe-Artifact.md` | 109 |
| `portfolio-ui/api/content-ops/deflection/report-model-contract.js` | 64 |
| `portfolio-ui/src/types/deflectionReportModel.ts` | 64 |
| `scripts/generate_deflection_frontend_contract_types.py` | 88 |
| `tests/test_generate_deflection_frontend_contract_types.py` | 108 |
| **Total** | **433** |
