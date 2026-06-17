# PR-Deflection-Process-Freshness-Contract

## Why this slice exists

#1612 is now green on the ATLAS-side paid artifact path after #1682, but the
root cause of the prior #1671/#1678 failures was a stale hosted API process
serving legacy deflection artifact behavior while current source already had
the correct contract. The live runner discovered that only after creating a
fresh request, paying it, rendering a PDF, and fetching paid endpoints.

Root cause: the public deflection API had no cheap, route-level process
contract marker that proved the running process included the current paid
artifact/report-model contract before the live proof began. The first version
of that marker also treated `deflection.v1` as a sufficient proxy for the
report model shape; that was still a false-green risk if fields changed without
a schema-version bump.

This change fixes that root inside the ATLAS service boundary by adding a
small process contract endpoint on the deflection report router and a preflight
checker for live-proof operators. The endpoint/checker now bind freshness to
the actual report-model contract shape derived from the section registry, not
only version strings. It does not try to solve deployment orchestration,
process supervision, or atlas-portfolio buyer-page proof.

Diff-size note: this intentionally exceeds the 400 LOC target because the new
guard script needs negative fixtures for missing route, malformed response,
wrong schema, partial requirements, and localhost rejection. Trimming those
would weaken the thing this slice is here to protect.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Production hardening

1. Add a public deflection report process-contract endpoint beside the existing
   snapshot/artifact/report-model routes. The response advertises the contract
   schema version, paid artifact requirements, report model schema, report
   model contract shape, evidence export schema, and current deflection route
   paths.
2. Add an operator preflight checker that fetches the process contract from a
   hosted API base and fails closed on missing route, non-JSON response, schema
   mismatch, report-model shape drift, or missing paid artifact requirements.
3. Add focused tests for the endpoint payload, report-model shape snapshot, and
   checker failure paths.
4. Enroll the new checker test in extracted pipeline checks and workflow path
   filters.

### Review Contract
- Acceptance criteria:
  - [ ] `/content-ops/deflection-reports/process-contract` exists on the same
        router as the paid artifact/report-model endpoints.
  - [ ] The process contract declares `deflection_report_process.v1`,
        `deflection.v1`, `deflection_evidence.v1`, and explicit requirements
        for object `report_model` and object `evidence_export`.
  - [ ] The process contract exposes the current `deflection.v1` report-model
        shape, and the checker fails when the version string matches but that
        shape drifts.
  - [ ] The checker exits non-zero when the route is absent, returns malformed
        JSON, has the wrong schema, or omits required artifact capabilities.
  - [ ] The checker succeeds for the current contract without needing a real
        paid request ID, token payload disclosure, PDF bytes, Stripe webhook, or
        database write.
  - [ ] Existing paid artifact/report-model route behavior is unchanged.
- Affected surfaces: extracted content-ops API router, operator smoke script,
  CI enrollment.
- Risk areas: public API compatibility, false-green deployment checks, CI
  coverage.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R14.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Process-Freshness-Contract.md`
- `scripts/check_deflection_process_contract.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_check_deflection_process_contract.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_control_surface_api.py`

## Mechanism

The router exposes a deterministic contract payload that is independent of a
specific report row. The payload is intentionally narrow: it says which
deflection report route contract this process knows how to serve, not whether a
given request is paid or whether the database contains a row.

The report module derives a small JSON-ready `report_model_contract` from the
`DEFLECTION_REPORT_SECTION_REGISTRY`: top-level report fields, section fields,
section IDs, titles, priorities, surfaces, default limits, and required data
keys. That shape is bound to `DEFLECTION_REPORT_SCHEMA_VERSION` with a snapshot
test, so changing the model structure without consciously updating the version
or snapshot turns red.

The checker joins the operator-provided base URL with the process-contract
path, attaches the same bearer token style used by the live runner when
provided, fetches JSON, and validates exact contract fields, including the
derived report-model shape. Its output is a small JSON status that is safe to
commit if needed because it does not include request IDs, result URLs, source
IDs, raw artifacts, PDF bytes, Stripe IDs, or tokens.

This gives the #1612 live runner sequence a fast first gate: if the hosted
process is stale, the process-contract check fails as a missing route or
contract mismatch before the operator spends a paid proof run.

## Intentional

- No deployment-process restart automation in this slice. The safe upstream
  boundary here is detection; deciding how to restart or supervise the public
  process is an operations/deployment slice.
- No atlas-portfolio buyer hosted-result proof. #1612 already records that the
  buyer URL is owned by `canfieldjuan/atlas-portfolio`, not this ATLAS service
  slice.
- No paid request lookup in the process-contract endpoint. That would recreate
  the expensive proof path this slice is meant to avoid.
- The process contract is public through the same deflection dependencies as
  snapshot/artifact/report-model routes. It exposes schema/route capability, not
  customer data.

## Deferred

- Wire the process-contract checker into the live proof runbook or wrapper as
  the first operator step after this PR lands.
- Deployment supervision/restart freshness automation remains a separate
  operations hardening slice if the public Funnel process can still outlive
  current source unexpectedly.
- atlas-portfolio buyer hosted-result proof remains outside this repo/lane.

Parked hardening: none.

## Verification

- `pytest tests/test_content_ops_deflection_report.py::test_deflection_report_model_contract_shape_requires_version_bump tests/test_extracted_content_control_surface_api.py::test_deflection_process_contract_route_advertises_paid_artifact_contract tests/test_check_deflection_process_contract.py -q`
  - 10 passed.
- Python compile check for the checker script, control-surface router, and
  changed report/checker/API test files.
  - passed.
- Extracted pipeline CI enrollment audit.
  - OK: 185 matching tests are enrolled.
- Extracted pipeline check bundle.
  - extracted reasoning core: 295 passed.
  - extracted content pipeline: 4627 passed, 10 skipped, 1 existing torch warning.
- Local PR review wrapper.
  - passed after this plan stopped presenting verification commands as fake
    path claims to the consistency audit.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `extracted_content_pipeline/api/control_surfaces.py` | 36 |
| `extracted_content_pipeline/faq_deflection_report.py` | 41 |
| `plans/PR-Deflection-Process-Freshness-Contract.md` | 163 |
| `scripts/check_deflection_process_contract.py` | 233 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_check_deflection_process_contract.py` | 240 |
| `tests/test_content_ops_deflection_report.py` | 107 |
| `tests/test_extracted_content_control_surface_api.py` | 39 |
| **Total** | **863** |
