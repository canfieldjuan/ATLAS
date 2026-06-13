# PR-Deflection-Zendesk-Tenant-Export-Route

## Why this slice exists

#1526 added the pure Zendesk API exporter that produces the full-thread
`{tickets: [...]}` artifact, and #1520 already lets `/deflection-reports/submit`
ingest that artifact. The remaining gap is a hosted, tenant-scoped route that
uses stored Zendesk credentials to produce the artifact without requiring the
operator or browser to handle API tokens. This is the next thin vertical slice
before wiring the portfolio UI to start from stored Zendesk credentials.
This PR is over the 400 LOC soft cap because the hosted route, API registration,
CI enrollment, review-fix error classification, and the non-happy-path route
tests need to ship together to avoid a token-leaking or cross-tenant false-green
export surface.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a Content Ops Zendesk export router that returns the full-thread artifact
   for the authenticated tenant.
2. Resolve credentials only through `content_ops_zendesk_credentials` tenant
   storage. Missing tenant credentials fail closed with no config/global
   fallback.
3. Register the route with the existing Content Ops auth bridge and DB pool.
4. Add focused API tests with a mocked exporter/transport; no live Zendesk call
   in CI.

### Review Contract
- Acceptance criteria:
  - [ ] The route requires the existing Content Ops auth dependency and uses the
        authenticated account id for credential lookup.
  - [ ] A tenant without stored Zendesk credentials gets a deterministic
        `404`/missing-credentials response and the exporter is not called.
  - [ ] Exporter failures are mapped to sanitized HTTP errors without exposing
        tokens, headers, or raw Zendesk payloads.
  - [ ] Request bounds (`limit`, `start_time`) are validated before any Zendesk
        call.
  - [ ] Tests mock the exporter boundary and assert request shape; CI does not
        call live Zendesk.
- Affected surfaces: host Content Ops Zendesk export API, API router
  registration, route-focused tests.
- Risk areas: cross-tenant data access, unscoped credential fallback, token
  leakage, route registration drift, live network in CI.
- Reviewer rules triggered: R1, R2, R3, R5, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_content_ops_generated_assets_checks.yml`
- `atlas_brain/_content_ops_zendesk_credentials.py`
- `atlas_brain/api/__init__.py`
- `atlas_brain/api/content_ops_zendesk_export.py`
- `plans/PR-Deflection-Zendesk-Tenant-Export-Route.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_atlas_content_ops_generated_assets_api.py`
- `tests/test_content_ops_zendesk_credentials.py`
- `tests/test_content_ops_zendesk_export_api.py`

## Mechanism

`create_content_ops_zendesk_export_router(...)` accepts a pool provider, auth
dependency, credential lookup function, and exporter function. The production
router resolves the authenticated account UUID, asks tenant credential storage
for that account's active Zendesk credentials, and calls
`export_zendesk_full_thread_artifact(credentials, limit=..., start_time=...)`.

The response is the exporter artifact plus small display-safe metadata
(`ticket_count`, `limit`, `start_time`, `importer_mode`). The route never
returns credentials, authorization headers, or raw transport errors. It maps
missing stored credentials to a deterministic HTTP response and maps
`ZendeskTicketExportError` to sanitized HTTP errors by error code. Credential
lookup/decrypt outages raise a typed lookup error so the route returns `503`
instead of mislabeling the outage as missing credentials, and transient
`httpx.RequestError` failures from the exporter return a sanitized `502`.

## Intentional

- No portfolio UI button in this PR. This slice proves the hosted API boundary;
  the UI can wire to it next.
- No live Zendesk call in tests. The exporter was already covered with a mocked
  transport in #1526; this route test mocks the exporter function to prove
  tenant scoping and HTTP behavior.
- No config-backed Zendesk fallback. Hosted deflection export is always
  tenant-scoped because it can pull real customer ticket data.

## Deferred

- Portfolio UI flow that starts the export from stored Zendesk credentials and
  hands the returned artifact into the existing private Blob/full-thread submit
  path.
- Optional guarded live trial-account smoke against the route after deployment.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_zendesk_export_api.py tests/test_content_ops_zendesk_credentials.py tests/test_atlas_content_ops_generated_assets_api.py -q`
  -- 37 passed, 1 existing torch/pynvml warning.
- `bash` `scripts/run_extracted_pipeline_checks.sh` -- 4062 passed, 10 skipped,
  1 existing torch/pynvml warning.
- `python` `scripts/audit_extracted_pipeline_ci_enrollment.py` -- OK, 176
  matching tests enrolled.
- `python` `scripts/audit_review_rules_triggered.py` --plan
  `plans/PR-Deflection-Zendesk-Tenant-Export-Route.md` origin/main
  -- OK, plan declares every triggered rule.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_generated_assets_checks.yml` | 8 |
| `atlas_brain/_content_ops_zendesk_credentials.py` | 11 |
| `atlas_brain/api/__init__.py` | 8 |
| `atlas_brain/api/content_ops_zendesk_export.py` | 201 |
| `plans/PR-Deflection-Zendesk-Tenant-Export-Route.md` | 121 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_atlas_content_ops_generated_assets_api.py` | 18 |
| `tests/test_content_ops_zendesk_credentials.py` | 33 |
| `tests/test_content_ops_zendesk_export_api.py` | 343 |
| **Total** | **744** |
