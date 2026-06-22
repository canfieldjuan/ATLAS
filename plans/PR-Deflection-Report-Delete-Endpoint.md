# PR-Deflection-Report-Delete-Endpoint

## Why this slice exists

Issue #1656 H1 says the portfolio cleanup can delete uploaded blobs and Neon
rows, but the ATLAS-derived deflection report row still has no request-scoped
delete path. The existing ATLAS retention helper deletes old rows by age, which
is useful for a future TTL job, but it does not let the portfolio cleanup delete
the specific derived report for a completed 30-day purge.

Root cause: `DeflectionReportArtifactStore` exposes read, paid-state, and
age-retention operations, but not the product cleanup operation keyed by the
authenticated service-token account plus `request_id`. The companion freshness
checker also validates the process contract shape without validating advertised
routes, so it could greenlight a stale hosted process that lacks the DELETE
capability. This PR fixes those roots for the backend half by adding an
idempotent same-account delete operation, a matching DELETE route, and
checker coverage that fails closed when that route is absent or stale.

## Scope (this PR)

Ownership lane: security/hardening-1656
Slice phase: Production hardening
Max files: 8

1. Add `delete_report(account_id, request_id)` to the deflection report store
   protocol and both in-memory/Postgres implementations.
2. Add `DELETE /api/v1/content-ops/deflection-reports/{request_id}` through the
   same public deflection dependencies and scope account resolution used by
   submit/read routes.
3. Expose the delete route in the deflection process contract so the portfolio
   cleanup side can discover the backend purge URL.
4. Prove same-account delete removes the report, repeated/missing deletes return
   the same 204-shaped result, and cross-account delete does not remove another
   account's report.
5. Teach the process-contract freshness checker to require the advertised route
   map, including `routes.delete`, so hosted stale builds fail preflight.

### Review Contract

- Acceptance criteria:
  - [ ] Deleting by the authenticated scope account plus `request_id` removes
        the stored snapshot/artifact row.
  - [ ] Deleting a missing report is idempotent and returns the same no-body
        success shape as deleting an existing report.
  - [ ] A cross-account delete attempt cannot remove another tenant's report.
  - [ ] The Postgres adapter issues a bounded
        `DELETE FROM content_ops_deflection_reports WHERE account_id = $1 AND request_id = $2`.
  - [ ] The route uses the same deflection public dependency lane as submit and
        snapshot/artifact reads.
- Affected surfaces: extracted deflection-report store, Content Ops control
  surface API, process contract metadata, process-contract freshness checker,
  tests.
- Risk areas: data deletion, tenant isolation, retry/idempotency, API
  backcompat.
- Reviewer rules triggered: R1, R2, R3, R5, R8, R10, R12, R14. Boundary-probe
  required because this is a data-deletion guard.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Report-Delete-Endpoint.md`
- `scripts/check_deflection_process_contract.py`
- `tests/test_check_deflection_process_contract.py`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_run_deflection_full_report_qa_live_runner.py`

## Mechanism

The store protocol gets a request-scoped `delete_report` method. The in-memory
implementation deletes the `(account_id, request_id)` row and removes any
in-memory delta rows that reference it, mirroring the Postgres `ON DELETE
CASCADE` foreign keys. The Postgres implementation issues a single tenant-bound
`DELETE` and parses the command count for tests/diagnostics. The FastAPI router
exposes a 204 DELETE endpoint that resolves scope with
`_required_scope_account_id(scope)`, calls the store, and intentionally ignores
the bool result so existing and missing rows are indistinguishable.
The freshness checker validates the hosted process contract's route map against
the current required paths; missing or stale route capabilities fail preflight
before the companion cleanup job can rely on them.

## Intentional

- This slice does not purge existing rows or add a scheduler. Issue #1656's
  deletion spec explicitly says not to purge a backlog; a TTL/cron job remains
  defense-in-depth after the request-scoped API exists.
- The endpoint returns 204 for both existing and missing rows to avoid existence
  leaks and to make portfolio cleanup retries safe.
- The process-contract checker validates literal route templates rather than
  probing every route method because the endpoint's job is a cheap freshness
  preflight; route behavior is covered by the API/store tests in this slice.

## Deferred

- ATLAS-side scheduled 30-day TTL purge, likely by wiring the existing
  retention helper into a production job after this endpoint lands.
- Portfolio cleanup integration in the companion atlas-portfolio issue so its
  cron calls this ATLAS delete route after Blob/Neon cleanup.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py::test_in_memory_report_retention_deletes_only_old_rows_with_limit tests/test_content_ops_deflection_report.py::test_postgres_report_retention_uses_cutoff_and_limited_delete tests/test_content_ops_deflection_report.py::test_in_memory_deflection_report_store_deletes_report_and_referencing_deltas tests/test_content_ops_deflection_report.py::test_postgres_delete_report_scopes_to_account_and_request_id tests/test_extracted_content_control_surface_api.py::test_deflection_process_contract_route_advertises_paid_artifact_contract tests/test_extracted_content_control_surface_api.py::test_deflection_report_delete_route_is_idempotent_and_scoped tests/test_extracted_content_control_surface_api.py::test_deflection_report_routes_use_public_and_trusted_dependencies -q` -- 7 passed.
- `python -m pytest tests/test_check_deflection_process_contract.py -q` -- 10 passed.
- `python -m pytest tests/test_run_deflection_full_report_qa_live_runner.py::test_live_runner_extracts_pdf_text_from_renderer_bytes_by_default tests/test_run_deflection_full_report_qa_live_runner.py::test_process_contract_drift_fails_before_paid_artifact_fetches -q` -- 2 passed.
- `python -m pytest tests/test_run_deflection_full_report_qa_live_runner.py -q` -- 17 passed.
- Python compile check for the changed implementation, checker, runner, and
  test modules -- passed.
- `git diff --check` -- passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Report-Delete-Endpoint.md --check` -- passed.
- Pending before push:
  - `bash scripts/push_pr.sh /tmp/PR-Deflection-Report-Delete-Endpoint.md -u origin HEAD`

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 17 |
| `extracted_content_pipeline/deflection_report_access.py` | 51 |
| `plans/PR-Deflection-Report-Delete-Endpoint.md` | 129 |
| `scripts/check_deflection_process_contract.py` | 32 |
| `tests/test_check_deflection_process_contract.py` | 51 |
| `tests/test_content_ops_deflection_report.py` | 66 |
| `tests/test_extracted_content_control_surface_api.py` | 44 |
| `tests/test_run_deflection_full_report_qa_live_runner.py` | 1 |
| **Total** | **391** |
