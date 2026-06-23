# PR-Deflection-Report-TTL-Purge

## Why this slice exists

Issue #1656's deletion-path spec made the Atlas delete endpoint the blocking
backend half for the public 30-day deletion claim, then explicitly called out
an optional fast-follow: Atlas should also own a 30-day TTL so report expiry
does not depend on a single portfolio-side cron.

The root cause is not missing row-deletion mechanics anymore: #1806 added the
request-scoped delete endpoint, and the retention CLI/store already support
guarded cut-off deletion. The remaining root is that no production owner invokes
that Atlas retention path on a schedule. This slice fixes that root by adding an
Atlas-owned scheduled/manual workflow that calls the existing guarded purge
script with a 30-day window and bounded batches, without adding paid
infrastructure.

Review on #1809 exposed three cross-path gaps around that orchestrator:
manual-dispatch refs could run branch code with the production DB secret, the
report retention delete did not remove companion delivery metadata, and checkout
authorization could start a new Stripe session inside the final 24 hours before
the scheduled 30-day purge. The upstream fix is in the Atlas workflow,
retention store, and checkout-authorization guard, not in a downstream waiver.
This is over the 400-line soft cap because the review fix spans the scheduled
workflow, the shared storage boundary, checkout authorization, and their
regression tests; splitting those would leave at least one deletion-safety
thread unresolved on an open PR.

## Scope (this PR)

Ownership lane: security/hardening-1656
Slice phase: Production hardening

Max files: 9

1. Add a scheduled/manual GitHub Actions workflow for
   `content_ops_deflection_reports` TTL purge.
2. Run the existing `scripts/purge_content_ops_deflection_reports.py` with a
   default 30-day retention window, default 1,000-row batch limit, JSON output,
   and confirmed deletion on scheduled runs.
3. Keep the database URL out of argv by passing only the secret-backed
   environment variable name to the purge script.
4. Reject non-main manual workflow refs before the purge job reaches the DB
   secret, and explicitly check out the default branch for the script.
5. Delete matching `content_ops_deflection_report_deliveries` rows whenever the
   retention path or request-scoped delete removes report rows.
6. Reject checkout authorization when a report is already inside the final
   24-hour open-session grace window before 30-day retention expiry.
7. Add workflow/store/control-surface tests and enroll the workflow-contract
   test in the pre-push audit workflow.

### Files touched

- `.github/workflows/content_ops_deflection_report_ttl_purge.yml`
- `.github/workflows/pre_push_audit.yml`
- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Report-TTL-Purge.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_deflection_report_ttl_workflow.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_pre_push_audit_workflow.py`

### Review Contract

Acceptance criteria:

- The TTL purge workflow runs only on `schedule` and manual
  `workflow_dispatch`, not on PR or push events.
- The scheduled path deletes reports older than 30 days by default and bounds
  each run to a default 1,000-row batch.
- Manual dispatch can opt into a dry run that omits `--confirm-delete`.
- The workflow passes `--database-url-env EXTRACTED_DATABASE_URL`, not a raw
  database URL on argv.
- Manual workflow dispatches on non-main refs do not reach the secret-bearing
  purge step, and checkout uses the default-branch script.
- Workflow permissions are minimal and external Actions are SHA-pinned.
- Report retention deletes matching delivery rows so payment/delivery metadata
  is not orphaned after the report row expires.
- Checkout authorization rejects reports inside the final 24-hour grace window
  before the 30-day purge cutoff, and fails closed when report age is unknown.
- The workflow-contract test is enrolled in pre-push audit CI.

Affected surfaces:

- GitHub Actions workflow config.
- Atlas deflection report purge/delete storage boundary.
- Atlas deflection report checkout-authorization control surface.
- PR-review tooling test enrollment.

Risk areas:

- Data deletion from an accidentally broad or unbounded scheduled job.
- Secret exposure through command-line arguments or logs.
- Paid-but-missing reports when a checkout session is opened too close to the
  retention cutoff.
- Orphaned paid delivery metadata after report expiry.
- CI drift if the workflow-contract tests are not enrolled.

Reviewer rules triggered: R1, R2, R3, R6, R7, R8, R10, R11, R12, R14.

## Mechanism

The new workflow runs daily at 08:23 UTC and can also be started manually. It
installs the repo runtime dependencies plus `asyncpg`, then invokes:

```bash
python scripts/purge_content_ops_deflection_reports.py \
  --database-url-env EXTRACTED_DATABASE_URL \
  --retention-days "$retention_days" \
  --limit "$delete_limit" \
  --json \
  --confirm-delete
```

For scheduled runs, the workflow falls back to `retention_days=30`,
`delete_limit=1000`, and `dry_run=false`, so `--confirm-delete` is appended. For
manual dispatch, `dry_run=true` leaves `--confirm-delete` out and the existing
purge runner reports the eligible count without deleting rows.

The purge job is restricted to scheduled runs and manual runs on `refs/heads/main`.
The checkout action also pins `ref` to the repository default branch, so the
secret-bearing purge step runs the merged purge script rather than branch code.

The workflow relies on the retention runner's existing fail-closed checks:
exactly one database URL source, positive retention, positive limit, preflighted
output, and no payload exposure in JSON output. The Postgres retention delete
now builds a `doomed` report set, deletes matching delivery rows first, then
deletes the report rows and returns the report deletion count. The
request-scoped delete path uses the same target-then-delivery-then-report
shape.

Checkout authorization reads `created_at` from the artifact record and rejects a
new checkout when `created_at + 30 days - 24 hours` has passed. That leaves room
for Stripe's default open-session window before the scheduled purge can remove
the row.

## Intentional

- No paid infrastructure. GitHub Actions is already used by this public repo
  and satisfies #1656's zero-paid-dependency constraint.
- No portfolio copy or cron changes. The companion repo still needs its own
  coordination, but Atlas now has an independent expiry backstop.
- No database URL on argv. The workflow passes the name of the secret-backed
  environment variable so process arguments and command logs do not carry the
  DSN.
- No schema migration in this slice. Delivery rows do not currently have a
  foreign key, so the narrow safe fix is an explicit delete in the existing
  retention/delete paths.

## Deferred

- #1656 / atlas-portfolio#313: keep the portfolio cleanup path coordinated
  with the Atlas delete endpoint and public security copy.
- #1656 medium-tier follow-ups remain separate: RLS, app-level encryption,
  alert delivery, token-auth removal, scanner ratchets, IR docs, structured
  logging, security.txt, and CVE SLA.

Parked hardening: none.

## Verification

- Command: python -m py_compile extracted_content_pipeline/deflection_report_access.py extracted_content_pipeline/api/control_surfaces.py tests/test_deflection_report_ttl_workflow.py tests/test_content_ops_deflection_report.py tests/test_extracted_content_control_surface_api.py
  -- passed.
- Command: python -m pytest tests/test_deflection_report_ttl_workflow.py tests/test_content_ops_deflection_report.py::test_postgres_delete_report_scopes_to_account_and_request_id tests/test_content_ops_deflection_report.py::test_postgres_report_retention_uses_cutoff_and_limited_delete tests/test_content_ops_deflection_report.py::test_postgres_report_retention_fails_closed_on_unparseable_delete_count tests/test_content_ops_deflection_report.py::test_retention_runner_requires_confirm_delete_and_valid_bounds tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_returns_canonical_terms_only tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_rejects_reports_inside_retention_grace_window tests/test_extracted_content_control_surface_api.py::test_deflection_checkout_authorization_fails_closed_without_usable_report_age -q
  -- 14 passed.
- Command: python -m pytest tests/test_deflection_report_ttl_workflow.py tests/test_pre_push_audit_workflow.py tests/test_content_ops_deflection_report.py tests/test_extracted_content_control_surface_api.py -q
  -- 315 passed, 1 skipped.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  -- passed.
- Command: bash scripts/check_ascii_python.sh
  -- passed.
- Command: python scripts/audit_workflow_security_posture.py .github/workflows
  -- passed with existing repository-wide WARN findings only.
- Command: python scripts/maturity_sweep.py extracted_content_pipeline --tests-root tests --baseline tests/maturity_sweep/baseline_extracted_content_pipeline.json --top 25
  -- passed; ratchet gate reported no new brittleness above baseline.
- Command: git diff --check -- passed.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Report-TTL-Purge.md --check
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/content_ops_deflection_report_ttl_purge.yml` | 73 |
| `.github/workflows/pre_push_audit.yml` | 2 |
| `extracted_content_pipeline/api/control_surfaces.py` | 34 |
| `extracted_content_pipeline/deflection_report_access.py` | 83 |
| `plans/PR-Deflection-Report-TTL-Purge.md` | 198 |
| `tests/test_content_ops_deflection_report.py` | 41 |
| `tests/test_deflection_report_ttl_workflow.py` | 77 |
| `tests/test_extracted_content_control_surface_api.py` | 50 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| **Total** | **564** |
