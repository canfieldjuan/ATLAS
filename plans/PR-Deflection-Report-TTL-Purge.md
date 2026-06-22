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
infrastructure or new database deletion logic.

## Scope (this PR)

Ownership lane: security/hardening-1656
Slice phase: Production hardening

1. Add a scheduled/manual GitHub Actions workflow for
   `content_ops_deflection_reports` TTL purge.
2. Run the existing `scripts/purge_content_ops_deflection_reports.py` with a
   default 30-day retention window, default 1,000-row batch limit, JSON output,
   and confirmed deletion on scheduled runs.
3. Keep the database URL out of argv by passing only the secret-backed
   environment variable name to the purge script.
4. Add workflow-contract tests and enroll them in the pre-push audit workflow.
5. Leave the existing purge script, access store, and API endpoint behavior
   unchanged.

### Files touched

- `.github/workflows/content_ops_deflection_report_ttl_purge.yml`
- `.github/workflows/pre_push_audit.yml`
- `plans/PR-Deflection-Report-TTL-Purge.md`
- `tests/test_deflection_report_ttl_workflow.py`
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
- Workflow permissions are minimal and external Actions are SHA-pinned.
- The workflow-contract test is enrolled in pre-push audit CI.

Affected surfaces:

- GitHub Actions workflow config.
- Existing Atlas deflection report purge CLI invocation.
- PR-review tooling test enrollment.

Risk areas:

- Data deletion from an accidentally broad or unbounded scheduled job.
- Secret exposure through command-line arguments or logs.
- CI drift if the workflow-contract tests are not enrolled.

Reviewer rules triggered: R1, R2, R3, R6, R7, R8, R11, R12, R14.

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

The workflow relies on the retention runner's existing fail-closed checks:
exactly one database URL source, positive retention, positive limit, preflighted
output, and no payload exposure in JSON output. This PR adds tests around the
workflow wrapper rather than duplicating those purge mechanics.

## Intentional

- No new purge code. The retention logic is already covered in
  `tests/test_content_ops_deflection_report.py`; this slice wires the
  production invoker and tests that wrapper contract.
- No paid infrastructure. GitHub Actions is already used by this public repo
  and satisfies #1656's zero-paid-dependency constraint.
- No portfolio copy or cron changes. The companion repo still needs its own
  coordination, but Atlas now has an independent expiry backstop.
- No database URL on argv. The workflow passes the name of the secret-backed
  environment variable so process arguments and command logs do not carry the
  DSN.

## Deferred

- #1656 / atlas-portfolio#313: keep the portfolio cleanup path coordinated
  with the Atlas delete endpoint and public security copy.
- #1656 medium-tier follow-ups remain separate: RLS, app-level encryption,
  alert delivery, token-auth removal, scanner ratchets, IR docs, structured
  logging, security.txt, and CVE SLA.

Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_deflection_report_ttl_workflow.py tests/test_pre_push_audit_workflow.py -q
  -- 14 passed.
- Command: python -m pytest tests/test_content_ops_deflection_report.py::test_retention_runner_requires_confirm_delete_and_valid_bounds tests/test_content_ops_deflection_report.py::test_retention_runner_resolves_database_url_from_non_argv_sources tests/test_content_ops_deflection_report.py::test_retention_runner_dry_run_counts_without_deleting_or_exposing_payload -q
  -- 3 passed.
- Command: python scripts/audit_workflow_security_posture.py .github/workflows
  -- passed with existing repository-wide WARN findings only.
- Command: git diff --check -- passed.
- Command: python scripts/sync_pr_plan.py plans/PR-Deflection-Report-TTL-Purge.md --check
  -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/content_ops_deflection_report_ttl_purge.yml` | 70 |
| `.github/workflows/pre_push_audit.yml` | 2 |
| `plans/PR-Deflection-Report-TTL-Purge.md` | 138 |
| `tests/test_deflection_report_ttl_workflow.py` | 67 |
| `tests/test_pre_push_audit_workflow.py` | 6 |
| **Total** | **283** |
