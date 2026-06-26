# PR-Deflection-Delta-Paid-Report-Ordering

## Why this slice exists

#1798 added the opt-in monthly deflection delta automation and fixed paid
account discovery to rank tenants by paid activity. A follow-up Copilot review
found the same coverage risk one level lower: after an account is discovered,
`compute_and_save_recent_deflection_deltas` calls
`store.list_reports(..., paid=True, limit=reports_per_account)`, but the
Postgres implementation still orders paid reports by `created_at DESC`.

Root cause: the report listing boundary has only one ordering policy, newest
created report first, even when the caller explicitly asks for paid reports.
That means an older report unlocked today can sit outside the
`reports_per_account` window until live cron pagination exists. This PR fixes
that root for the paid listing window by ordering paid report scans by paid
activity while keeping non-paid/all report listings creation-ordered.

This branch also archives the just-merged #1798 plan as the AGENTS.md teardown
step, folded into this follow-up branch because direct `main` housekeeping is
not available in this connector/runtime.

## Scope (this PR)

Ownership lane: issue-1316/deflection-delta-paid-report-ordering
Slice phase: Hardening slice

1. Order paid report listing by paid activity in the in-memory and Postgres
   deflection report artifact stores.
2. Add regression tests for a recently paid older report staying inside the
   bounded paid report window.
3. Preserve existing `list_reports` ordering for `paid=None` and `paid=False`.
4. Archive the merged #1798 plan and refresh the plans index.

### Review Contract

Acceptance criteria:
- `list_reports(account_id=..., paid=True, limit=N)` prefers reports with the
  newest `paid_at`, falling back to `updated_at`/`created_at` in Postgres and to
  in-memory timestamps in test parity.
- Non-paid/all report listings keep their existing newest-created-first
  behavior.
- The monthly batch naturally benefits because it already calls
  `list_reports(..., paid=True)`.
- No live cron enablement, pagination, alerting, delivery, or entitlement logic
  is added in this slice.

Affected surfaces:
- `extracted_content_pipeline.deflection_report_access`
- `tests.test_content_ops_deflection_delta_persistence`
- `plans/`

Risk areas:
- Store protocol parity between in-memory and Postgres implementations.
- Accidentally changing free/unpaid report listing order.
- Treating ordering as a complete replacement for future cap pagination.

Reviewer rules triggered: R1, R8, R10, R14.

### Files touched

- `extracted_content_pipeline/deflection_report_access.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Paid-Report-Ordering.md`
- `plans/archive/PR-Deflection-Delta-Monthly-Automation.md`
- `tests/test_content_ops_deflection_delta_persistence.py`

## Mechanism

- In-memory `list_reports` sorts matching rows by created time for the existing
  all/unpaid behavior, and switches to paid-activity time when `paid=True`.
  Ties remain deterministic by request ID.
- Postgres `list_reports` keeps `ORDER BY created_at DESC, request_id ASC` for
  all/unpaid reads and uses
  `ORDER BY COALESCE(paid_at, updated_at, created_at) DESC, request_id ASC`
  for paid reads.
- Focused tests pin the bounded-window case where a recently paid older report
  should beat a newer report with older paid activity, plus a Postgres SQL
  assertion for the paid ordering clause.

## Intentional

- This does not add pagination or cap-saturation alerting. It closes the
  newly-paid older-report miss under the current bounded scan, while broader
  cap coverage remains a live-cron prerequisite.
- The readonly MCP search surface also uses `list_reports(..., paid=True)`, so
  explicit paid-only search results now use paid-activity ordering. That is an
  intentional, benign parity change; default search ordering with `paid=None`
  remains created-date ordered.
- This does not alter baseline selection, delta identity, delivery copy, or
  subscription entitlement logic.

## Deferred

- Paginate or alert on saturated `account_limit` / `reports_per_account`
  windows before enabling live monthly cron.
- D0 stable identity foundation from #1316 remains the gate before
  customer-facing delta delivery.
- Monthly delivery email, customer-facing copy, and entitlement checks remain
  out of scope.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_content_ops_deflection_delta_persistence.py -q`
  -- 15 passed.
- `python -m pytest tests/test_deflection_delta_automation_task.py -q` -- 10
  passed after installing the repo-declared `apscheduler` dependency into the
  bundled Python runtime.
- `python -m py_compile extracted_content_pipeline/deflection_report_access.py`
  -- passed.
- `python scripts/archive_plans.py index` -- passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Paid-Report-Ordering.md --check`
  -- passed.
- `git diff --check` -- passed; Git emitted Windows checkout LF-to-CRLF
  warnings for touched Python files.
- Pending before merge: GitHub Actions on the opened PR.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/deflection_report_access.py` | 33 |
| `plans/INDEX.md` | 141 |
| `plans/PR-Deflection-Delta-Paid-Report-Ordering.md` | 129 |
| `plans/archive/PR-Deflection-Delta-Monthly-Automation.md` | 0 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 96 |
| **Total** | **399** |
