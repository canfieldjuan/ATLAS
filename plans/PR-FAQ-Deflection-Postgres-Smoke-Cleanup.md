# PR-FAQ-Deflection-Postgres-Smoke-Cleanup

## Why this slice exists

`PR-FAQ-Deflection-Paid-Postgres-Smoke` added a guarded live/ephemeral Postgres
smoke for the FAQ deflection paid gate and deferred an optional cleanup mode.
Operators can already inspect or manually remove the returned `smoke-...` row,
but repeated smoke runs should not require manual database cleanup when the
paid-gate proof succeeds.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Production hardening

1. Add `--cleanup-on-success` to the existing Postgres smoke.
2. Delete only the exact tenant/request row created by the smoke, and only when
   the request id is still an ephemeral `smoke-...` id.
3. Report cleanup intent and result in the JSON payload.
4. Cover cleanup success and cleanup miss/failure behavior with focused tests.

### Files touched

- `plans/PR-FAQ-Deflection-Postgres-Smoke-Cleanup.md` -- slice plan.
- `scripts/smoke_content_ops_deflection_paid_postgres.py` -- optional cleanup
  mode for successful smoke rows.
- `tests/test_faq_deflection_paid_postgres_smoke.py` -- cleanup regression
  coverage.

## Mechanism

When `--cleanup-on-success` is present, the smoke runs the same guarded paid
flow first. After the final reread proves the row is paid, the script executes a
targeted delete:

```sql
DELETE FROM content_ops_deflection_reports
WHERE account_id = $1
  AND request_id = $2
  AND request_id LIKE 'smoke-%'
```

The `smoke-` predicate is duplicated in SQL so a later bug in Python request-id
handling cannot delete non-ephemeral rows. Cleanup runs only after the smoke
has succeeded; preflight skips and failed paid-gate checks do not delete
anything.

## Intentional

- No general cleanup command. This only cleans up the row created by a
  successful smoke invocation.
- Cleanup miss is reported as a failed smoke result because the operator asked
  for cleanup and the row remained or could not be confirmed deleted.
- The script keeps direct SQL for cleanup rather than widening the shared
  deflection report store port for an operator-only maintenance behavior.

## Deferred

- Future robust-testing slice: hosted browser/API validation for the portfolio
  result page once it consumes the merged Checkout contract.
- Parked hardening considered: none in `HARDENING.md` touch this cleanup slice.

## Verification

- `python -m pytest tests/test_faq_deflection_paid_postgres_smoke.py -q`
  (7 passed, 1 warning from host torch import)
- `python -m py_compile scripts/smoke_content_ops_deflection_paid_postgres.py tests/test_faq_deflection_paid_postgres_smoke.py`
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Postgres-Smoke-Cleanup.md`
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Postgres-Smoke-Cleanup.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py`
- `bash scripts/check_ascii_python.sh`
- `git diff --check`
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-deflection-postgres-smoke-cleanup.md`

## Estimated diff size

| Area | Estimate |
|---|---:|
| Plan | ~85 |
| Smoke script | ~50 |
| Tests | ~55 |
| Total | ~190 |
