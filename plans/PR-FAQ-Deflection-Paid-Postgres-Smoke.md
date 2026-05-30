# PR-FAQ-Deflection-Paid-Postgres-Smoke

## Why this slice exists

`PR-FAQ-Deflection-Paid-Flow-Validation` proved the free snapshot, Stripe
completion helper, and paid artifact release flow with an in-memory store. Its
deferred validation gap is a live/ephemeral Postgres smoke that uses the real
`content_ops_deflection_reports` adapter and the billing paid-flag helper
together. The smoke must be guarded because it writes paid-state rows, even when
the row is an ephemeral smoke fixture.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Functional validation

1. Add an operator smoke script that seeds one ephemeral deflection report row
   in Postgres, verifies it is locked, routes a fake paid Checkout session
   through the existing billing helper, and verifies the row unlocks.
2. Require explicit `--database-url`, `--account-id`, and
   `--confirm-postgres-write` before opening a pool or writing data.
3. Fail closed when the request id is not an ephemeral `smoke-...` id or when
   a generated smoke request id already exists.
4. Cover the guarded branches and fake Postgres success path with focused
   tests in the main suite.

### Files touched

- `plans/PR-FAQ-Deflection-Paid-Postgres-Smoke.md` -- slice plan.
- `scripts/smoke_content_ops_deflection_paid_postgres.py` -- guarded operator
  Postgres smoke.
- `tests/test_faq_deflection_paid_postgres_smoke.py` -- preflight and fake
  Postgres coverage.

## Mechanism

The script is invoked only with explicit operator inputs:

```bash
python scripts/smoke_content_ops_deflection_paid_postgres.py \
  --database-url "$DATABASE_URL" \
  --account-id "$ACCOUNT_ID" \
  --confirm-postgres-write \
  --json
```

It creates a generated `smoke-...` request id unless the operator supplies one,
rejects non-smoke request ids, saves a small snapshot/artifact through
`PostgresDeflectionReportArtifactStore`, confirms the artifact record is locked,
then calls
`billing._handle_content_ops_deflection_report_checkout_completed(...)` with a
fake paid Checkout session carrying `metadata.source=content_ops_deflection_report`.
The final read uses the same Postgres store to prove `paid=true` and the payment
reference persisted.

## Intentional

- No real Stripe API call. This validates the Postgres paid-state flow behind
  the webhook helper, not Stripe signature construction.
- No environment reads for secrets. The database URL is an explicit argument.
- The test file is kept out of extracted-check filename patterns because this
  smoke imports host billing code and validates host Postgres wiring.
- No cleanup step in this slice; the generated `smoke-...` request id is
  returned in the JSON payload so operators can inspect or remove the row.

## Deferred

- Future robust-testing slice: hosted browser/API validation for the portfolio
  result page once it consumes the merged Checkout contract.
- Future hardening slice: optional cleanup mode for smoke rows if operators
  want automatic deletion after successful inspection.
- Parked hardening considered: none in `HARDENING.md` touch this gating slice.

## Verification

- `python -m pytest tests/test_faq_deflection_paid_postgres_smoke.py -q`
  (5 passed, 1 warning from host torch import)
- `python -m py_compile scripts/smoke_content_ops_deflection_paid_postgres.py tests/test_faq_deflection_paid_postgres_smoke.py`
- `python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Paid-Postgres-Smoke.md`
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Paid-Postgres-Smoke.md`
- `python scripts/audit_extracted_pipeline_ci_enrollment.py`
- `bash scripts/check_ascii_python.sh`
- `git diff --check`
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-deflection-paid-postgres-smoke.md`

## Estimated diff size

| Area | Estimate |
|---|---:|
| Plan | ~90 |
| Smoke script | ~335 |
| Tests | ~195 |
| Total | ~620 |

The estimate exceeds the 400 LOC soft cap because the live-write guard and fake
Postgres coverage need to ship together; otherwise the safety claim is only
documented, not enforced.
