# PR-Deflection-Delta-Go-Live-Runbook

## Why this slice exists

#1865 closed the last status-precedence blocker in the report-deltas lane. The
remaining go-live risk is not another status string; it is operator drift. The
monthly delta worker is opt-in and live delivery is behind a separate dry-run
gate, but the activation sequence currently lives in review notes instead of a
checked runbook.

Root cause: the repo has the production code path, migrations, scheduler seed,
and delivery dry-run switch, but no durable operator contract tying them into
one safe rehearsal-then-live sequence. This slice fixes that root for the
operator surface by adding a go-live runbook and a test that pins the
load-bearing gates. Review follow-up tightened that contract so the runbook
does not pass on operator/test emails and does not treat a zero-row dry-run as
delivery proof.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Production hardening

1. Add a checked go-live runbook for monthly Report Delta activation.
2. Pin the runbook's migration, scheduler, dry-run rehearsal, and live-send
   gates in a focused test.
3. Enroll that focused test in the extracted-pipeline CI runner.
4. Archive the just-merged #1865 plan doc as teardown for the previous slice.

### Review Contract

- Acceptance criteria:
  - The runbook requires a real paid account with at least two paid reports and
    delivery email before live activation.
  - The runbook requires migration `341_content_ops_deflection_delta_deliveries`
    to be applied before enabling the task, using the `schema_migrations` stem
    actually recorded by the Atlas migration runner.
  - The paid-pair query surfaces delivery addresses and requires the current
    paid report to be deliverable with an earlier paid baseline.
  - The first manual run uses `delivery_dry_run: true`; live email send is
    allowed only after that rehearsal is reviewed.
  - The dry-run proceed criteria require sender config, no missing config, at
    least one enqueued delta delivery, and at least one dry-run delivery row.
  - The production flip names both required gates:
    `ATLAS_DEFLECTION_DELTA_ENABLED=true` and
    `ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false`.
  - The runbook names that the global delivery dry-run flag is shared with paid
    report delivery and avoids a manual live run for one buyer until an
    account-scoped activation exists.
  - Tests fail if those gates disappear from the runbook.
  - The runbook test is enrolled in `scripts/run_extracted_pipeline_checks.sh`.
- Affected surfaces:
  - Operator documentation only.
  - Focused runbook test only.
- Risk areas:
  - Money/customer email path activation.
  - Operator runbook drift.
- Triggered reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R3 Security/auth/money path.
  - R6 Workflow/job behavior.
  - R8 Persistence/data lifecycle.
  - R14 Codebase verification.

### Files touched

- `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md`
- `plans/INDEX.md`
- `plans/PR-Deflection-Delta-Go-Live-Runbook.md`
- `plans/archive/PR-Deflection-Delta-Deferred-Status.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_delta_go_live_runbook.py`

## Mechanism

Add `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md`
with the operator sequence:

1. Confirm the paid report pair and delivery email prerequisite.
2. Confirm migration 341 is present in `schema_migrations`.
3. Enable only after the brain is restarted with the delta flag.
4. Run the task manually with `delivery_dry_run: true`.
5. Poll the task execution by UUID and review the generated/delivery payload.
6. Require a positive dry-run delivery count before treating the rehearsal as
   proof.
7. Flip `ATLAS_DEFLECTION_DELIVERY_DRY_RUN=false` only after the dry-run
   rehearsal passes.

Add `tests/test_content_ops_deflection_delta_go_live_runbook.py` to assert the
checked runbook still contains the migration name, task name, dry-run override,
live-send env gates, and the paid-account SQL predicate. This mirrors the
existing parser-light runbook tests used by the Content Ops validation docs.

## Intentional

- No production default changes. `settings.deflection_delta.enabled` remains
  false by default and `settings.deflection_delivery.dry_run` remains true by
  default; real activation is an environment/operator action.
- No live database or email send is executed in this PR. The runbook names the
  commands and the test pins the guardrails.

## Deferred

- Running the go-live rehearsal against production is deferred to the operator
  with the real paid customer, database, and sender credentials.
- Entitlement/billing changes for selling monthly Report Deltas as a
  subscription remain outside this activation runbook.

Parked hardening: none.

## Verification

- python -m py_compile tests/test_content_ops_deflection_delta_go_live_runbook.py
- pytest tests/test_content_ops_deflection_delta_go_live_runbook.py (5 passed)
- python scripts/audit_extracted_pipeline_ci_enrollment.py
- python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Go-Live-Runbook.md --check

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md` | 184 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Delta-Go-Live-Runbook.md` | 129 |
| `plans/archive/PR-Deflection-Delta-Deferred-Status.md` | 0 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_delta_go_live_runbook.py` | 110 |
| **Total** | **427** |
