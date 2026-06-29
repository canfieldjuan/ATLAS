# PR-Deflection-Launch-Preflight-Runbook

## Why this slice exists

#1921 tracks the final launch preflight for the Resolution Audit funnel. The
code paths for Snapshot email/PDF and paid report email/PDF exist, but the
launch proof was still ambiguous: "email sent" could degrade into link-only
delivery, paid delivery is opt-in/dry-run by default, and #1440 still needed a
repeatable operator sequence that proves the emailed URLs and attachments.

Root cause: the repo had the individual surfaces and tests, but no checked
operator contract tying deployed config, Snapshot delivery, Stripe unlock,
paid delivery, PDF shape, hosted URL, cleanup, and tracker closeout into one
launch gate. This slice fixes the operator-contract root by adding a checked
runbook plus parser-light tests that pin the load-bearing gates.

The diff is slightly over the 400 LOC target because the runbook needs to be a
complete operator sequence, and the companion tests intentionally pin each
load-bearing launch gate so the doc cannot drift back to a vague checklist.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Add the launch preflight runbook for Snapshot email/PDF and paid report
   email/PDF proof.
2. Add a focused runbook test so the attachment, scheduler/config, curated PDF,
   hosted URL, cleanup, and tracker-closeout gates cannot silently disappear.
3. Enroll the focused test in the extracted-pipeline CI runner.

### Review Contract

- Acceptance criteria:
  - The runbook treats Snapshot email/PDF and paid report email/PDF as required
    launch surfaces.
  - The runbook fails launch proof if Snapshot fetch is skipped or either email
    is link-only without its PDF attachment.
  - The runbook names the portfolio intake path for Snapshot delivery and the
    ATLAS paid delivery drain for paid report delivery.
  - The runbook pins ATLAS delivery config, pre-rehearsal scheduler safety,
    paid delivery and checkout authorization migrations, real Stripe Checkout
    unlock, target queue isolation, queue-only dry-run, local paid-PDF render
    validation, live send, deployed scheduler enablement plus live
    `dry_run_enabled=false` execution proof, curated PDF/TOC shape, hosted URL
    checks, local proof-file cleanup, sanitized proof scorecard redaction, and
    #1921/#1440/#1386 closeout.
  - Tests fail if those gates disappear.
  - The new test is enrolled in `scripts/run_extracted_pipeline_checks.sh`.
- Affected surfaces:
  - Operator documentation only.
  - Focused parser-light test only.
- Risk areas:
  - Money/customer email path activation.
  - Launch proof passing on incomplete delivery.
  - Operator runbook drift.
- Triggered reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R3 Security/auth/money path.
  - R6 Workflow/job behavior.
  - R8 Persistence/data lifecycle.
  - R14 Codebase verification.

### Files touched

- `docs/extraction/validation/content_ops_deflection_launch_preflight_runbook.md`
- `plans/PR-Deflection-Launch-Preflight-Runbook.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_content_ops_deflection_launch_preflight_runbook.py`

## Mechanism

Add `docs/extraction/validation/content_ops_deflection_launch_preflight_runbook.md`
with the operator sequence:

1. Verify deployed portfolio/ATLAS email and delivery config.
2. Verify paid delivery/checkout authorization migrations and scheduler state.
3. Submit through deployed portfolio intake and require Snapshot email with PDF
   attachment.
4. Complete checkout/webhook unlock and require a pending paid delivery row.
5. Keep the scheduler from auto-sending before rehearsal, prove the target row
   is the only claimable delivery row, and run the current unscoped drain only
   after that isolation check passes.
6. Treat the dry run as queue-only, render the target artifact locally with the
   real paid PDF renderer, then send live only after queue isolation and render
   validation pass.
7. Verify the deployed scheduler has been restarted with live paid delivery
   config before launch, prove the scheduler loop reports `running`, prove
   there are zero claimable rows, then run the real autonomous task and inspect
   its repr-encoded execution result for `dry_run_enabled=false`; the manual
   drain proves one row, not public automation.
8. Verify the paid email attachment is the curated PDF with TOC/caps and that
   complete evidence remains on the hosted/export surface.
9. Open the exact emailed URLs and close out cleanup/legal/support plus
   #1921/#1440/#1386 using only a sanitized proof scorecard that passes the
   full-report proof-bundle redaction gate.

Add `tests/test_content_ops_deflection_launch_preflight_runbook.py` to assert
the checked runbook still contains the required surfaces, env gates,
migrations, checkout authorization gate, skip-log blocker, delivery script,
live-send proof, deployed scheduler enablement, scheduler-loop running proof,
actual `dry_run_enabled=false` execution proof, builtin-result repr decoding,
PDF shape, URL checks, legacy-route rejection, real-checkout wording, queue
isolation, queue-only dry-run wording, local render validation, proof-file
cleanup, sanitized scorecard redaction, and tracker closeout.
Enroll that test in the extracted-pipeline runner next to the existing delta
go-live runbook test.

## Intentional

- No production defaults change. Paid report delivery remains opt-in and dry-run
  by default until an operator performs the runbook.
- No live database, Stripe, portfolio, or Resend call is executed in this PR.
  This slice creates the checked operator contract; the live proof remains a
  separate #1921 execution artifact.
- No launch copy changes here. Landing-page copy waits until the delivery proof
  gates settle.

## Deferred

- PR 2 from #1921: make the Snapshot delivery proof fail/flag when the Snapshot
  PDF attachment is skipped.
- PR 3 from #1921: make the paid delivery proof fail/flag when the paid PDF
  attachment is missing.
- PR 4 from #1921: run the full deployed proof and update #1440/#1386/#1921
  with artifacts.

Parked hardening: none.

## Verification

- python -m py_compile tests/test_content_ops_deflection_launch_preflight_runbook.py -- passed
- pytest tests/test_content_ops_deflection_launch_preflight_runbook.py -q -- 7 passed
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 195 matching tests are enrolled
- python scripts/sync_pr_plan.py plans/PR-Deflection-Launch-Preflight-Runbook.md --check -- passed

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_deflection_launch_preflight_runbook.md` | 423 |
| `plans/PR-Deflection-Launch-Preflight-Runbook.md` | 146 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_content_ops_deflection_launch_preflight_runbook.py` | 211 |
| **Total** | **781** |
