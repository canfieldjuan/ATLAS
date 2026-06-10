# PR-Deflection-Delivery-Reconciliation

## Why this slice exists

#1386 deferred the paid deflection report delivery URL/scheduler reconciliation
after #1397 wired Stripe paid events to enqueue delivery rows. #1400 then made
the launch-readiness proof explicit: the operator needs a queued paid-report
delivery to send automatically to canfieldjuan24@gmail.com and link to the live
unlocked result route, with any stubbed or missing path called out rather than
papered over with a one-off send.

The existing delivery worker can send pending rows and already builds the live
portfolio result route, but it is only reachable through a manual script. This
slice promotes that worker into the autonomous scheduler lane and locks the URL
contract with tests so the go-live path is automatic once the task is enabled.

Diff budget note: this lands above the 400 LOC target because the slice needs a
new plan doc, a new autonomous task wrapper, focused negative/positive tests for
fail-closed config, scheduler registration, live sender wiring, result URL
drift, and the reviewer-requested existing-row opt-in regression. Splitting
those tests away would leave the go-live wiring unprotected.

## Scope (this PR)

Ownership lane: deflection/go-live
Slice phase: Production hardening

1. Add typed deflection delivery scheduler settings and a disabled-by-default
   builtin interval task that drains queued paid-report delivery rows.
2. Add an autonomous task handler that uses the existing delivery worker,
   campaign sender, DB pool, and live result-route builder; no hand-fired
   delivery path is introduced.
3. Lock the default result URL contract to
   `/systems/support-ticket-deflection/results/{request_id}?checkout=success`
   and keep explicit template override support for deployment-specific URLs.
4. Add focused tests for scheduler seeding, task registration/config gating,
   CLI URL wiring, and the delivery URL contract.

### Review Contract

Acceptance criteria:
- `content_ops_deflection_report_delivery` is registered as a builtin
  autonomous task, seeded as a disabled interval task, and gets its interval
  from typed `ATLAS_DEFLECTION_DELIVERY_*` settings.
- The autonomous handler returns a clear disabled/config-missing summary when
  not enabled or not fully configured, without sending emails.
- When enabled and configured, the handler calls
  `send_pending_deflection_report_deliveries` with the real DB pool,
  campaign sender, configured limit, dry-run flag, and result URL settings.
- Default delivery links point at the live portfolio result route rather than
  the old FAQ service path.
- Tests prove the above; #1400's live inbox send remains a post-deploy
  verification step unless live credentials and DNS state are available in this
  local run.

Affected surfaces:
- Autonomous scheduler default task seeding and builtin handler registration.
- Deflection delivery config and result URL construction.
- CLI smoke coverage for the manual delivery script remains intact.

Risk areas:
- Sending emails while config is incomplete or the task is still disabled.
- Seeding an enabled production job accidentally.
- Delivering stale or incorrect result URLs.
- Drift between the script worker and scheduler worker.

Reviewer rules triggered: R1, R2, R6, R8, R11, R12.

### Files touched

- `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml`
- `atlas_brain/autonomous/scheduler.py`
- `atlas_brain/autonomous/tasks/__init__.py`
- `atlas_brain/autonomous/tasks/content_ops_deflection_report_delivery.py`
- `atlas_brain/config.py`
- `plans/PR-Deflection-Delivery-Reconciliation.md`
- `scripts/send_content_ops_deflection_report_deliveries.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_deflection_report_delivery_task.py`
- `tests/test_scheduler.py`
- `tests/test_send_content_ops_deflection_report_deliveries.py`

## Mechanism

Add a `DeflectionDeliveryConfig` settings model under the existing nested
Atlas settings tree with `ATLAS_DEFLECTION_DELIVERY_` env bindings. The model
owns the scheduler opt-in, interval, batch limit, dry-run mode, sender fields,
and portfolio result URL settings.

Seed `content_ops_deflection_report_delivery` as a disabled interval builtin in
the scheduler, resolve its interval from settings, and register a new builtin
task module. The module validates the typed settings, builds the campaign
sender through the existing campaign sender service, gets the DB pool through
the normal storage helper, and delegates all queue claiming / marking /
rendering to `send_pending_deflection_report_deliveries`.

The delivery URL remains centralized in
`atlas_brain.content_ops_deflection_delivery.deflection_report_result_url`.
Tests assert the default route and template override so future route drift is
caught before a live send.

## Intentional

- The scheduler task remains disabled by default. Enabling live sends is an
  operator/deployment action because #1400 requires real inbox, route, and
  deliverability proof.
- This PR does not seed a synthetic paid report or send a live email from local
  tests. Local verification proves the automatic queue path and URL contract;
  #1400 tracks the hands-on deployed proof.
- The manual script stays available for dry-run/operations, but the slice does
  not introduce a one-off send path that could be mistaken for automatic
  fulfillment.

## Deferred

- #1400 live proof: after deployment has `ATLAS_DEFLECTION_DELIVERY_ENABLED`,
  sender credentials, portfolio URL, and email DNS configured, trigger a real
  paid report or seeded paid test report and confirm automatic delivery to
  canfieldjuan24@gmail.com plus the unlocked result link.
- Paid-funnel incident observability into the production alert sink remains a
  later #1386 go-live slice.
- Variant-aware authorization for partner checkout re-enable remains a later
  #1386 go-live slice.

Parked hardening: none.

## Verification

- `pytest tests/test_deflection_report_delivery_task.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_scheduler.py::TestDefaults::test_deflection_delivery_default_seed_uses_typed_interval_and_opt_in -q`
  - 24 passed, 1 warning.
- `pytest tests/test_scheduler.py::TestDefaults::test_deflection_delivery_default_seed_uses_typed_interval_and_opt_in tests/test_scheduler.py::TestDefaults::test_config_managed_enabled_opt_in_syncs_existing_seeded_task tests/test_scheduler.py::TestDefaults::test_config_managed_enabled_sync_preserves_manual_disable_after_prior_sync -q`
  - 3 passed, 1 warning.
- `pytest tests/test_deflection_report_delivery_task.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py -q`
  - 23 passed, 1 warning.
- Broader exploratory run:
  `pytest tests/test_deflection_report_delivery_task.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_scheduler.py -q`
  - 59 passed, 2 failed in pre-existing `TestAutonomousApiRunNow`
    retry/timeout cases unrelated to this slice.
- `pytest tests/test_extracted_pipeline_route_ci_contract.py tests/test_audit_extracted_pipeline_ci_enrollment.py -q`
  - 21 passed.
- `bash scripts/local_pr_review.sh --allow-dirty`
  - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_content_ops_deflection_delivery_checks.yml` | 11 |
| `atlas_brain/autonomous/scheduler.py` | 53 |
| `atlas_brain/autonomous/tasks/__init__.py` | 1 |
| `atlas_brain/autonomous/tasks/content_ops_deflection_report_delivery.py` | 130 |
| `atlas_brain/config.py` | 22 |
| `plans/PR-Deflection-Delivery-Reconciliation.md` | 159 |
| `scripts/send_content_ops_deflection_report_deliveries.py` | 12 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 14 |
| `tests/test_deflection_report_delivery_task.py` | 218 |
| `tests/test_scheduler.py` | 167 |
| `tests/test_send_content_ops_deflection_report_deliveries.py` | 18 |
| **Total** | **805** |
