# PR-Deflection-Delta-Entitlement-Allowlist

## Why this slice exists

#1316 and the archived delta plans repeatedly deferred subscription entitlement
checks before monthly Report Delta go-live. #1868 made manual activation safely
account/current-request scoped, but the global monthly path still uses "has at
least one paid deflection report" as the account roster.

Root cause: paid report access is a one-time purchase state, while monthly delta
delivery is a subscription entitlement. Reusing paid-report discovery as the
subscription roster can enqueue recurring emails for accounts that bought a
single report but did not opt into monthly deltas.

This PR fixes the root for the current architecture by adding an explicit,
fail-closed account allowlist at the automation/store boundary. It does not add
a Stripe subscription checkout or webhook sync; that larger Billing-backed
entitlement source remains deferred until pricing/product enrollment is ready.

Diff-size note: this is over the 400 LOC target because the smallest root-cause
slice has to update the typed config, the access-boundary protocol plus both
store implementations, the scheduled task guard, and tests at both layers. A
task-only guard would leave the batch generator reusable without the entitlement
filter and would be a symptom fix.

## Scope (this PR)

Ownership lane: deflection/report-deltas
Slice phase: Production hardening

1. Add an `ATLAS_DEFLECTION_DELTA_ENTITLED_ACCOUNT_IDS` allowlist to the typed
   delta config.
2. Thread the allowlist into recent-delta generation and pending delivery drain
   queries so global monthly runs scan and send only entitled accounts.
3. Fail closed when the automation is enabled without any entitled accounts, or
   when a manual `target_account_id` is not entitled.
4. Add tests proving global generation excludes non-entitled paid accounts and
   targeted activation rejects non-entitled accounts before delivery.
5. Update the go-live runbook so operators set the required allowlist before
   dry-run or live monthly delta activation.

### Review Contract

Acceptance criteria:
- Monthly/global delta automation must not call the batch generator when no
  entitlement allowlist is configured.
- Manual targeted automation must not generate or drain delivery for a
  `target_account_id` outside the entitlement allowlist.
- Batch generation must intersect paid-report discovery with the entitlement
  allowlist so a paid-but-unentitled account is not scanned/enqueued.
- Pending delta delivery drain/count queries must also honor the entitlement
  allowlist so already-queued rows for revoked accounts are not sent later.
- The go-live runbook must document
  `ATLAS_DEFLECTION_DELTA_ENTITLED_ACCOUNT_IDS` as required activation config.
- Existing scoped `target_account_id` + `current_request_id` behavior from
  #1868 remains intact for entitled accounts.

Affected surfaces:
- Deflection delta task configuration and automation payloads.
- Recent paid report discovery in the deflection report access boundary.
- Unit/in-memory persistence tests for batch selection.
- Monthly delta go-live runbook and its contract test.

Risk areas:
- Accidentally widening a blank entitlement list into all paid accounts.
- Breaking the manual go-live path that now requires a configured entitled
  account.
- Confusing one-time paid report state with recurring subscription state.

Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R11, R12, R14.
- R1 Requirements match.
- R2 Test evidence.
- R3 Security/auth/billing.
- R6 Jobs/scheduled automation.
- R8 Persistence/data lifecycle.
- R10 Gate/predicate behavior.
- R14 Codebase verification.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/config.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Entitlement-Allowlist.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_delta_go_live_runbook.py`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`

## Mechanism

- Add a comma-separated `entitled_account_ids` field under
  `settings.deflection_delta`.
- Parse it into normalized account IDs in
  `content_ops_deflection_delta_automation.run`.
- If the task is global and the allowlist is empty, return a skip payload before
  generation or delivery.
- If the task is targeted and the target is not in the allowlist, return a
  fail-closed skip payload before generation or delivery.
- Extend `compute_and_save_recent_deflection_deltas` with an optional entitled
  account list. Global account discovery uses only the paid accounts in that
  list, preserving the existing paid gate while separating the recurring
  entitlement roster from one-time report purchases.
- Thread the same allowlist into `send_pending_deflection_delta_deliveries` and
  `pending_deflection_delta_delivery_count`. The delivery claim/read SQL applies
  an account filter before rows are scanned, claimed, counted, or emailed.
- Require `ATLAS_DEFLECTION_DELTA_ENTITLED_ACCOUNT_IDS` in the dry-run and live
  activation snippets of the go-live runbook, and pin that requirement in the
  runbook contract test.

## Intentional

- No Stripe API call, Checkout Session, subscription webhook, or Customer Portal
  flow lands here. Per the Stripe Billing guidance, the durable product path
  should use Billing subscriptions/Prices; this slice only adds the local
  fail-closed roster needed before enabling monthly automation.
- No wildcard/all-paid escape hatch. A blank allowlist means no monthly delta
  accounts, because the failure mode we are closing is recurring delivery to
  one-time report buyers.
- No new DB table for entitlements in this PR; the source of truth is the
  typed config until the subscription product is ready.

## Deferred

- Replace the config allowlist with a Stripe Billing-backed subscription
  entitlement source once the monthly product/Price IDs and webhook lifecycle
  are defined.
- #1869: DB-backed live-drain integration coverage once the deflection delivery
  CI gate has a Postgres service.

Parked hardening: none.

## Verification

- Passed locally:
  - python -m py_compile atlas_brain/config.py atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py atlas_brain/content_ops_deflection_delivery.py extracted_content_pipeline/deflection_report_access.py tests/test_deflection_delta_automation_task.py tests/test_content_ops_deflection_delta_persistence.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_content_ops_deflection_delta_go_live_runbook.py
  - pytest tests/test_deflection_delta_automation_task.py tests/test_content_ops_deflection_delta_persistence.py
    - 53 passed, 1 warning.
  - pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_deflection_delta_automation_task.py -q
    - 59 passed, 1 warning.
  - pytest tests/test_content_ops_deflection_delta_go_live_runbook.py -q
    - 6 passed.
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Delta-Entitlement-Allowlist.md --check

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 68 |
| `atlas_brain/config.py` | 7 |
| `atlas_brain/content_ops_deflection_delivery.py` | 23 |
| `docs/extraction/validation/content_ops_deflection_delta_go_live_runbook.md` | 7 |
| `extracted_content_pipeline/deflection_report_access.py` | 75 |
| `plans/PR-Deflection-Delta-Entitlement-Allowlist.md` | 161 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 40 |
| `tests/test_content_ops_deflection_delta_go_live_runbook.py` | 13 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 116 |
| `tests/test_deflection_delta_automation_task.py` | 106 |
| **Total** | **616** |
