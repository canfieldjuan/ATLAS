# PR: Deflection billing reconciliation (ATLAS #1462)

## Why this slice exists

ATLAS #1462 is a P1 money-path bug. In `billing.py`, the deflection checkout
webhook handler raises `HTTPException(409)` whenever `mark_paid` finds no report
row. Stripe retries non-2xx webhook responses for hours, so:

- For the **permanent** case (the report row never existed), the buyer paid but
  is never marked/delivered; after retries exhaust, the payment is lost into the
  void with only a transient log line.
- Only the **transient write-ordering race** (the report row write lagging the
  webhook) actually benefits from the retry.

This blocks the #1440 live run alongside #1461: a real paid run should not be
able to land a paid-but-undelivered report that silently disappears.

## Scope (this PR)

Ownership lane: content-ops/deflection-billing
Slice phase: Production hardening

1. Discriminate the permanent paid-but-missing case from the transient race by
   **event age** (`event.created` vs now, grace window from config).
2. Permanent (event aged past the window): return **2xx** (stop the retry
   storm) + record a durable reconciliation row + emit the existing
   `paid_report_missing_after_payment` alert.
3. Race (recent event, or unknown event age): keep the **409** so Stripe retries
   and finds the report row once its write commits.
4. Add a `content_ops_deflection_paid_reconciliation` table + a small atlas_brain
   write helper + failure-first tests for both branches.

Out of scope: #1461 (delivery idempotency, separate merged slice); any
buyer-facing surface change; the pre-existing migration-prefix-collision test
failure (281/282/283/298) which is unrelated to this slice.

- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12. (R3/R8
  billing + idempotency money path; R4 new migration/table; R5 webhook handler
  return-contract change; R6 error-handling/observability; R11/R12 new config
  field.)

### Files touched

- `atlas_brain/api/billing.py`
- `atlas_brain/config.py`
- `atlas_brain/content_ops_deflection_reconciliation.py`
- `atlas_brain/storage/migrations/336_content_ops_deflection_paid_reconciliation.sql`
- `plans/PR-Deflection-Billing-Reconciliation.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

`_handle_content_ops_deflection_report_checkout_completed` gains an
`event_created` parameter (passed from both webhook dispatch sites as
`event.created`). `_event_age_seconds(event_created)` returns the seconds since
the Stripe event was created, or `None` if unknown.

In the `if not marked:` branch:

- **Aged past grace** (`age is not None and age > grace`,
  `grace = settings.saas_auth.stripe_content_ops_deflection_report_reconcile_grace_seconds`,
  default 300): the report row will not appear on retry, so this is permanent.
  `record_paid_report_missing(...)` inserts a row into
  `content_ops_deflection_paid_reconciliation` (idempotent via
  `ON CONFLICT (account_id, request_id, stripe_session_id) DO NOTHING`), the
  alert fires with `disposition="reconciled"`, and the handler **returns** (2xx).
- **Within grace, or unknown age**: the existing alert fires and the handler
  raises `409` (unchanged), so Stripe retries the genuine race. Real Stripe
  events always carry `created`, so a genuinely permanent miss ages past the
  window; only the transient race (or a malformed/timestampless event, e.g. a
  hand-built test session) lands in the 409 branch.

The reconciliation table is atlas_brain-only (migration 336), matching the
existing deflection report/delivery tables (migrations 328/332); the write
helper lives in atlas_brain (`content_ops_deflection_reconciliation.py`) rather
than the extracted store, since the billing webhook is host-only.

## Intentional

- `account_id` is `TEXT` to match `content_ops_deflection_reports` (328), not a
  uuid column.
- Unknown event age conservatively keeps the 409 (retry) rather than recording
  a reconciliation row. This preserves the prior behavior for timestampless
  sessions and never converts a possible race into a permanent record; the
  permanent path requires a real, aged event timestamp.
- The grace config is validated `ge=1` (fail-closed): a non-positive value would
  make a fresh event (`age == 0`) satisfy `age > grace` and record a genuine
  write-ordering race as permanent (2xx), bypassing the retry. With `ge=1` such
  a value cannot load (review R11/R8 + Codex). A regression asserts -1/0 raise.
- The reconciliation row is durable and queryable (operator-actionable), per the
  operator decision over an alert/log-only record.
- The race-branch alert is byte-identical to the prior behavior, so existing
  incident-payload assertions are unaffected; the permanent branch adds
  `disposition`/`event_age_seconds` to its alert.

## Deferred

- A sweeper/operator surface that lists and clears
  `content_ops_deflection_paid_reconciliation` rows (this slice only writes the
  ledger; reading/clearing is a follow-up).
- The earlier unrelated CI-red paid-flow and migration-prefix expectation drift
  was handled in #1516; this branch is rebased on that fix so the
  `atlas-content-ops-deflection-stripe-paid-checks` lane can validate this slice.

Parked hardening: none.

## Verification

- Focused pytest over `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
  -- passed, 46 tests (incl. the aged->reconcile-2xx test, the recent->409-race
  test, and the config-rejects-non-positive-grace test; the existing
  missing-report tests still 409 because they pass no `event_created`).
- Focused pytest over the three deflection-billing test files
  `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`,
  `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`, and
  `tests/test_content_ops_deflection_incidents.py` -- passed, 59 tests.
- Full `atlas-content-ops-deflection-stripe-paid-checks` pytest list -- passed,
  184 tests after rebasing on #1516.
- `python -c "from atlas_brain.config import settings; ..."` -- the new
  `stripe_content_ops_deflection_report_reconcile_grace_seconds` field loads
  (default 300).
- Migration 336 is the next free atlas_brain prefix and is collision-free
  (`_find_duplicate_migration_prefixes` does not list 336). The historical
  migration-prefix allowlist drift was handled in #1516; 336 remains outside the
  historical collision set.
- Non-ASCII scan of the touched atlas_brain files + migration -- clean.
- `python -m py_compile` for the touched modules -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 67 |
| `atlas_brain/config.py` | 12 |
| `atlas_brain/content_ops_deflection_reconciliation.py` | 48 |
| `atlas_brain/storage/migrations/336_content_ops_deflection_paid_reconciliation.sql` | 22 |
| `plans/PR-Deflection-Billing-Reconciliation.md` | 139 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 88 |
| **Total** | **376** |
