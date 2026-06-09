# PR-Deflection-Refund-Dispute-Revocation

## Why this slice exists

Issue #1386 calls out the launch-gating paid-funnel gap: a buyer can pay,
unlock the full deflection report, then receive a refund or file a dispute and
keep access because the Stripe webhook path has `mark_paid` but no inverse.

Earlier #1386 slices already handled checkout authorization, async payment
success, and delivery scheduling. This slice closes the concrete revocation
path for refund/dispute webhooks while keeping the paid gate server-owned.

This intentionally exceeds the 400 LOC soft cap because the refund/dispute
event class has several safety branches that must be proven in the same slice:
direct metadata, Checkout Session lookup by `payment_intent`, lookup failure,
partial refund no-op, delivery cancellation, in-flight delivery recheck,
unmapped events, and duplicate-event idempotency. Splitting the tests from the
handler would leave a payment revocation path under-proven.

## Scope (this PR)

Ownership lane: go-live-deflection-cleanup
Slice phase: Production hardening

1. Add an account/request-scoped `mark_unpaid` storage transition for
   paid-gated deflection reports.
2. Route Stripe `charge.refunded` and `charge.dispute.created` events for
   `content_ops_deflection_report` purchases to revoke report access.
3. Resolve Charge/Dispute events through direct metadata when present, or by
   looking up the Checkout Session from the event's `payment_intent` because
   the portfolio checkout currently stores the deflection metadata on the
   Checkout Session.
4. Treat only full `charge.refunded` reversals as access revocations; partial
   refunds are observed without relocking the report.
5. Emit structured revocation/missing-mapping logs and preserve the existing
   `billing_events` audit row for idempotency and operator traceability.
6. Recheck report paid state and delivery status immediately before sending a
   queued paid-report email so an in-flight delivery fails closed after
   revocation.
7. Add focused regression tests for refund revocation, dispute revocation,
   metadata fallback through `payment_intent`, lookup failure retry,
   partial-refund no-op, missing metadata no-unlock, in-flight delivery
   suppression, and duplicate-event idempotency.

### Review Contract

- Acceptance criteria:
  - [ ] A paid deflection report becomes locked again when a matching
        `charge.refunded` event arrives.
  - [ ] A paid deflection report becomes locked again when a matching
        `charge.dispute.created` event arrives.
  - [ ] Refund/dispute events can map through Checkout Session lookup by
        `payment_intent` when the Charge/Dispute object does not carry
        deflection metadata directly.
  - [ ] Checkout lookup failures raise a 503 so Stripe retries instead of
        silently leaving a report unlocked.
  - [ ] Partial refunds do not revoke report access or cancel delivery.
  - [ ] Matching refund/dispute events cancel pending/sending delivery rows so
        a revoked report is not emailed after access is relocked.
  - [ ] The delivery worker rechecks `paid=true` and `delivery_status=sending`
        immediately before `sender.send` and suppresses an in-flight send when
        revocation wins the race.
  - [ ] Unknown or unmapped refund/dispute events do not unlock, requeue
        delivery, or mutate unrelated reports; they log operator-readable
        context and still get the existing Stripe event audit row.
  - [ ] Already-processed refund/dispute event IDs remain idempotent and do not
        run a second revocation.
- Affected surfaces: Stripe webhook routing, deflection report access storage,
  paid-funnel observability/audit logs, billing webhook tests, report delivery
  worker tests.
- Risk areas: false-positive revocation, webhook idempotency, Stripe lookup
  failures, audit logging after side effects, account/request isolation.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10.

### Files touched

- `atlas_brain/api/billing.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Refund-Dispute-Revocation.md`
- `tests/test_atlas_billing_content_ops_deflection_paid_flow.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_content_ops_deflection_report.py`

## Mechanism

`DeflectionReportArtifactStore` gains a `mark_unpaid` inverse to the
existing `mark_paid()` transition. The Postgres implementation updates only the
matching `(account_id, request_id)` row, sets `paid=false`, clears `paid_at`,
and leaves the artifact/snapshot in place so the existing artifact route locks
the report again without data deletion.

The Stripe webhook router handles `charge.refunded` and
`charge.dispute.created`. For deflection events it extracts
`source/account_id/request_id` from object metadata when present. When that
metadata is absent, it reads the event's `payment_intent`, asks Stripe for the
related Checkout Session, and uses the original Checkout Session metadata and
session ID as the payment reference. That matches the current portfolio
checkout implementation, which sets `metadata[...]` on the Checkout Session
only.

The revocation helper rejects malformed or unmapped events without mutating
reports, logs stable operator-readable context, and then lets the existing
webhook audit insert record the Stripe event. Checkout lookup failures raise a
503 so Stripe retries instead of recording a false-green no-op. Duplicate event
IDs still return `already_processed` before side effects through the existing
billing-events idempotency gate.

For delivery, the worker still claims rows as `sending`, but it performs a
second guarded database transition immediately before `sender.send`. That
transition only returns a row while the delivery is still `sending` and the
joined report is still `paid=true`, so a refund/dispute that lands after claim
but before send suppresses the email.

## Intentional

- This slice does not add new report-state columns such as `revoked_at` or
  `revocation_reason`; relocking the existing `paid` flag is the narrow launch
  fix and preserves the current artifact contract.
- Partial refunds are logged and audited but do not relock the report. A
  refund revokes access only when Stripe marks the Charge fully refunded or the
  refunded amount covers the captured amount.
- The handler uses a Stripe Checkout Session lookup as a fallback instead of
  changing the portfolio checkout metadata shape in this ATLAS PR. A portfolio
  follow-up can add `payment_intent_data[metadata]` to make Charge events
  self-describing, but ATLAS still needs the fallback for already-created
  sessions.
- The observability in this slice is the webhook audit row plus structured
  warning/error logs for revocation, missing metadata, lookup failures, and
  revoke misses. A production alert sink remains separate because no dedicated
  paid-funnel alert sink exists in this repo yet.
- The paid-flow smoke assertion is refreshed for the already-landed
  resolution-evidence summary fields so this slice can keep using that adjacent
  paid-gate smoke as verification.

## Deferred

- #1386 follow-up: route paid-funnel incident events into the production alert
  sink once the sink contract is selected.
- #1386 follow-up: add a dispute-closed/manual-restore path. This slice makes
  revocation one-way for disputes so disputed access fails closed until an
  operator restores it.
- Portfolio follow-up: copy deflection metadata onto `payment_intent_data` so
  refund/dispute Charge events are self-describing without a Checkout Session
  lookup.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q`
  - Result: `36 passed, 1 warning in 2.37s`.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  - Result: `12 passed in 0.19s`.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q`
  - Result: `1 passed, 1 warning in 2.57s`.
- `python -m pytest tests/test_content_ops_deflection_report.py -q`
  - Result: `37 passed in 0.16s`.
- Python compile check for `atlas_brain/api/billing.py`,
  `atlas_brain/content_ops_deflection_delivery.py`, and
  `extracted_content_pipeline/deflection_report_access.py`
  - Result: passed.
- `git diff --check`
  - Result: passed.
- `scripts/validate_extracted_content_pipeline.sh` via bash
  - Result: passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - Result: passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt`
  - Result: passed, 0 findings.
- `scripts/check_ascii_python.sh` via bash
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 207 |
| `atlas_brain/content_ops_deflection_delivery.py` | 33 |
| `extracted_content_pipeline/deflection_report_access.py` | 67 |
| `plans/PR-Deflection-Refund-Dispute-Revocation.md` | 187 |
| `tests/test_atlas_billing_content_ops_deflection_paid_flow.py` | 2 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 419 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 39 |
| `tests/test_content_ops_deflection_report.py` | 35 |
| **Total** | **989** |
