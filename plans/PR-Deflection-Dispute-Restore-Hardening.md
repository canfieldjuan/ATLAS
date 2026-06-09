# PR-Deflection-Dispute-Restore-Hardening

## Why this slice exists

#1386 closed the main pay -> refund/dispute -> relock gap in prior slices, then
routed paid-funnel incidents into the alert sink. The remaining in-lane
hardening item is the restore side of that lifecycle: if a disputed payment is
closed as won, the report should become paid again and eligible for delivery
without an operator hand-edit.

## Scope (this PR)
Ownership lane: go-live-deflection-cleanup
Slice phase: Production hardening

1. Route `charge.dispute.closed` for content-ops deflection payments.
2. Restore paid access and requeue delivery only when Stripe marks the closed
   dispute `won`.
3. Leave `lost`, `warning_closed`, missing/unmapped metadata, and non-deflection
   dispute-close events observed but non-mutating.
4. Emit a bounded paid-funnel incident if a won-dispute restore maps to a
   missing report.

### Review Contract
- Acceptance criteria:
  - [ ] `charge.dispute.closed` with `status="won"` and deflection metadata
        marks the report paid again and queues delivery.
  - [ ] `charge.dispute.closed` with a non-`won` status does not restore access
        or queue delivery.
  - [ ] A won-dispute restore that maps to no report emits
        `paid_report_restore_missed_report` without leaking customer email,
        raw ticket text, evidence, or report content.
  - [ ] Existing full-refund/dispute-created revocation behavior remains
        unchanged, including full-refund-only revocation and pending/sending
        delivery cancellation.
- Affected surfaces: Stripe webhook, paid report state, delivery queue,
  observability, CI-enrolled Python tests.
- Risk areas: payment authorization, webhook idempotency, delivery retry safety,
  report access restoration.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R12.

### Files touched

- `atlas_brain/api/billing.py`
- `plans/PR-Deflection-Dispute-Restore-Hardening.md`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`

## Mechanism

The webhook router adds a `charge.dispute.closed` branch for the deflection
paid-report flow. A new handler checks the dispute status first; only `won`
continues to the same Stripe payment-event metadata mapping used by the
revocation handler. After account/request validation, the handler calls the
existing `PostgresDeflectionReportArtifactStore.mark_paid(...)` inverse and
then `_queue_content_ops_deflection_report_delivery(...)`, which already
preserves delivered/sending rows and moves revoked/failed/pending rows back to
`pending`.

Non-won statuses are logged and audited through the existing `billing_events`
insert, but do not change report state. Missing reports emit the existing
PII-safe incident envelope style under a new incident type.

## Intentional

- No new Stripe object retrieval by Charge ID. The current deflection checkout
  metadata path is Checkout Session lookup by `payment_intent` or direct
  metadata; this slice keeps that bounded lookup surface rather than adding a
  broader Charge retrieve path.
- No schema change. `mark_paid`, `mark_unpaid`, and delivery upsert semantics
  already model the restore transition.
- Only `status="won"` restores access. Stripe closed statuses such as `lost`
  and `warning_closed` do not restore access.

## Deferred

- PaymentIntent/Charge metadata copy in atlas-portfolio remains a separate
  hardening option so future dispute objects are self-describing even when
  Checkout Session lookup is unavailable.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` -- 42 passed, 1 warning.
- `python -m pytest tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` -- 1 passed, 1 warning.
- `python -m pytest tests/test_alerts.py tests/test_atlas_billing_stripe_hardening.py tests/test_b2b_vendor_briefing.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py tests/test_content_ops_deflection_incidents.py tests/test_mcp_content_ops_deflection_readonly.py -q` -- 180 passed, 1 warning.
- Python syntax compile over changed Python files and `git diff --check` -- passed.
- Push-wrapper local review -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/billing.py` | 110 |
| `plans/PR-Deflection-Dispute-Restore-Hardening.md` | 96 |
| `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py` | 193 |
| **Total** | **399** |
