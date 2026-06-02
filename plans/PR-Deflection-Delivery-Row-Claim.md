## Why this slice exists

The paid FAQ deflection delivery loop is now operator-runnable, but the worker
still reads pending rows with a plain `SELECT`. Two operator runs, or an
operator run plus a future scheduler, can therefore observe the same pending
row and send the same paid report link twice. This production-hardening slice
closes that double-send risk before scheduler automation.

## Scope (this PR)

Ownership lane: ai-content-ops/faq-deflection-paid-unlock
Slice phase: Production hardening

1. Claim live delivery rows atomically before sending by moving them from
   `pending` to `sending` with `FOR UPDATE SKIP LOCKED`.
2. Keep dry-run read-only; dry-run still scans pending rows but does not claim,
   send, or update.
3. Preserve `sending` rows when duplicate Stripe webhooks requeue the same paid
   report, so a webhook race cannot reopen an in-flight delivery.
4. Add focused tests proving live claim SQL, dry-run read-only behavior, stale
   `sending` retry eligibility, and duplicate-webhook status preservation.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `atlas_brain/api/billing.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_atlas_billing_content_ops_deflection_stripe_paid.py`
- `plans/PR-Deflection-Delivery-Row-Claim.md`

## Mechanism

Live delivery fetches use a writable CTE:

```sql
WITH claimed AS (
  SELECT account_id, request_id
  FROM content_ops_deflection_report_deliveries
  WHERE delivery_status = 'pending'
  FOR UPDATE SKIP LOCKED
  LIMIT $1
)
UPDATE content_ops_deflection_report_deliveries ...
RETURNING ...
```

The worker only sends rows returned by that claim statement. Success moves a
claimed row from `sending` to `delivered`; validation/provider failures move it
to `failed`. Rows left in `sending` by a crashed worker become eligible after a
bounded stale-claim interval. Dry-run keeps using the existing read-only pending
query.

The Stripe delivery queue upsert preserves both `delivered` and `sending`.
Other statuses can still be reset to `pending` by a verified paid webhook.

## Intentional

- No scheduler/cron is added; the manual script remains the only operator
  entrypoint.
- No new migration is added; `delivery_status` is already `TEXT`, and the
  pending partial index still serves the normal queue path.
- Stale `sending` retry uses a conservative fixed interval rather than a new
  CLI/config field; operational tuning can wait until scheduler automation.

## Deferred

- Future slice: scheduler/cron wiring for the claimed worker.
- Future slice: provider webhook ingestion for delivered/open/click events.
- Future slice: abandoned checkout follow-up policy and unsubscribe handling.
- Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/content_ops_deflection_delivery.py atlas_brain/api/billing.py tests/test_atlas_content_ops_deflection_delivery.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py && python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py -q` -- 27 passed.
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py tests/test_send_content_ops_deflection_report_deliveries.py tests/test_atlas_billing_content_ops_deflection_stripe_paid.py tests/test_atlas_billing_content_ops_deflection_paid_flow.py -q` -- 34 passed.
- `scripts/run_extracted_pipeline_checks.sh` -- run with bash; 2955 passed, 10 skipped.
- Cross-layer caller hint for `send_pending_deflection_report_deliveries` is
  covered by `tests/test_send_content_ops_deflection_report_deliveries.py` in
  the combined focused run.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Worker claim SQL | ~45 |
| Billing queue guard | ~5 |
| Tests | ~95 |
| Plan doc | ~80 |
| **Total** | **~225** |
