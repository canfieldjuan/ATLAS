# PR: Deflection delivery idempotency (ATLAS #1461)

## Why this slice exists

ATLAS #1461 is a P1 money-path bug. In `content_ops_deflection_delivery.py` the
send loop calls `await sender.send(...)` (`:110`) and only afterwards calls
`_mark_delivered` (`:132`). A crash or DB blip between those two leaves the row
`delivery_status = 'sending'`. The claim SQL re-claims `'sending'` rows older
than `DELIVERY_CLAIM_STALE_AFTER = "15 minutes"` (`:422`), and the pre-send
`_confirm_delivery_still_sendable` check passes (the row is still `'sending'` +
`paid`), so the worker **sends the report email a second time** to a paying
customer. The stored `provider_message_id` does not help: in the crash case it
was never written.

This blocks the #1440 live run: we should not invite a real paid run while a
single crash window double-emails the buyer.

## Scope (this PR)

Ownership lane: content-ops/deflection-delivery
Slice phase: Production hardening

1. Add an optional, backward-compatible `idempotency_key` field to the shared
   `SendRequest` port.
2. Have the Resend sender forward it as the Resend `Idempotency-Key` HTTP
   request header (Resend dedupes identical keys server-side for 24h).
3. Have the deflection delivery path set a **deterministic** key derived purely
   from `(account_id, request_id)`, so a re-claimed `'sending'` row recomputes
   the identical key and Resend dedupes the resend.
4. A regression test that simulates a crash between `send()` and
   `_mark_delivered`, re-claims, and asserts the resend carries the same key and
   no second email is emitted.

Out of scope: the billing 409 retry-storm (#1462, separate slice); any schema /
migration change; the SES sender.

- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `plans/PR-Deflection-Delivery-Idempotency.md`
- `extracted_content_pipeline/campaign_ports.py`
- `extracted_content_pipeline/campaign_sender.py`
- `atlas_brain/content_ops_deflection_delivery.py`
- `tests/test_atlas_content_ops_deflection_delivery.py`
- `tests/test_extracted_campaign_sender.py`

## Mechanism

The key is a pure function of identifiers already on the row:
`deflection-report:{account_id}:{request_id}`. Because the claim re-tries after
15 minutes -- far inside Resend's 24h idempotency window -- the recomputed key
on re-claim matches the original send, and Resend returns the original email
instead of sending a new one. No new column is needed: the key is recomputed,
not stored.

- `extracted_content_pipeline/campaign_ports.py` (owned): `SendRequest` gains
  `idempotency_key: str | None = None` as a trailing optional field
  (backward-compatible -- existing positional/keyword construction unaffected).
- `extracted_content_pipeline/campaign_sender.py` (owned):
  `ResendCampaignSender.send` adds `headers["Idempotency-Key"] =
  request.idempotency_key` to the HTTP request headers when the key is set. The
  SES sender ignores it (SES has no native idempotency header, and it is not on
  the deflection path).
- `atlas_brain/content_ops_deflection_delivery.py`: a `_delivery_idempotency_key`
  helper, used by `_send_request`.

## Intentional

- The key is deterministic and recomputed, not persisted -- no migration, and
  re-claim is guaranteed to reproduce it.
- Resend-only. The deflection delivery sender is `create_campaign_sender(
  "resend", ...)`, so the native Resend `Idempotency-Key` is the correct,
  simplest mechanism. The `idempotency_key` field is provider-agnostic; if a
  non-Resend sender is ever wired into deflection delivery, the same key is the
  right primitive and a provider-appropriate dedupe (e.g. a Sent-mailbox lookup
  for an API without idempotency) would be added then.
- This relies on Resend honoring the key within 24h. The 15-minute claim window
  keeps the resend well inside that window; the dependency is documented rather
  than belt-and-suspendered with a second DB-level dedupe, to keep the slice
  small and avoid a schema change.

## Deferred

- #1462 (billing 409 -> Stripe retry storm) is the next money-path slice.
- A DB-level idempotency ledger (provider-independent, survives a >24h stall)
  is a future hardening if delivery ever needs a claim window longer than the
  provider idempotency TTL.

Parked hardening: none.

## Verification

- Focused pytest passed over `tests/test_atlas_content_ops_deflection_delivery.py`
  (incl. the new crash/re-claim no-double-send test),
  `tests/test_extracted_campaign_sender.py` (the two Resend Idempotency-Key
  HTTP-header forwarding tests), and
  `tests/test_send_content_ops_deflection_report_deliveries.py`.
- ASCII gate `scripts/check_ascii_python.sh` -- passed.
- Python compile check for the three touched modules -- passed.
- Full gauntlet `scripts/run_extracted_pipeline_checks.sh` -- 3906 passed,
  10 skipped, 0 failed; existing torch/pynvml warning.
- Standalone audit `scripts/audit_extracted_standalone.py` (with --fail-on-debt)
  -- 0 import findings.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_ports.py` | 5 |
| `extracted_content_pipeline/campaign_sender.py` | 5 |
| `atlas_brain/content_ops_deflection_delivery.py` | 14 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 65 |
| `tests/test_extracted_campaign_sender.py` | 26 |
| `plans/PR-Deflection-Delivery-Idempotency.md` | 97 |
| **Total** | **212** |
