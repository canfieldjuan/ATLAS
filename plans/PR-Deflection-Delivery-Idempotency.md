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

- Reviewer rules triggered: R1, R2, R6, R8, R10. (R6 error-handling/observability
  and R8 concurrency/idempotency: this is a retry/idempotency slice that changes
  how a provider conflict is handled on the delivery path.)

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

### Idempotent-replay conflict (Codex #1511 review)

Resend returns `409 invalid_idempotent_request` when the same key is reused with
a *different* payload. The paid report carries a generated PDF attachment, and
the real PDF renderer stamps a per-call creation timestamp / file id, so the
re-claim re-renders to different bytes -> the retry would be a 409, which the
delivery loop's generic `except` would mark `failed` (a false send-failure
incident) even though the original email already went out. To close that:

- `campaign_ports.IdempotentReplayConflict` -- a typed exception meaning "an
  email was already accepted for this key" (delivered, not failed).
- `ResendCampaignSender.send` detects the documented `invalid_idempotent_request`
  409 body and raises `IdempotentReplayConflict` instead of a generic HTTP error
  (other 409s still raise normally via `raise_for_status`).
- The delivery loop catches it before the generic failure branch, emits an
  `info` `paid_report_delivery_idempotent_replay` incident, and marks the row
  `delivered` (`provider_message_id = resend:idempotent-replay`).

This makes the fix robust to *any* payload drift, not only the PDF timestamp.

## Intentional

- The key is deterministic and recomputed, not persisted -- no migration, and
  re-claim is guaranteed to reproduce it.
- A re-claim that re-renders a byte-different PDF is treated as delivered via the
  `IdempotentReplayConflict` path above, not failed. Chosen over forcing a
  byte-deterministic PDF render because it is contained to the delivery
  path/sender and robust to all payload drift; the sentinel
  `resend:idempotent-replay` id costs nothing here because the deflection
  delivery's `provider_message_id` is write-only (no webhook/bounce correlation
  reads it, unlike the campaign flow).
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
  (the crash/re-claim no-double-send test + the idempotent-replay-conflict
  marks-delivered-not-failed test), `tests/test_extracted_campaign_sender.py`
  (the two Resend Idempotency-Key HTTP-header forwarding tests + the 409
  `invalid_idempotent_request` -> `IdempotentReplayConflict` test and the
  non-idempotency-409 still-raises test), and
  `tests/test_send_content_ops_deflection_report_deliveries.py`, plus an end-to-end
  regression using the real `ResendCampaignSender` + a payload-aware fake HTTP
  client + the real render->link-only fallback (same key, different body ->
  delivered, not failed).
- ASCII gate `scripts/check_ascii_python.sh` -- passed.
- Python compile check for the touched modules -- passed.
- Full gauntlet `scripts/run_extracted_pipeline_checks.sh` -- 3910 passed,
  10 skipped, 0 failed; existing torch/pynvml warning.
- Standalone audit `scripts/audit_extracted_standalone.py` (with --fail-on-debt)
  -- 0 import findings.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/campaign_ports.py` | 24 |
| `extracted_content_pipeline/campaign_sender.py` | 36 |
| `atlas_brain/content_ops_deflection_delivery.py` | 38 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 97 |
| `tests/test_extracted_campaign_sender.py` | 62 |
| `plans/PR-Deflection-Delivery-Idempotency.md` | 152 |
| **Total** | **~409** |
