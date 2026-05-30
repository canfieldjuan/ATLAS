# PR-FAQ-Deflection-Stripe-Paid-Replay-Smoke

## Why this slice exists

PR-FAQ-Deflection-Stripe-Paid-Retry-Idempotency locked the Stripe paid-unlock
retry contract in the host billing tests. The hosted smoke still proves only
the first signed webhook unlocks the artifact. Operators need an optional
deployed-host replay probe that posts the same Stripe event again and confirms
the webhook reports the already-processed state without relocking the paid
artifact.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-gating
Slice phase: Robust testing

1. Add an opt-in `--replay-webhook` mode to the hosted deflection Stripe
   paid-unlock smoke.
2. Require the replay response to be HTTP 200 with
   `{"status": "already_processed"}`.
3. Keep the post-webhook artifact check in place after replay so the smoke
   still proves the paid report remains accessible.
4. Document the replay flag in the deflection submit handoff runbook.

### Files touched

- `plans/PR-FAQ-Deflection-Stripe-Paid-Replay-Smoke.md`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`
- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`

## Mechanism

The smoke already builds one deterministic Stripe event body and signature.
When `--replay-webhook` is present, it posts that same signed event a second
time after the first webhook returns `{"status": "ok"}` and before fetching the
paid artifact. The result JSON gains a `replay_webhook` status field so operator
artifacts show whether the duplicate-event guard was exercised.

## Intentional

- Replay stays opt-in because the normal smoke is a paid-unlock proof; operators
  may not want every run to post duplicate webhook traffic.
- The replay expectation is strict: `already_processed`, not either `ok` or
  `already_processed`. If the audit row is absent after the first successful
  webhook, the deployed idempotency guard did not prove the contract this smoke
  is meant to validate.
- The smoke remains stdlib-only and keeps using the configured hosted URL and
  signed webhook secret. No Stripe CLI dependency is introduced.

## Deferred

- Parked hardening: none.
- Full browser Checkout replay is still outside this slice; this probes the
  deployed ATLAS webhook idempotency boundary with a locally signed
  Stripe-compatible event.

## Verification

- Command: python -m py_compile scripts/smoke_content_ops_deflection_stripe_paid_unlock.py tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py
  - Result: passed.
- Command: python -m pytest tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py -q
  - Result: 9 passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - Result: extracted_reasoning_core 295 passed; extracted_content_pipeline
    2855 passed, 10 skipped, 1 warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 76 |
| Smoke script | 40 |
| Tests | 67 |
| Runbook | 6 |
| **Total** | **189** |
