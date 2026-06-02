## Why this slice exists

PR-Deflection-Delivery-Email-Metadata persisted `delivery_email` as store-only
metadata for future post-purchase report delivery. The reviewer verified by
trace that the address does not leak through customer payloads, but flagged a
non-blocking privacy hardening gap: there is no direct regression test proving
future projections cannot accidentally expose the buyer email.

This slice locks that privacy boundary before adding any delivery worker.

## Scope (this PR)

Ownership lane: content-ops/faq-deflection-backend

Slice phase: Production hardening

1. Add negative regression coverage proving `delivery_email` is absent from the
   free snapshot response.
2. Add negative regression coverage proving `delivery_email` is absent from the
   paid artifact response after unlock.
3. Add negative regression coverage proving the gated execute/result payload
   does not expose `delivery_email`.
4. Leave storage, checkout, paid unlock, and email sending unchanged.

### Files touched

- `tests/test_extracted_content_deflection_submit.py`
- `plans/PR-Deflection-Delivery-Privacy-Regressions.md`

## Mechanism

The existing submit-route regression already creates a report with
`contact_email`, verifies the locked gated payload, and checks the stored
metadata. This slice extends that path to assert the same email is absent from:

- the immediate gated submit payload,
- `GET /deflection-reports/{request_id}/snapshot`, and
- `GET /deflection-reports/{request_id}/artifact` after the trusted paid route
  unlocks the report.

The test uses the existing in-memory store and route helpers, so it exercises
the same projection boundaries that customer routes use without adding new
infrastructure.

## Intentional

- This is test-only hardening. The prior slice already added the storage
  behavior; this slice locks the privacy invariant the reviewer requested.
- No email is sent and no delivery worker/template is introduced.
- No changes to Stripe metadata, checkout, paid unlock, snapshot shape, or
  artifact shape.

## Deferred

- Future slice: post-webhook report delivery email using the persisted
  `delivery_email` and canonical result URL.
- Future slice: abandoned/checkout-cancel follow-up capture policy and opt-out
  rules before any non-buyer nurture email is sent.
- Parked hardening: none.

## Verification

- Py compile for `tests/test_extracted_content_deflection_submit.py` - passed.
- Focused pytest for `tests/test_extracted_content_deflection_submit.py` - 18
  passed in 0.47s.
- Local PR review bundle with the current PR body - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Tests | ~20 |
| Plan doc | ~70 |
| **Total** | **~90** |

Under the 400 LOC soft cap.
