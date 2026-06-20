# PR-Deflection-Paid-Unlock-Metadata-Guard

## Why this slice exists

#1612 is now proven end to end via Stripe test mode: a no-live-charge Checkout
completed, the hosted ATLAS webhook unlocked the report, the buyer page rendered
the full report, and the buyer report email arrived. The manual proof exposed a
tooling defect first: the paid-unlock smoke trusted caller-supplied
`account_id` metadata even though the persisted report row already had the
authoritative account. That allowed a valid test payment to be signed for the
wrong tenant and fail later as `paid_report_missing_after_payment`.

Root cause: the proof runner accepts account/report metadata from the operator
instead of verifying it against the stored `content_ops_deflection_reports`
record before posting a signed Stripe-shaped webhook.

This change fixes the root for the existing signed-webhook proof path by adding
an opt-in persisted-row guard that derives or verifies the account id from the
report store before the webhook is sent. It does not build the future hosted
Stripe test Checkout helper; that remains the next #1440 proof tooling slice.

Diff-size note: this is over the 400 LOC soft cap because guard-shaped billing
proof code needs both sides of the boundary covered: derived success,
explicit-mismatch rejection, missing-row rejection, env-default handling, and
existing unguarded compatibility.

## Scope (this PR)

Ownership lane: content-ops/report-delivery-live-funnel
Slice phase: Production hardening

1. Extend the existing paid-unlock smoke with an opt-in persisted-report
   metadata guard.
2. When the guard is enabled, require an explicit Postgres DSN, look up the
   report by request id, and either derive the account id or fail if the
   supplied account id disagrees.
3. Keep the existing no-DB signed-webhook mode for deterministic CI tests and
   older runbooks.
4. Add focused regression tests for derived metadata, mismatched metadata,
   missing report row, and sanitized result output.

### Review Contract

- Acceptance criteria:
  - [ ] Guarded mode posts webhook metadata using the persisted report row's
        account id when the caller omits account id.
  - [ ] Guarded mode fails before posting a webhook when caller account id
        differs from the persisted report row.
  - [ ] Guarded mode fails clearly when the requested report row is missing.
  - [ ] Existing unguarded signed-webhook smoke behavior remains compatible.
  - [ ] Result artifacts do not expose token or webhook secret values.
- Affected surfaces: operator smoke script, smoke tests, #1612/#1440 proof
  tooling.
- Risk areas: billing proof correctness, tenant isolation, webhook
  idempotency, CI enrollment.
- Reviewer rules triggered: R2, R3, R6, R8, R10, R12, R14.

### Files touched

- `plans/PR-Deflection-Paid-Unlock-Metadata-Guard.md`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`

## Mechanism

Add a guarded metadata resolution step before the script performs its locked
artifact check and webhook post. In guarded mode the script reads the report
row from `content_ops_deflection_reports` by request id, expects exactly one
row, and uses that row's account id as the Stripe metadata account. If the
operator also supplied an account id, it must match the persisted row.

The lookup is isolated behind a small injectable helper so tests can exercise
the behavior without a real Postgres connection. The helper uses an explicit
DSN supplied to the smoke command; it does not introduce a new Atlas config
field or depend on the app's already-initialized DB pool.

The result payload records whether persisted metadata was used, whether the
caller supplied an account id, and whether the account id was derived, but it
continues to avoid secrets and does not add raw webhook bodies to output.

## Intentional

- No hosted Stripe Checkout creation in this slice. The existing script signs a
  Stripe-shaped webhook for deterministic paid-gate proof; the next slice can
  add a separate operator-run Checkout helper for live Stripe test mode.
- No required DB lookup in the default path. CI and older runbooks can keep
  using the current explicit account id flow.
- The Postgres lookup is by request id and fails if zero or multiple rows are
  returned. Guessing on ambiguity would recreate the tenant-metadata problem.

## Deferred

- #1440 follow-up: add an operator-run Stripe test-mode Checkout helper that
  creates the hosted Checkout Session from the persisted report row, waits for
  webhook unlock, and emits sanitized proof.
- #1440 follow-up: run the full-volume funnel using the hardened paid-unlock
  proof path.

Parked hardening: none.

## Verification

- Compile check for the paid-unlock smoke script and focused test file -
  passed.
- Focused paid-unlock smoke tests - 16 passed.
- Extracted pipeline checks - 4665 passed, 10 skipped, 1 warning.
- Pending before push: local PR review via the push wrapper.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Paid-Unlock-Metadata-Guard.md` | 116 |
| `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py` | 153 |
| `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py` | 217 |
| **Total** | **486** |
