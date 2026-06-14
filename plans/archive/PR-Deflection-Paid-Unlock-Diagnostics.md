# PR-Deflection-Paid-Unlock-Diagnostics

## Why this slice exists

#1557 proved the hosted full-volume ATLAS intake and snapshot path for #1440,
but the paid-unlock proof stopped at a Stripe webhook `400`. A manual probe
showed the deployed endpoint returned `Invalid signature`, which means the
hosted webhook secret and the operator-provided signing secret are not aligned.

The root cause is not weak webhook handling: ATLAS must keep rejecting bad
Stripe signatures. The correct upstream fix for this slice is to make the paid
unlock smoke artifact preserve safe server failure details, so future live runs
identify configuration drift without a second manual probe and without exposing
tokens, webhook secrets, signed payloads, paid report Markdown, PDFs, or email
bodies.

## Scope (this PR)

Ownership lane: deflection-full-50k-e2e-proof
Slice phase: Functional validation

1. Add redacted HTTP error details to the paid-unlock smoke result artifact
   when hosted webhook or artifact checks fail.
2. Keep successful smoke artifacts backward-compatible: status fields remain
   unchanged, and no secret-bearing request values are emitted.
3. Document the `Invalid signature` interpretation in the #1440 handoff
   runbook as a deployed Stripe signing-secret alignment blocker, not as a
   reason to bypass verification.

### Review Contract

- Acceptance criteria:
  - [ ] A hosted webhook failure with JSON `detail` is recorded in the smoke
        result artifact as a short safe diagnostic.
  - [ ] Reflected secret-like values in error details are redacted before being
        written to disk or printed in JSON mode.
  - [ ] Existing happy-path and replay paid-unlock smoke behavior stays
        unchanged.
  - [ ] The runbook names `Invalid signature` as a deployment/config alignment
        issue and keeps the signed-webhook trust boundary intact.
- Affected surfaces: scripts, validation docs, payment smoke artifacts.
- Risk areas: security, observability, backcompat, CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R6, R8, R10, R12, R14.

### Files touched

- `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md`
- `plans/PR-Deflection-Paid-Unlock-Diagnostics.md`
- `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py`
- `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py`

## Mechanism

The smoke already reads HTTP error bodies as JSON when a hosted endpoint returns
an error. This slice adds a small result-summary helper that includes the HTTP
status and, only when present, a redacted scalar `detail` value from the JSON
error envelope. The expected pre-webhook artifact lock remains status-only, so
successful smoke artifacts do not gain an `error_detail` field for the normal
`payment_required` response.

Redaction happens before the artifact is assembled. It masks common Atlas and
Stripe secret prefixes and bearer-token shapes, then truncates the diagnostic
to a short bounded string. The smoke still fails when the webhook status is not
`200`; it only records why the server rejected the call.

## Intentional

- No change to `atlas_brain/api/billing.py`: accepting the mismatched local
  signing secret would weaken Stripe webhook verification. The live blocker is
  deployment/config alignment, not a webhook-code bug.
- No raw response body capture: the server currently returns safe JSON detail,
  but recording arbitrary response text would create unnecessary leakage risk.
- No new CI enrollment is needed because the existing smoke test file is
  already listed in `scripts/run_extracted_pipeline_checks.sh` and the
  extracted-checks workflow path filters.
- The cross-layer caller hint for the result payload helper is name-only. The
  referenced scripts define their own local helpers; they do not import or call
  the paid-unlock smoke helper changed here.
- AI reconciliation: fixed the Codex P2 by suppressing `error_detail` for the
  expected pre-webhook locked artifact response while preserving detail for
  unexpected webhook/artifact failures.

## Deferred

- Live #1440 paid unlock and email/PDF delivery remain blocked until the
  deployed Stripe webhook signing secret matches the secret used by the smoke
  or the deployed secret is rotated to include it.
- The portfolio result-page `404` and repeat-volume gate calibration remain
  separate #1440 blockers and are not changed here.

Parked hardening: none.

## Verification

- Command: pytest tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py - 10 passed.
- Command: python -m py_compile scripts/smoke_content_ops_deflection_stripe_paid_unlock.py - passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh - 4183 passed, 10 skipped, 1 existing torch warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr_body_deflection_paid_unlock_diagnostics.md - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/content_ops_faq_deflection_submit_handoff_runbook.md` | 6 |
| `plans/PR-Deflection-Paid-Unlock-Diagnostics.md` | 108 |
| `scripts/smoke_content_ops_deflection_stripe_paid_unlock.py` | 50 |
| `tests/test_smoke_content_ops_deflection_stripe_paid_unlock.py` | 49 |
| **Total** | **213** |
