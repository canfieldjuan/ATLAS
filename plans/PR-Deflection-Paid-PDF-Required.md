# PR-Deflection-Paid-PDF-Required

## Why this slice exists

#1921 split the launch proof into a checked runbook plus guard slices for the
buyer-facing delivery surfaces. The Snapshot guard is handled in
atlas-portfolio #473. The paid report path still has the same shape: when
`render_deflection_full_report_pdf(...)` fails, `_pdf_attachments(...)` catches
the exception, returns no attachments, and `send_pending_deflection_report_deliveries(...)`
still sends a link-only email and marks the delivery row `delivered`.

Root cause: the paid delivery worker treats the PDF attachment as optional even
though the launch contract and customer-facing paid deliverable require the
curated PDF attachment. This change fixes the worker boundary root by making PDF
render failure a delivery failure before the sender is called.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Production hardening

1. Make paid report PDF render/missing-artifact failures fail the delivery row
   instead of sending a link-only paid email.
2. Update the existing worker regression that currently locks in link-only
   fallback so it proves no sender call, failed status, and an incident.
3. Keep idempotent replay coverage for regenerated-but-different PDF payloads
   so the provider conflict branch remains protected.
4. Treat stale `sending` reclaims differently from first-attempt `pending`
   sends when PDF rendering fails, so a transient render outage cannot
   terminal-fail a delivery row after the buyer may already have received the
   first email.

### Review Contract

- Acceptance criteria:
  - A paid report delivery with a malformed/missing artifact does not call
    `sender.send(...)`.
  - A paid report delivery whose PDF renderer raises does not call
    `sender.send(...)`.
  - Both cases mark the delivery row failed with a bounded error.
  - Both cases emit a paid-funnel incident so the launch proof can see the
    missing paid PDF as a hard failure.
  - Stale `sending` reclaims whose PDF renderer fails do not call
    `sender.send(...)` and are reset to `pending` with a warning incident
    instead of being marked terminal `failed`.
  - Successful deliveries still attach the PDF and mark delivered.
- Affected surfaces:
  - Paid report delivery worker only.
  - Focused worker tests only.
- Risk areas:
  - Paid buyer email delivery.
  - Delivery status lifecycle.
  - Launch proof passing on link-only paid email.
- Triggered reviewer rules:
  - R1 Requirements match.
  - R2 Test evidence.
  - R3 Security/auth/money path.
  - R8 Persistence/data lifecycle.
  - R14 Codebase verification.

### Files touched

- `atlas_brain/content_ops_deflection_delivery.py`
- `docs/INCIDENT_RESPONSE.md`
- `plans/PR-Deflection-Paid-PDF-Required.md`
- `tests/test_atlas_content_ops_deflection_delivery.py`

## Mechanism

Replace `_pdf_attachments(...)`'s silent fallback with a small required renderer
helper that raises when the artifact is missing/malformed or when
`render_deflection_full_report_pdf(...)` fails. The claim query now returns the
row's previous delivery status before it is claimed as `sending`, so the worker
can distinguish first-attempt `pending` sends from stale `sending` reclaims.

First-attempt PDF render failures still emit `paid_report_delivery_send_failed`,
mark the row failed, increment `failed`, and skip `sender.send(...)`. Stale
reclaim PDF render failures emit
`paid_report_delivery_pdf_render_reclaim_deferred`, reset the row to `pending`,
increment `failed` for this run, and skip `sender.send(...)`; the next worker
run can retry rendering instead of terminal-failing a row whose first email may
already have reached Resend. The new warning incident is listed in the paid
funnel incident response catalog so the security-policy docs gate and launch
triage surface stay aligned with the worker emitter.

The test that currently expects link-only fallback becomes the regression for
the new contract. It asserts the worker records a failed delivery, no email
request is built, and the incident payload names the PDF failure.

## Intentional

- No CLI flag or configuration escape hatch. Paid report PDF attachment is a
  launch contract, not an optional enhancement.
- Dry-run remains queue-only and does not render PDF. The runbook already names
  dry-run as queue/paid/email gating only; live-send and local render proof own
  PDF rendering.
- This slice does not change delta emails. Deltas are a separate subscription
  surface and do not currently attach a paid report PDF.
- Tests use the existing worker pool/sender harness because this slice changes
  the worker branch before sender execution and does not change SQL or adapter
  behavior. The idempotent replay regression still uses the real
  `ResendCampaignSender` with a transport fake.

## Deferred

- #1921 PR 4 still owns the deployed full-funnel proof artifact with a real
  opted-in buyer email and attached paid PDF.

Parked hardening: none.

## Verification

- `pytest tests/test_atlas_content_ops_deflection_delivery.py -q` - 36 passed,
  1 skipped.
- `python -m unittest tests.test_security_policy_docs` - 20 passed.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Paid-PDF-Required.md --check`
  - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/content_ops_deflection_delivery.py` | 69 |
| `docs/INCIDENT_RESPONSE.md` | 1 |
| `plans/PR-Deflection-Paid-PDF-Required.md` | 127 |
| `tests/test_atlas_content_ops_deflection_delivery.py` | 111 |
| **Total** | **308** |
