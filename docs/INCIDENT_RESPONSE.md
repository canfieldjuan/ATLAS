# Incident Response

This runbook covers security and money-path incidents for ATLAS. It is the
operator companion to the public disclosure policy in `SECURITY.md`.

## Severity Levels

- SEV0 - Active compromise, exposed live secrets, unauthorized data access,
  payment unlock abuse, or ongoing customer-impacting data loss.
- SEV1 - Confirmed vulnerability with likely customer impact, failed paid report
  delivery after payment, cross-account access risk, or disabled security gate.
- SEV2 - Security control degraded without confirmed customer impact, repeated
  paid-funnel incident alerts, or a high-risk dependency requiring coordinated
  remediation.
- SEV3 - Low-risk vulnerability report, scanner finding, suspicious but
  unconfirmed activity, or documentation/process gap.

## Roles and Ownership

- Incident owner: the first responding Atlas operator. They keep the timeline,
  assign tasks, and decide when the incident can close.
- Technical lead: the engineer who owns the affected component or the smallest
  safe fix path.
- Communications owner: the operator who sends external updates and keeps
  private vulnerability reporters informed.
- Rotation owner: the person with provider access for Stripe, Anthropic,
  OpenRouter, Resend, AWS SES, SignalWire, Google, or other affected services.

If one person fills multiple roles, record that explicitly in the timeline.

## Intake and Triage

1. Open a private incident note with the report URL, reporter contact, received
   time, affected systems, suspected data class, and initial severity.
2. Confirm whether the issue affects ATLAS only, atlas-portfolio only, or a
   cross-repo flow.
3. Preserve evidence before changing state: relevant request IDs, account IDs,
   provider event IDs, deployment SHAs, sanitized logs, and screenshots.
4. Classify severity using the SEV table above. Raise severity if payment,
   authentication, authorization, customer ticket content, or credential exposure
   is involved.
5. Assign the incident owner, technical lead, communications owner, and rotation
   owner.

## Communications

- Acknowledge private vulnerability reports within five business days when
  possible, matching `SECURITY.md`.
- For SEV0 and SEV1, create an internal timeline immediately and update it at
  least every hour while containment is active.
- For SEV2, update the timeline at least once per business day until resolved or
  explicitly accepted.
- Keep public GitHub issues free of exploit details, secrets, customer data, and
  provider event IDs.
- If atlas-portfolio is involved, link the companion issue or incident note so
  the two repositories share one timeline.

## Paid Funnel Incident Types

Paid deflection funnel incidents are emitted with the
`DEFLECTION_PAID_FUNNEL_INCIDENT` marker. Treat these as payment-adjacent until
triage proves otherwise:

- `paid_report_checkout_terms_mismatch`
- `paid_report_missing_after_payment`
- `paid_report_revocation_missed_report`
- `paid_report_restore_missed_report`
- `paid_report_delivery_report_not_paid`
- `paid_report_delivery_missing_email`
- `paid_report_delivery_no_longer_sendable`
- `paid_report_delivery_idempotent_replay`
- `paid_report_delivery_pdf_render_reclaim_deferred`
- `paid_report_delivery_send_failed`

For these incidents, capture the Stripe event ID, checkout session ID, account
ID, request ID, delivery ID when present, and the deployed commit SHA. Do not
paste raw customer ticket content into the timeline.

## Credential Rotation

1. Treat credentials in git history, logs, screenshots, or third-party reports as
   exposed until the provider confirms revocation.
2. Create the replacement credential in the provider console or secret manager.
3. Update deployed secret stores for every consumer before revoking the old value
   when the provider requires overlap.
4. Revoke the old credential and record the provider confirmation time.
5. Run the narrow smoke check for the affected integration.
6. Update the incident timeline with the old credential name, provider, rotation
   time, verifier, and follow-up owner. Do not record secret values.

Stripe and deflection funnel credentials may be shared with atlas-portfolio, so
coordinate both repositories before revocation.

## Containment and Recovery

- Stop active abuse first: disable the affected route, revoke the credential,
  pause the webhook, or turn off the feature flag if needed.
- Prefer the smallest reversible mitigation while the root cause is still under
  investigation.
- Keep customer-facing claims truthful. If deletion, payment unlock, delivery,
  or security copy is no longer true, update or disable it before announcing
  closure.
- After the fix lands, verify it on the deployed path or the closest available
  local reproduction. Record the command, URL, provider event, or artifact used
  as evidence.

## Postmortem Template

Use this template for SEV0, SEV1, repeated SEV2 incidents, or any incident where
customer trust, payment integrity, or credential exposure was involved.

```text
Incident:
Severity:
Incident owner:
Technical lead:
Communications owner:
Opened:
Closed:

Summary:

Impact:

Root cause:

Timeline:
- YYYY-MM-DD HH:MM TZ -

Detection:

Containment:

Recovery verification:

Customer or reporter communication:

What worked:

What failed:

Follow-up actions:
- [ ] Owner - action - due date
```
