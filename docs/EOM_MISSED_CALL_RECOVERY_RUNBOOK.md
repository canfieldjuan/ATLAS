# EOM missed-call recovery runbook

This runbook operates the Atlas-owned recovery sequence that begins only after
an authenticated office operator records an unanswered call for a current EOM
residential estimate lead. It does not alter public form acknowledgement,
confirm an estimate appointment, or send a message merely because a form was
submitted.

## Ownership and current deployment shape

- Atlas owns call-attempt evidence, sequence/step state, eligibility,
  scheduling, delivery evidence, retries, and cancellation.
- The tracker proxies authenticated operator actions; the Website's CRM Leads
  page displays status and never calls Atlas or Resend directly.
- The 2026-08-21 preflight found production on the full
  `atlas_brain.main:app` service. Re-check the active unit before every deploy;
  the slim `atlas_brain.main_eom:app` configuration is kept compatible but is
  not assumed to be the live process.

Do not copy a booking URL, EOM service token, email token, contact email, or
phone number into this document, shell history, PR text, fixtures, or logs.

## Configuration

Set these values in the authoritative Atlas runtime configuration. The booking
URL must be a real public Google Calendar appointment-request URL; it is not an
Atlas private booking endpoint and requesting a time does not confirm an
appointment.

| Setting | Safe value / purpose |
| --- | --- |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_RECOVERY_ENABLED` | `false` by default. `true` permits the worker only after all preflight checks below pass. |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_BOOKING_LINK` | Private deploy-time Google Calendar URL. Never commit it. A blank value produces an inspectable blocked sequence and sends nothing. |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_TIMEZONE` | `America/Chicago` unless EOM's business timezone changes through an approved policy update. |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_POLL_INTERVAL_SECONDS` | Optional bounded worker interval; default `60`. |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_MAX_DELIVERY_ATTEMPTS` | Optional bounded definite-rejection retry limit; default `3`. |
| `ATLAS_EOM_FUNNEL_MISSED_CALL_DELIVERY_TIMEOUT_SECONDS` | Optional bounded provider request timeout; default `10`. |

The existing Atlas email configuration remains authoritative for sender and
transport. Do not add a second EOM Resend key or browser-visible sender
configuration. If email transport is unavailable, Atlas records the call but
creates `blocked_configuration / email_transport_unavailable`; it does not
claim or send a step.

## Safe rollout order

1. Deploy the Atlas provider code and migration `389_eom_missed_call_recovery`
   with recovery disabled. Verify the full application starts, the migration is
   recorded, and no worker is running. On the compatible slim EOM profile,
   `ATLAS_EOM_RUN_MIGRATIONS=true` applies this additive schema even while the
   recovery flag remains disabled; that flag controls delivery, not schema
   readiness.
2. Deploy the tracker capability-backed proxy after Atlas is serving the named
   endpoints.
3. Deploy the Website CRM Leads card after the tracker advertises the exact
   capability names and routes.
4. Configure the private booking link and validate the process still starts
   with recovery disabled. Do not place a placeholder or test link in customer
   configuration.
5. Verify the existing email transport is enabled and the sender remains the
   established EOM sender. Then set recovery enabled and restart the Atlas
   service.
6. Use only controlled non-customer test data and a fake/sandbox provider for
   pre-production verification. Do not test this by emailing a real lead.

The sender is dormant until a real operator action creates an eligible sequence.
Form submission by itself cannot create one.

Email 2 becomes due at 09:00 on the next Monday-Friday date in the configured
time zone. Email 3 becomes due three calendar days after Atlas records
successful delivery of Email 2, so a delayed second email never compresses the
final follow-up interval.

## Operator-visible states and recovery

| State | Meaning | Safe response |
| --- | --- | --- |
| `active` | Eligible sequence with a pending step or a completed earlier step. | The worker rechecks current lead state immediately before each send. |
| `blocked_configuration` | Call evidence was saved, but recovery is disabled, no booking URL is present, or transport is unavailable. | Correct deploy configuration, then use the explicit resume action. Configuration alone does not silently send an old sequence. |
| `cancelled` | A current lifecycle/response/suppression/recipient rule stopped future steps. | Do not restart it from the old call. Record a new real call only if follow-up is again appropriate. |
| `completed` | All three approved messages were confirmed as sent. | No further automation occurs. |
| `failed` | A definite provider rejection exhausted its bounded retries. | Investigate the provider/configuration. Do not assume an email was sent. A new sequence requires a new real call attempt. |
| `recovery_required` | Provider delivery could not be proved without risking a duplicate message, or its idempotency window elapsed during recovery. | Treat as potentially sent. Do not retry or resend automatically; verify externally and use a separately recorded, deliberate operator action if needed. |

Every current sequence state and step event is durable Atlas evidence. The
legacy `sent_emails` history write is secondary: a post-send history failure
never reverses truthful sequence delivery state or authorizes another send.

## Immediate safety stop / rollback

To halt new recovery mail, set
`ATLAS_EOM_FUNNEL_MISSED_CALL_RECOVERY_ENABLED=false` and restart the Atlas
service. Startup durably moves every currently active sequence to
`blocked_configuration / recovery_disabled`, then stops the worker while
retaining immutable call attempts and sequence history. Restoring the flag does
not send those overdue emails: an authenticated operator must explicitly resume
each still-eligible sequence. Do not drop migration 389 tables, events, or
steps as a routine rollback.

After an incident, retain the rows and inspect the sequence state, step state,
provider message identifier, event ledger, and current contact/interactions
before any manual customer outreach. A missing delivery confirmation is never
evidence that a message was not accepted.

## Current proof boundary

Atlas stops a sequence from state it can prove: a recorded inbound CRM response,
callback/conversation record, explicit opt-out, new estimate request, lifecycle
advancement, customer conversion, loss/closure, commercial classification,
recipient change/invalidity, or cancellation. The current system does **not**
prove EOM email-reply correlation, eVoice call events, or a Google Calendar
appointment-request selection for a canonical contact. Those channels must be
recorded through the CRM/operator flow until a separately verified ingestion
slice exists.
