# PR-EOM-Lead-Ntfy

## Why this slice exists

The operator (Juan) asked: "I want to know if we can get a ntfy when a lead
comes in. We have a webhook that fires when a form is submitted... I get a lot
of emails and tend to ignore them" — and chose "a dedicated eom-leads topic".
Today a website estimate submission writes the CRM and (optionally) emails the
lead an acknowledgement, but the operator's only inbound signal is email, which
he does not reliably see. This slice adds an instant push notification to a
dedicated ntfy topic the moment a new lead is captured, so a real lead surfaces
outside the email noise. It mirrors the pattern already in production for the
atlas-api healthcheck and paid-deflection alerts (private ntfy.sh topic).

### Problem-derived contract

- Root cause: The lead-intake path (`_process_lead_intake` in
  `atlas_brain/api/leads.py`) has no operator-facing real-time signal; the only
  notification is the lead-facing acknowledgement email, and even that is
  skipped for phone-only leads and past the hourly ack cap. The operator can
  miss a new lead entirely.
- Correct fix must touch/change: Emit a push notification on a *new* lead
  capture. It must fire exactly when a lead is freshly logged (the same
  `freshly_logged` flag that gates the ack email), independent of the email path
  so a phone-only lead still notifies; it must be fire-and-forget so a
  notification failure never fails the already-committed lead; and it must be
  configurable + off-by-default via a dedicated topic so it does not fire in
  tests or environments that have not opted in. Add a `leads_ntfy_topic` field
  to `AlertsConfig` (env `ATLAS_ALERTS_LEADS_NTFY_TOPIC`), a fire-and-forget
  publisher reusing the existing `ntfy_enabled`/`ntfy_url` alerts config, an
  injectable notifier hook in `_process_lead_intake`, and tests.
- Must not change: No DB schema/migration; no change to the CRM write, the
  honeypot drop, the throttle, the same-day dedupe, the acknowledgement-email
  behavior, tenant/source stamping, or CORS. No new dependency (httpx is already
  used by `atlas_brain/tools/notify.py`). The existing single-topic
  `NotificationTool` alert path is untouched — the leads push is a separate,
  dedicated topic. Off by default: with no `leads_ntfy_topic` configured, intake
  behaves exactly as before.

## Scope (this PR)

Ownership lane: eom-crm/lead-notify
Slice phase: Vertical slice

1. Add `AlertsConfig.leads_ntfy_topic` (default `""`) and a fire-and-forget
   `_publish_lead_ntfy` + `_default_lead_notifier`; call the notifier from
   `_process_lead_intake` guarded by `if freshly_logged:` (injectable via a new
   `_notify_dependency`).
2. Add proof: fires on a new lead (with channels), does not on
   honeypot/duplicate, still fires for a phone-only lead, never fails the
   request on notifier error, and the transport posts to the configured topic
   only when enabled + a topic is set.

### Review Contract

- Acceptance criteria:
  - A freshly-logged lead invokes the notifier once with `(payload, email,
    phone_digits)` — settled by
    `tests/test_leads_intake.py::test_new_lead_fires_notification_with_channels`.
  - A honeypot submission and a same-day duplicate (`inserted=False`) do NOT
    notify — settled by `::test_honeypot_does_not_notify` and
    `::test_same_day_duplicate_does_not_notify`.
  - A phone-only lead (no email) still notifies — settled by
    `::test_phone_only_lead_still_notifies`.
  - A notifier that raises does not fail the intake (still `success: True`) —
    settled by `::test_notifier_failure_never_fails_request`.
  - The transport posts to `"{ntfy_url}/{leads_ntfy_topic}"` with
    `Title`/`Priority: high`/`Tags: moneybag` ONLY when `ntfy_enabled` is true
    AND a topic is set; it is skipped otherwise — settled by
    `::test_publish_posts_to_configured_leads_topic`,
    `::test_publish_skipped_when_topic_unset`,
    `::test_publish_skipped_when_ntfy_disabled`, and swallows transport errors
    (`::test_publish_swallows_transport_error`).
  - `ATLAS_ALERTS_LEADS_NTFY_TOPIC` populates `AlertsConfig.leads_ntfy_topic` —
    settled by the `env_prefix="ATLAS_ALERTS_"` on `AlertsConfig` (config.py:807)
    and confirmed by a direct `AlertsConfig()` env read.
- Reachability proof: Real entrypoint is `POST /api/v1/leads/intake`
  (`lead_intake` route → `_process_lead_intake`). Observable effect: on a new
  lead, an HTTP POST to the dedicated ntfy topic → push on the operator's phone.
  Off-by-default: the effect is inert until `ATLAS_ALERTS_LEADS_NTFY_TOPIC` is
  set in the runtime `.env`.
- Affected surfaces: `atlas_brain/api/leads.py` (notifier + hook + route DI),
  `atlas_brain/config.py` (`AlertsConfig`), `tests/test_leads_intake.py`.
- Risk areas: (1) the notifier blocking the intake response (bounded by a 5s
  httpx timeout; the website posts with keepalive pre-redirect and does not wait
  on the body; the ack email is already awaited); (2) firing on a duplicate or
  honeypot (gated by `freshly_logged`, which honeypot returns before);
  (3) leaking PII to a public relay (see Intentional).
- Reviewer rules triggered: R1 (requirements match), R2 (test evidence), R3
  (security/auth — PII on the relay), R5 (backward compatibility — additive,
  off by default), R6 (fail-safe: notification failure never fails intake),
  R11 (dependencies/config — new alerts field), R12 (deployment safety — `.env`
  topic add + service restart).

### Boundary-change enumeration

- Boundary path/seam: the notification gate in `_publish_lead_ntfy` —
  `if not alerts.ntfy_enabled or not topic: return` — and the emit gate in
  `_process_lead_intake` — `if freshly_logged:`.
- Replaced-path behaviors: previously there was no operator notification at all;
  the new gate adds one only on the freshly-logged + configured path. The
  honeypot early-return (before `freshly_logged` is computed) and the duplicate
  path (`freshly_logged=False`) keep their exact prior behavior (no notify).
- Guard-relevant fields: `AlertsConfig.ntfy_enabled` (bool),
  `AlertsConfig.leads_ntfy_topic` (str, empty = disabled), and the
  `freshly_logged` flag from `interaction["inserted"]`.
- Caller x input shape: new lead (insert) → notify; duplicate (no insert) → no
  notify; honeypot (early return) → no notify; validation/throttle error (raise
  before the CRM write) → no notify.

### Deployed-config probing

- Deployed/default config values: field default `leads_ntfy_topic=""` (feature
  OFF). Runtime `.env` already sets `ATLAS_ALERTS_NTFY_ENABLED=true` and points
  `ATLAS_ALERTS_NTFY_URL` at the public ntfy.sh server; this PR's deploy adds
  `ATLAS_ALERTS_LEADS_NTFY_TOPIC=eom-leads-6a01fbd83c92`.
- Explicit value probe: topic set + enabled → posts to the topic
  (`::test_publish_posts_to_configured_leads_topic`).
- Absent value probe: topic `""` → no HTTP client is even opened
  (`::test_publish_skipped_when_topic_unset`); `ntfy_enabled=false` → skipped
  (`::test_publish_skipped_when_ntfy_disabled`). The empty default keeps every
  other test in the module hermetic (no network).
- Default-session/default-context probe: N/A (no per-session context; global
  alerts config).
- Side-effect ordering: the notify fires AFTER the CRM write commits and
  `freshly_logged` is known, BEFORE the ack-email block, so a phone-only lead
  (which never enters the email block) still notifies; a notifier exception is
  caught and cannot short-circuit the email block or the response.

### Files touched

- `atlas_brain/api/leads.py`
- `atlas_brain/config.py`
- `plans/PR-EOM-Lead-Ntfy.md`
- `tests/test_leads_intake.py`

## Mechanism

`_process_lead_intake` already computes `freshly_logged` from the interaction's
`inserted` flag (the same signal that gates the acknowledgement email). Directly
after that, when `freshly_logged` is true, it awaits an injectable
`lead_notifier(payload, email, phone_digits)` inside a `try/except` that logs and
swallows any error. The route wires the real notifier via `_notify_dependency`
(`_default_lead_notifier`), which builds a title (`New lead: <name>`) and a
scannable body (`<phone> · <email>` / `<service> · <frequency>` / `<address>`)
and calls `_publish_lead_ntfy`. That publisher reads `settings.alerts`, returns
immediately unless `ntfy_enabled` is true and a `leads_ntfy_topic` is set, and
otherwise POSTs the body to `"{ntfy_url}/{topic}"` with `Title`,
`Priority: high`, and `Tags: moneybag` headers over a 5s-timeout httpx client.
Tests inject a fake notifier to assert call/no-call without HTTP, and patch
`httpx.AsyncClient` to assert the transport shape.

## Intentional

- PII on a public relay: the push body carries lead name/phone/email/address to
  ntfy.sh (a public server) on an unguessable random topic
  (`eom-leads-6a01fbd83c92`). This matches the trust model already accepted for
  the operator's healthcheck and paid-deflection topics (the topic string is the
  only secret). Chosen for parity with the operator's existing setup; if the
  operator later wants lead PII off the public relay, the `ntfy_url` can point at
  a self-hosted ntfy without any code change (the localhost default already
  exists). Noted for the operator, not blocked.
- Await (not background task): the notify is awaited, bounded by a 5s timeout,
  rather than detached — the intake path is an ASGI middleware where a detached
  task risks cancellation on request end, and the client does not wait on the
  response body. Simpler and safe.
- `Priority: high` (4), not `urgent` (5): a lead deserves prominence but is not
  an emergency like the atlas-api DOWN alert.

## Deferred

- None. (The topic subscription on the operator's phone + a real end-to-end lead
  test are deploy-time steps, done after merge.)

Parked hardening: none.

## Verification

- `/.venv/bin/python -m pytest tests/test_leads_intake.py -q` → 64 passed
  (48 prior + 16 new), run against the runtime venv.
- Config env wiring confirmed: `ATLAS_ALERTS_LEADS_NTFY_TOPIC=... AlertsConfig()`
  → `leads_ntfy_topic` populated.
- Post-merge deploy: add the `.env` topic line, restart `atlas-api.service`,
  subscribe the phone to the topic, submit a real test lead, confirm the push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/leads.py` | 86 |
| `atlas_brain/config.py` | 1 |
| `plans/PR-EOM-Lead-Ntfy.md` | 194 |
| `tests/test_leads_intake.py` | 226 |
| **Total** | **507** |
