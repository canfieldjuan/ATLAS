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

**Why this slice is indivisible (over the 400-line soft cap).** The executable
change is ~90 LOC (`leads.py` + one config field); the rest is this plan and the
test matrix. It cannot be split smaller without shipping something unsafe or
unreviewable: the feature is a single guard-shaped emit path whose *correctness
is the gate matrix* — fires-on-new-lead, silent on honeypot/duplicate,
phone-only, notifier-failure-never-fails-intake, ASCII-only header, secret-topic
log redaction, true 5s + volume-query deadlines, the hourly cap on both sides,
and the disabled-path inertness. Landing the ~90 LOC without those tests would
ship an untested public, PII-handling, abuse-exposed surface; landing the tests
in a later PR would merge a guard whose second side is unproven. The plan doc is
mandatory (AGENTS.md) and grew with each Codex review round's closure/ordering
requirements. So the diff is dominated by the proof and the record, not by
divisible feature code — the narrowest viable slice is code + its full proof.

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
  (security/auth — PII + secret topic on the relay), R5 (backward compatibility
  — additive, off by default, inert when disabled), R6 (fail-safe: notification
  failure never fails intake), R7 (latency — true 5s wall-clock deadline), R8
  (resource/abuse — global hourly push ceiling), R11 (dependencies/config — new
  alerts field), R12 (deployment safety — `.env` topic add + service restart),
  R13 (defect class — the notification-gate closure declaration below).

### Boundary-change enumeration

**Closure declaration (notification-gate inventory).** The set of gates that
decide whether a new-lead push is emitted is **CLOSED** and **code-owned
(DERIVED** from the single emit path, not a literal list maintained here): it is
exactly the conjunction of gates on `_process_lead_intake` →
`_maybe_notify_new_lead` → `_default_lead_notifier` → `_publish_lead_ntfy`.
Membership is every branch on that one path; there is no side channel that emits
a push. **Outside-the-set default: no push.** Any input/condition that does not
satisfy every "emit" case defaults to NOT emitting — the safe and cheap side: a
missed push only loses a convenience signal (the lead is still captured; the ack
email and CRM write are unaffected), whereas an over-emit risks phone-spam or a
PII/secret-topic leak.

Enumerated gates (ALL must pass to emit; each is proven by a test):
1. `freshly_logged` — only a newly-inserted interaction (honeypot returns
   earlier; a same-day duplicate has `inserted=False`).
2. `lead_notifier is not None` — the route wires the production notifier; a
   direct caller that omits it emits nothing.
3. `_leads_push_configured()` — `ntfy_enabled` AND non-empty `leads_ntfy_topic`
   (re-checked inside `_publish_lead_ntfy`); off ⇒ inert (no volume query, no POST).
4. `await notify_volume() <= GLOBAL_NOTIFY_HOURLY_CAP` — global hourly ceiling;
   over the cap (or a volume-query error) ⇒ skip (lead still captured).

- Guard-relevant fields: `AlertsConfig.ntfy_enabled`, `AlertsConfig.leads_ntfy_topic`,
  `freshly_logged` (from `interaction["inserted"]`), the injected `lead_notifier`,
  and the hourly count from `notify_volume` / `_hourly_lead_notification_volume`.
- Caller x input shape: new lead (insert) + configured + under cap → notify;
  duplicate / honeypot / no-notifier / disabled / over-cap / volume-query-error /
  validation-or-throttle-error → no notify (fail-closed on the volume error).

**Cap concurrent-execution model (3k.4).** The hourly ceiling is a **best-effort
rate limiter, not an exact transactional counter**, and is intentionally so.
Execution model: each request runs `_hourly_lead_notification_volume()` as an
autonomous `READ COMMITTED` `COUNT` over committed `contact_interactions` rows,
then decides independently — there is no row lock, `SELECT … FOR UPDATE`,
advisory lock, or serializable snapshot shared across concurrent submissions, so
counting and publishing are deliberately not atomic. Property-level invariant it
guarantees: **the sustained push rate is bounded near the cap** — because every
lead's interaction row is committed before its own COUNT runs (the CRM write is
`await`ed first) and each subsequent request sees the accumulated rows, a
*sustained* flood converges to ≤ cap/hour. What it explicitly does NOT guarantee:
an exact per-hour count under a concurrent burst — up to `N-1` requests
in-flight at the same instant can each read a pre-threshold count and all emit,
overshooting by at most the momentary concurrency. That residual is accepted
because (a) the harm of a few extra pushes is trivial — a phone buzz, never a
security, billing, or data-integrity effect; (b) leads are low-volume and the
per-identity daily throttle already caps single-identity bursts; and (c) this is
the exact model of the sibling `GLOBAL_ACK_HOURLY_CAP` email ceiling this one
mirrors. A strict cap would need a serialized counter (a dedicated locked row or
a token bucket) whose write contention is not justified to shave a handful of
notifications off an adversarial burst that is already otherwise bounded.

**Open-input guard closures (3k.1 — class-closure, not string-closure).** Two
guards take open input and are closed at a choke point with a generative
property test, so a new adversarial string cannot reopen either finding:
- *Header safety* (free-text `name` → HTTP `Title`): the choke point
  `_header_value_ascii` keeps only printable ASCII (fail-closed default: drop
  anything else), so the Title is always a single ASCII line the transport
  accepts and the exact name rides the UTF-8 body. Closed by
  `test_lead_push_title_is_header_safe_for_all_inputs`, which generates 500 names
  across every codepoint family (C0/C1, CR/LF, latin-1, CJK, astral) and asserts
  the Title is always ASCII + single-line — not a fixture list of the reported
  strings.
- *URL construction* (`leads_ntfy_topic` → request URL path): the choke point
  `_SAFE_NTFY_TOPIC_RE` (`[-_A-Za-z0-9]{1,64}`) rejects any topic that could
  alter the path, failing closed (no HTTP client opened). Closed by
  `test_publish_rejects_url_unsafe_topic`.

### Deployed-config probing

- Deployed/default config values: field default `leads_ntfy_topic=""` (feature
  OFF). Runtime `.env` already sets `ATLAS_ALERTS_NTFY_ENABLED=true` and points
  `ATLAS_ALERTS_NTFY_URL` at the public ntfy.sh server; this PR's deploy adds
  `ATLAS_ALERTS_LEADS_NTFY_TOPIC=eom-leads-<secret-suffix>` — the real topic
  value is generated at deploy time and kept ONLY in the runtime `.env` (never
  committed; the topic string is the sole secret, so versioning it would leak
  lead PII to anyone with repo access).
- Pinned destination: lead PII is sent to `leads_ntfy_url` (default the public
  ntfy.sh relay), NOT to `ntfy_url`. `ntfy_url` is runtime-mutable via the
  unauthenticated public `PATCH /api/v1/settings/notifications`
  (`_NOTIFY_ALERTS_FIELDS`), so routing lead PII through it would let an attacker
  redirect it. `leads_ntfy_url` is deliberately absent from that mutable set, so
  the lead relay can only be set at deploy time; if it is blank the push is
  skipped rather than falling back to the mutable URL.
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
- `tests/conftest.py`
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
  ntfy.sh (a public server) on an unguessable random `eom-leads-<secret-suffix>`
  topic. This matches the trust model already accepted for the operator's
  healthcheck and paid-deflection topics (the topic string is the only secret).
  Because the topic IS the secret, it is generated at deploy time and kept only
  in the runtime `.env` — never in a versioned plan or commit. If the operator
  later wants lead PII off the public relay, the `ntfy_url` can point at a
  self-hosted ntfy without any code change (the localhost default already
  exists). Noted for the operator, not blocked.
- Await (not background task), bounded by a TRUE 5s wall-clock deadline
  (`asyncio.wait_for`, since httpx's own Timeout is per-phase): the intake path
  is an ASGI middleware where a detached task risks cancellation on request end,
  and the client does not wait on the response body. Simpler and safe.
- Failure logging never records the request URL or a raised transport error
  verbatim — the URL embeds the topic, which is the only secret. On failure we
  log a status code or the error class only.
- The hourly-volume DB query runs ONLY when pushes are enabled, so the
  off-by-default config stays inert (no extra COUNT/JOIN per lead).
- `Priority: high` (4), not `urgent` (5): a lead deserves prominence but is not
  an emergency like the atlas-api DOWN alert.

## Deferred

- The topic subscription on the operator's phone + a real end-to-end lead test
  are deploy-time steps, done after merge.
- PRE-EXISTING (out of this slice's scope, tracked separately): the public
  `PATCH /api/v1/settings/notifications` mutates `alerts.ntfy_url` without
  authentication, so it can already redirect the existing paid-deflection /
  healthcheck alert destinations. This PR closes the vector for LEAD PII (pinned
  `leads_ntfy_url`) but does not fix the settings-endpoint auth itself — filed as
  a follow-up issue so the broader alert channels get the same protection.

Parked hardening: none.

## Verification

- `/.venv/bin/python -m pytest tests/test_leads_intake.py -q` → 82 passed,
  run against the runtime venv. Includes the Codex hardening rounds: ASCII-only
  Title safety, route-level delivery proof, the hourly notification cap, the
  direct-caller no-op, secret-topic log redaction, a true 5s wall-clock
  deadline, and skipping the volume query when notifications are disabled.
- `maturity_sweep.py atlas_brain/api --min-score 8` → ratchet gate passed
  (no new brittleness; no baseline change).
- Config env wiring confirmed: `ATLAS_ALERTS_LEADS_NTFY_TOPIC=... AlertsConfig()`
  → `leads_ntfy_topic` populated.
- Post-merge deploy: add the `.env` topic line (deploy-only value), restart
  `atlas-api.service`, subscribe the phone to the topic, submit a real test
  lead, confirm the push.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/api/leads.py` | 224 |
| `atlas_brain/config.py` | 2 |
| `plans/PR-EOM-Lead-Ntfy.md` | 293 |
| `tests/conftest.py` | 19 |
| `tests/test_leads_intake.py` | 570 |
| **Total** | **1108** |
