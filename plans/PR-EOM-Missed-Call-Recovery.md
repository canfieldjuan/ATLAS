# PR-EOM-Missed-Call-Recovery

## Why this slice exists

EOM's public estimate form already captures a lead and sends one immediate
transactional acknowledgement. There is no canonical record for Juan's later
unanswered call and no durable, eligibility-gated recovery sequence. ATLAS
#2474 previously scoped a first-email exploration; the approved product
contract now supplies the three messages, timing, stop conditions, and
explicit-operator trigger for this provider slice.

Diff-budget override: this provider slice is intentionally above the 400-LOC
target (about 5,000 added lines including real-Postgres proof and operating
documentation) because a safe vertical boundary needs its additive database
schema, durable two-phase provider claim, exact approved copy, both supported
app lifespans, capability contract, controlled real-Postgres proof, and
operating runbook in the same deployable change. Splitting those pieces would
either expose a mutable/browser-owned sequence or ship a sender without its
safety state. Tracker and Website consumption remain separate, smaller
follow-up PRs.

### Problem-derived contract

- Root cause: an authenticated EOM operator cannot persist a no-answer outcome
  against a canonical lead or safely cause a retryable follow-up. The generic
  telecom callback has no proved lead identity and must not send customer mail.
- Correct fix must touch/change: an additive Atlas migration, an Atlas-owned
  call-attempt/sequence/step/event state machine, deterministic approved email
  rendering, deploy-time configuration, a bounded in-process EOM worker, named
  capability-backed funnel routes for the later tracker/UI bridge, and an
  operating/recovery runbook.
- Must not change: the public Website -> Web3Forms -> Atlas intake path, the
  immediate acknowledgement copy and idempotency, generic telecom callbacks,
  onboarding drafts, B2B campaigns, existing lead lifecycle commands, or the
  in-flight tracker #237 / Website #258 CRM-directory work.

## Scope (this PR)

Ownership lane: eom/missed-call-recovery
Slice phase: Vertical slice
Max files: 13

1. Add a provider-owned missed-call recovery sequence for one eligible
   residential web-estimate lead after a deliberate, actor-authenticated
   `Called - no answer` operation. It records immutable evidence before it
   queues any email, calculates step 1 immediately, step 2 at 09:00 on the
   next Monday-Friday business day in the configured EOM time zone, and step 3
   three calendar days after successful delivery of step 2 at the same
   configured local clock time.
2. Add a locked, idempotent outbox worker with a durable pre-send claim and
   stable Resend idempotency key, bounded retry only while that key is still
   provider-valid, and an explicit `recovery_required` terminal state for
   ambiguous delivery. It re-reads the canonical
   contact/interaction/suppression state immediately before send and records
   every send, skip, retry, cancellation, failure, or configuration block in an
   append-only sequence-event ledger.
3. Expose only additive Atlas routes: record no-answer, bounded status batch,
   explicit resume after a configuration block, and explicit cancellation for
   a manually verified callback/response/opt-out. The consumer bridge is
   intentionally deferred until the published heads of tracker #237 and
   Website #258 are stable.

### Review Contract

- Acceptance criteria:
  - `record_no_answer` commits one actor-attributed call attempt and one active
    sequence only for a current EOM lead whose latest intake proves the
    residential estimate variant; focused tests prove form submission alone has
    no sequence, commercial/closed leads do not start one, and same-key retries
    return the original call-attempt result.
  - A globally unique, append-only operation receipt binds every no-answer,
    resume, or cancellation idempotency key to one contact, mutation kind, and
    request fingerprint before state changes. Cross-contact/key-reuse fails
    closed; a unique active-sequence constraint plus transaction-scoped contact
    and sequence locks makes contemporaneous qualifying calls converge on one
    sequence. A first transaction persists `attempting`, a claim token, and
    the provider key expiry before external I/O; the second transaction locks
    and rechecks current state immediately before delivery. The real-Postgres
    concurrent-worker test proves one external gateway call, and the
    crash-after-claim test proves expired unconfirmed work becomes visible
    `recovery_required` rather than reusing a stale provider key.
  - Each persisted step has deterministic subject/body/recipient snapshot and
    a stable provider key. The Resend gateway attaches that key and retries only
    inside its 24-hour evidence window; tests prove success, definite rejection,
    retry exhaustion, and ambiguous crash/transport recovery do not duplicate
    a send.
  - The eligibility query and contacts/contact-interactions triggers cancel
    remaining work for lifecycle advancement, customer conversion, lost or
    archived state, commercial reclassification, later estimate request,
    tracked inbound response, suppression, missing/invalid recipient, or an
    explicit cancellation. Recipient-change cancellation compares the actual
    latest estimate-form recipient rather than a non-authoritative contact
    email correction, and interaction cancellation requires evidence occurring
    after the sequence began. SMS is positive-evidence-gated so a generic
    outbound CRM SMS cannot impersonate a lead reply. Tests settle each
    permitted proof path and show a change after scheduling but before delivery
    skips the send.
  - The booking link is accepted only from deploy-time configuration and is
    never returned by status endpoints or logged. A missing link, disabled
    recovery flag, or unavailable email transport leaves a visible blocked
    sequence and no email. Worker startup and each dispatch persist that block
    for every previously active sequence. A pre-send claim racing that pause is
    preserved as unproven provider evidence, not skipped; restoring
    configuration requires an explicit resume and cannot silently send an
    overdue email.
  - Both `main.py` and `main_eom.py` call one shared prepare boundary during a
    canonical-database lifespan. It blocks active rows when disabled or
    misconfigured, starts only a delivery-ready worker, and makes an enabled
    partial schema a startup failure.
  - Existing acknowledgement tests stay unchanged and all new fixtures use
    `.example.test` identities plus a recording gateway; no test uses an Atlas
    email credential or a real recipient.
- Reachability proof: `POST /api/v1/eom-funnel/leads/{contact_id}/missed-call-attempts`
  -> transactionally persisted call attempt/sequence/step rows -> background
  worker claim -> current-state recheck -> Resend request with durable
  idempotency key -> sequence event and tenant-scoped sent-email history. The
  companion `GET /missed-call-recovery-status` exposes only status and next due
  time for the tracker bridge.
- Affected surfaces: EOM funnel configuration/startup, canonical contacts and
  interactions, migration/readiness contract, Resend send adapter, and the
  named Atlas service routes. Tracker #237 and Website #258 are explicitly out
  of this provider PR.
- Risk areas: customer-email duplication, recipient/config leakage, stale
  lifecycle state, multiple workers, partial migration/startup, retry after
  unknown delivery result, branch collisions, and accidental changes to intake
  acknowledgement behavior.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R11, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: authenticated tracker -> Atlas no-answer mutation;
  Atlas worker -> Resend; contact/interaction state -> eligibility and
  cancellation; configuration -> email rendering/startup readiness.
- Replaced-path behaviors: none. The public intake acknowledgement and telecom
  callback remain independent and unchanged.
- Guard-relevant fields: EOM tenant, contact id, contact type/status/stage,
  customer type, latest intake variant, snapshot recipient, operation key,
  worker claim lease/provider idempotency key, booking URL, configuration
  enablement, current interaction/suppression evidence.
- Caller x input shape: tracker service credential + authenticated actor + a
  UUID lead path and standard idempotency key; untrusted/malformed path and
  JSON input fail before a CRM write; absent/older consumers omit all new calls
  and therefore observe no behavior change.

### Guard-class closure

- **Lifecycle admission (`lead`, `active`, `new`, non-commercial, residential
  intake): CLOSED / DERIVED / BLOCK.** Membership comes from the canonical
  contact row plus the latest `web_form` estimate interaction already produced
  by `atlas_brain/api/leads.py`; anything outside the affirmative shape records only the
  call evidence and never creates a sequence. This is the safer side because a
  false admission can send customer mail.
- **Recovery states, step numbers, cancellation reasons, and capability names:
  CLOSED / AUTHORED HERE or DERIVED / OMIT-BLOCK.** Migration 389 owns the
  finite state/reason/step sets; route capability membership is mechanically
  derived from registered router paths. Unknown database/API values are not
  rendered as a sendable state, and unregistered capabilities are omitted so a
  consumer disables its control.
- **Booking-link host admission: CLOSED / AUTHORED HERE / BLOCK.** The only
  approved public Google Calendar hosts are `calendar.google.com` and
  `calendar.app.google`; malformed, credential-bearing, root, or alternate-port
  URLs fail settings validation before a message can render. A generated
  scheme/host/credential/port/path/suffix grammar matrix proves the exact
  admission rule and unsafe-character rejection rather than approving only a
  fixed list of sample URLs.
- **Inbound-response recognition: OPEN / EVIDENCE-GATED / NO AUTOMATIC STOP.**
  Free-form CRM interaction history cannot prove that every generic `sms` row
  is inbound. The one currently verified EOM producer writes
  `crm_event_id="sms:<provider id>"`; an explicit `direction=inbound` is a
  future structural equivalent. All other SMS shapes continue the normal
  eligibility path rather than claiming a reply. Dedicated inbound/response
  interaction types, lifecycle transitions, suppression, and explicit operator
  cancellation remain positive stop evidence. This preserves truthfulness while
  the missing eVoice/email/calendar correlation is deferred.

### Deployed-config probing

- Deployed/default config values: recovery is disabled by default, booking URL
  is blank by default, time zone defaults to `America/Chicago`, worker polling
  and retry limits are bounded typed settings.
- Explicit value probe: valid enablement plus HTTPS Google Calendar booking URL
  yields rendered/persisted steps; time-zone tests cover Friday and weekend
  origins.
- Absent value probe: disabled or missing booking URL creates a blocked,
  inspectable sequence with no rendered/sendable step and no provider call.
- Default-session/default-context probe: a lead outside `effingham_maids`, a
  non-residential intake, or a non-active/new contact is not admitted; status
  reads never disclose another tenant's result.
- Side-effect ordering: call attempt + sequence + event commit first; worker
  claim and fresh eligibility check commit before provider I/O; provider result
  becomes durable delivery evidence before optional sent-email-history write.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_missed_call_recovery.py`
- `atlas_brain/storage/migrations/389_eom_missed_call_recovery.sql`
- `atlas_brain/templates/email/missed_call_recovery.py`
- `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md`
- `plans/PR-EOM-Missed-Call-Recovery.md`
- `render.eom.yaml`
- `tests/test_eom_missed_call_recovery.py`
- `tests/test_eom_render_profile.py`

## Mechanism

Migration 389 creates five additive EOM-only tables: globally unique immutable
operation receipts; immutable no-answer call attempts; one current sequence per
contact; exactly three deterministic outbox steps; and an append-only event
ledger. The operation receipt binds key, contact, mutation kind, and request
fingerprint before a mutation, so a stale retry cannot cross contacts or reuse
another recovery action. The operator mutation also appends a canonical
`contact_interactions` `call` / `no_answer` record and lifecycle event, so
existing CRM history retains the action. Partial unique indexes and
transaction-scoped row locks make an active sequence non-overlapping.

The sequence starts only when the canonical contact is active, a lead at the
`new` stage, has current residential web-estimate evidence, and has a valid
recipient. Existing `unknown` customer type is permitted only because the
web-form interaction carries the already-used residential acknowledgement
variant; an explicit later commercial classification cancels the sequence. The
contact, lifecycle, and interaction axes otherwise remain semantically
unchanged.

An explicit operator cancellation acquires/cancels the current sequence before
it appends the ordinary CRM interaction in the same transaction. That ordering
preserves the authenticated actor and stated reason in the sequence-event
ledger; the interaction trigger then has no active sequence left to cancel.

The worker first claims a due row with `FOR UPDATE SKIP LOCKED` and commits
`attempting`, a random claim token, and a 23-hour provider-key safety window
**before** any provider call. A second transaction locks the exact token,
re-reads all eligibility, sends with `Idempotency-Key:
eom-missed-call/<step UUID>`, and commits provider acceptance. A retry keeps
the same immutable payload/key. A definite pre-acceptance rejection may retry
up to the configured limit; an uncertain result, claim crash beyond the
provider-key window, or unrecognized idempotency conflict becomes
`recovery_required` rather than authorizing a duplicate email.

The current code has no verified EOM email-reply correlation and eVoice does
not emit canonical call/SMS events. The service therefore cancels only for
responses it can prove now (a recorded inbound CRM interaction or an explicit
operator cancellation), plus existing lifecycle and recipient evidence. This
limitation is exposed in the issue/operating notes rather than represented as
an unsupported automatic claim.

## Intentional

- Do not infer a no-answer from SignalWire/eVoice callbacks: the current
  callback does not prove a canonical EOM contact.
- Do not use the B2B campaign tables: their tenant, unsubscribe, and delivery
  semantics are not the EOM CRM contract.
- Do not use browser timers or direct browser email calls.
- Use next weekday at 09:00 local time for email 2 because current EOM code
  has no business-holiday calendar or office-hours scheduling representation;
  email 3 is exactly three local-calendar days after successfully delivered
  email 2 at the same local clock time, including across daylight-saving time.
- Persist a short claim transaction before external I/O, then hold the
  canonical contact lock only during the final recheck and bounded outbound
  request. A concurrent lifecycle/response commit therefore linearizes either
  before the final recheck (skip) or after the send (truthfully current at
  send), while a process crash retains the claim/key window as durable evidence.
- Use the real Resend adapter in a mock transport test so accepted, definite
  rejection, and ambiguous response classifications prove its exact wire
  behavior without sending an actual email.
- Preserve any durable pre-send claim across a configuration pause. After an
  explicit resume, the normal provider-key recovery window decides whether it
  can be safely retried or must become `recovery_required`; the pause itself
  never rewrites that unknown outcome as a skipped message.

## Deferred

1. Tracker proxy + Website lead-card action/status integration follows published
   tracker #237 then Website #258, so it does not race their active
   `time_tracker_api.py` / `lead-review.js` edits. See ATLAS #2474.
2. EOM-specific inbound email reply correlation and eVoice/telephony event
   ingestion are not currently evidenced; add them as a separate verified
   intake slice before claiming automatic coverage beyond recorded interactions.
   This is parked in ATLAS #2474 because an uncorrelated inbox/call record is
   not safe stop evidence.
3. A holiday calendar/office-hours policy is deferred; the initial definition
   of "business day" is Monday-Friday in the configured time zone.

Parking predicate: inbound-channel correlation and holiday policy are deferred
only because no current canonical producer can prove them. Customer-mail safety,
sequence correctness, and deployment readiness remain in this provider slice.

Parked hardening: ATLAS #2474 follow-up for verified EOM email/eVoice/Google
Calendar correlation.

## Verification

- `ruff format --check atlas_brain/services/eom_missed_call_recovery.py
  atlas_brain/templates/email/missed_call_recovery.py
  tests/test_eom_missed_call_recovery.py` — pass.
- `python -m py_compile atlas_brain/services/eom_missed_call_recovery.py
  atlas_brain/templates/email/missed_call_recovery.py atlas_brain/eom_api/config.py
  atlas_brain/eom_api/funnel.py atlas_brain/main.py atlas_brain/main_eom.py` —
  pass.
- `ruff check --ignore E402 atlas_brain/eom_api/config.py
  atlas_brain/eom_api/funnel.py atlas_brain/main.py atlas_brain/main_eom.py
  atlas_brain/services/eom_missed_call_recovery.py
  atlas_brain/templates/email/missed_call_recovery.py
  tests/test_eom_missed_call_recovery.py tests/test_eom_render_profile.py` —
  pass. `E402` is the existing two-entrypoint dotenv-before-import startup
  pattern; this slice does not reformat or move that runtime behavior.
- `pytest -q tests/test_eom_missed_call_recovery.py
  tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package
  tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_and_receivables_auth
  tests/test_eom_render_profile.py::test_eom_missed_call_recovery_migration_helper_uses_funnel_curated_set
  tests/test_leads_intake.py` — `99 passed, 37 skipped, 1 upstream pynvml
  deprecation warning`; the real-Postgres cases are correctly skipped locally
  because `ATLAS_MIGRATION_TEST_DATABASE_URL` is unavailable.
- `git diff --check` — pass.
- Full pipeline, migration integration, and security checks run on GitHub CI
  after the ready-for-review PR is opened; do not duplicate that full suite
  locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 11 |
| `atlas_brain/eom_api/config.py` | 128 |
| `atlas_brain/eom_api/funnel.py` | 218 |
| `atlas_brain/main.py` | 35 |
| `atlas_brain/main_eom.py` | 66 |
| `atlas_brain/services/eom_missed_call_recovery.py` | 2337 |
| `atlas_brain/storage/migrations/389_eom_missed_call_recovery.sql` | 602 |
| `atlas_brain/templates/email/missed_call_recovery.py` | 114 |
| `docs/EOM_MISSED_CALL_RECOVERY_RUNBOOK.md` | 111 |
| `plans/PR-EOM-Missed-Call-Recovery.md` | 340 |
| `render.eom.yaml` | 9 |
| `tests/test_eom_missed_call_recovery.py` | 1905 |
| `tests/test_eom_render_profile.py` | 79 |
| **Total** | **5955** |
