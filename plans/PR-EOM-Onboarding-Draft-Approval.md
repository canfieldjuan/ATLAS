# PR-EOM-Onboarding-Draft-Approval

## Why this slice exists

A2 (merged) enqueues one pending onboarding-email draft per booked first
cleaning into `eom_onboarding_email_drafts`, whose migration-360 header
documents the exact send protocol the approval surface must execute: claim
`pending -> sending` atomically with the readiness predicate inline, send
OUTSIDE any open transaction with the draft id as the transport idempotency
key, confirm `sending -> sent` only after transport acceptance, and treat a
stuck `sending` row as operator reconciliation evidence rather than a
silent retry. Today nothing executes that protocol: drafts accumulate as
`pending` rows with no office surface to review, fix, approve, or revoke
them, and the funnel's office-controlled conversion decision (issue #2188)
requires that Atlas's first customer-facing email leave only on an explicit
office action. This slice is that surface (arc #2275, slice A3). Operator
decision (2026-08-04): per-draft subject/body editing is in scope.

Diff-budget justification: this is over the 400 LOC soft cap because the
protocol is indivisible at the transport boundary. The claim/confirm/revoke
state-machine methods, the idempotent transport sender, the approve
orchestration, the review/edit surface that makes blocked drafts sendable,
and the regression tests proving single-winner claims and stuck-`sending`
recovery must land together: shipping the claim without the confirm path
records unsent email as in-flight forever, shipping approve without the
blocker edit strands every no-email draft, and shipping any of it untested
leaves a double-send hazard on the first customer-facing send path.

### Problem-derived contract

- Root cause: the draft queue's send protocol lives only as documentation in
  migration 360; no code claims, sends, confirms, revokes, or repairs a
  draft. Two verified constraints shape the correct executor: the generic
  email provider port supports no transport idempotency key, and calling
  its send() inside the slim EOM Render profile imports the
  `atlas_brain.tools` registry whose dependencies are deliberately absent
  from `requirements.eom.txt` -- it would crash at request time AFTER the
  claim, manufacturing exactly the wedged-`sending` state the protocol
  exists to prevent.
- Correct fix must touch/change: add thin provider methods that implement
  the migration-360 state machine verbatim (guarded UPDATE ... RETURNING
  for claim/confirm/revoke, readiness predicate inside the claim, edit only
  while pending with recipient-sets-clear-blocker); add a service that
  orchestrates preflight -> claim -> send -> confirm with the transport as
  an injectable seam; send through a direct Resend POST over httpx with
  `Idempotency-Key: eom-onboarding-draft:<draft_id>` and Resend's 409
  invalid_idempotent_request treated as proof of prior delivery (the
  proven shape in atlas_brain/content_ops_deflection_delivery.py); record
  sent_emails history and a CRM interaction only after transport
  acceptance as secondary evidence that never flips the outcome; expose
  five private funnel routes (list/edit/approve-send/revoke/confirm-sent)
  on the existing auth stack; keep a failed transport in `sending` for
  operator reconciliation.
- Must not change: no email leaves without an explicit office action; no
  auto-retry or auto-reclaim of stuck `sending` rows (operator-in-the-loop
  per migration 360, diverging deliberately from deflection's 15-minute
  auto-reclaim -- the same 15-minute threshold instead gates when operator
  reconciliation may touch a `sending` row at all); no change to A2's
  enqueue, booking, or handoff semantics or tests; no change to the
  generic email provider port or its callers; no public/customer-facing
  route; no tracker UI.

## Scope (this PR)

Ownership lane: eom-lead-funnel-onboarding-approval
Slice phase: vertical slice

1. Add five private funnel routes under `/eom-funnel/onboarding-drafts`:
   the office queue list (closed projection, keyset pagination), pending
   draft edit (subject/body/recipient; recipient clears `no_email`),
   approve-send (two-phase claim/send/confirm), revoke (pending, or
   `sending` as reconciliation), and confirm-sent (`sending` verified in
   the transport log).
2. Add the migration-360 state-machine methods to `DatabaseCRMProvider`:
   `list_eom_onboarding_drafts`, `get_eom_onboarding_draft`,
   `update_eom_onboarding_draft`, `claim_eom_onboarding_draft`,
   `confirm_eom_onboarding_draft_sent`, `revoke_eom_onboarding_draft`.
   The claim additionally admits only drafts whose contact is still an
   active `effingham_maids` contact -- the same activity admission the
   booking family applies -- so archiving a lead after its draft was
   enqueued blocks approval with a precise 409.
3. Add `atlas_brain/services/eom_onboarding_drafts.py`: approve
   orchestration with injectable sender and sent-emails history seams,
   transport preflight before any claim, the direct Resend sender with
   deterministic idempotency key and 409-as-delivered, and
   post-acceptance evidence writers (sent_emails + CRM interaction) that
   log-and-continue on failure. Both default evidence writers bind to the
   CRM provider's own pool (public `pool` property; `EmailRepository`
   gains a pool override and `log_interaction` uses the provider pool),
   so evidence lands in the store that owns the draft even when the slim
   funnel profile points the provider at its own connection string.
4. Actor provenance: the claim stores `approved_by_employee_id/name`
   (widened to BIGINT by migration 361 to match the funnel actor boundary
   and the handoff precedent); revoke and confirm-sent record the acting
   employee through a CRM interaction note. Operator reconciliation of a
   `sending` row (revoke or confirm-sent) is admitted only once the claim
   is at least 15 minutes stale, so an active send between the transport
   POST and its confirmation can never be recorded as revoked or as
   delivered ahead of its actual outcome.
5. Tests: unit route/service coverage on fakes (state machine, failure
   matrix, sender contract, boundary guards), real-Postgres proofs
   (pipeline from first-clean booking through edited approved send,
   two-session single-winner claim at the provider layer, blocker
   resolution, stuck-`sending` reconciliation including the
   revoked-while-sending race, list projection), and slim Render profile
   route exposure.
6. Enroll the new service module in the EOM lead pipeline workflow path
   filters.
7. Gate the slim funnel on the actor-id widening: the shared funnel
   datastore guard additionally requires
   `approved_by_employee_id` to be BIGINT, so a canonical store with
   migration 360 but not 361 fails readiness closed instead of failing
   approvals after the claim. Recipient edits are bounded at the same
   254-character cap as the public intake email field.

### Review Contract

- Acceptance criteria:
  - [ ] All five draft routes reject disabled-API, missing/malformed
        bearer, and malformed actor evidence before any CRM call, settled
        by `tests/test_eom_lead_conversion.py`.
  - [ ] The queue list returns only the closed camelCase projection with
        contact full name, defaults to `pending`, rejects unknown status
        filters 422 before the CRM call, and paginates by keyset cursor,
        settled by `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Editing is admitted only while `status='pending'`; setting a valid
        recipient clears `blocker='no_email'`; blank fields, invalid email
        shapes, unknown fields, and empty edits reject 422 before the CRM
        call; non-pending drafts reject 409, settled by
        `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] Approve-send claims with migration 360's exact statement (readiness
        predicate inline, actor recorded), sends with
        `Idempotency-Key: eom-onboarding-draft:<draft_id>`, confirms
        `sending -> sent` only after transport acceptance, and returns 201
        with the transport message id, settled by
        `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] An already-sent draft replays 200 idempotent with no second
        transport call; blocked, recipient-less, in-flight `sending`, and
        revoked drafts reject 409 without a transport call, settled by
        `tests/test_eom_lead_conversion.py`.
  - [ ] Two concurrent approvals settle to exactly one claim winner at the
        provider layer against real Postgres, settled by
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] A transport failure leaves the row in `sending` (no auto-retry, no
        rollback to pending) and returns 502 naming reconciliation; while
        the claim is fresh both operator reconciliation actions
        (confirm-sent and revoke) reject 409 as still-in-flight, and once
        the claim is stale (15 minutes) they succeed with actor
        provenance; a zombie in-flow confirmation arriving after a stale
        revoke fails loudly instead of re-recording delivery, settled by
        `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The draft approver column stores the full funnel actor range:
        migration 361 widens `approved_by_employee_id` to BIGINT (handoff
        precedent), atomically with its ledger row, settled by
        `tests/test_migrations_runner.py` and the integration schema
        fixture applying 361.
  - [ ] Editing the new service module re-triggers both guarding CI lanes:
        the EOM lead pipeline workflow and the invoicing-checks workflow
        (which runs the slim Render profile import-isolation proof) both
        list `atlas_brain/services/eom_onboarding_drafts.py` in their path
        filters, settled by inspection of the two workflow files.
  - [ ] Resend's 409 invalid_idempotent_request is treated as proof of
        prior delivery: the draft confirms sent and the response flags the
        transport replay, settled by `tests/test_eom_lead_conversion.py`.
  - [ ] A missing or disabled transport configuration rejects 503 BEFORE
        claiming, so no row can enter `sending` that the deployment cannot
        send, settled by `tests/test_eom_lead_conversion.py`.
  - [ ] Confirmed delivery records sent_emails history
        (`business_context_id='effingham_maids'`,
        `template_type='onboarding_welcome'`, transport message id) and a
        CRM interaction, and failures of either writer never flip the send
        outcome, settled by `tests/test_eom_lead_conversion.py` and the
        real-Postgres pipeline proof (which runs with those writers'
        global pool deliberately uninitialized).
  - [ ] The slim EOM Render profile exposes all five routes without
        loading `atlas_brain.config`, the tools registry, or the full API
        package at import time, settled by
        `tests/test_eom_render_profile.py`.
  - [ ] A pending draft whose contact is no longer an active
        `effingham_maids` contact rejects approval 409 with zero state
        change, and restoring the contact makes the same draft claimable,
        settled by `tests/test_eom_lead_conversion.py` and
        `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The shared funnel datastore guard fails readiness closed against a
        store whose `approved_by_employee_id` column is still INTEGER
        (migration 360 without 361) and passes once it is BIGINT, settled
        by `tests/test_eom_lead_conversion_integration.py` when
        `ATLAS_MIGRATION_TEST_DATABASE_URL` is configured.
  - [ ] The default evidence writers bind to the CRM provider's own pool:
        with the global pool deliberately uninitialized, the sent_emails
        history row and the CRM interaction land in the store that owns
        the draft, settled by `tests/test_eom_lead_conversion.py` (service
        default-history seam proof) and the real-Postgres pipeline proof.
  - [ ] Recipient edits reject a 255-character address 422 before any CRM
        call, matching the public intake boundary's 254-character cap,
        settled by `tests/test_eom_lead_conversion.py`.
- Reachability proof: every route is exercised through the real FastAPI
  router with service-auth headers in `tests/test_eom_lead_conversion.py`,
  asserting both the JSON responses and the fake CRM/sender call
  sequences.
- Affected surfaces: private EOM funnel API, `DatabaseCRMProvider` draft
  methods, the new onboarding-draft service and its Resend sender,
  sent_emails history, contact interactions, slim Render profile route
  set, EOM lead pipeline workflow filters.
- Risk areas: double-send under concurrent approvals, wedged claims when
  transport fails, send-without-approval, blocked drafts becoming
  permanently unsendable, slim-profile import breakage, response-shape
  compatibility for the private tracker client.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R12.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: atlas_brain/eom_api/funnel.py
  - Replaced-path behaviors: no draft routes existed; drafts written by A2
    were unreachable by any office surface.
  - Guard-relevant fields: bearer token digest, `X-EOM-Actor`,
    `X-EOM-Actor-ID`, `draft_id` path UUID, list `status` filter (closed
    Literal set), `limit`, `cursor`, edit body (`subject`/`body` bounded
    non-blank, `recipient_email` validated against the same conservative
    pattern the public intake boundary uses, extra fields forbidden, at
    least one field required).
  - Caller x input shape: the server-side tracker/office caller only; the
    draft-action routes take no Idempotency-Key header by design -- the
    draft id plus the migration-360 status machine is the idempotency
    mechanism (booking/handoff headers CREATE operation identity; a
    draft's identity already exists) -- replays settle to 200-idempotent
    responses and concurrent approvals lose the atomic claim; list reads
    alter nothing.
- Boundary path/seam: atlas_brain/services/crm_provider.py
  - Replaced-path behaviors: the provider could only enqueue drafts and
    look them up by operation key; no state transitions existed.
  - Guard-relevant fields: draft `status` ladder
    (`pending`/`sending`/`sent`/`revoked`), `blocker`, `recipient_email`,
    `approved_by_employee_id`/`approved_by_name`, `claimed_at`/`sent_at`/
    `revoked_at`, list status allowlist.
  - Caller x input shape: claim admits only
    `pending AND blocker IS NULL AND recipient_email IS NOT NULL` plus an
    active `effingham_maids` contact (the booking family's activity
    admission) and is
    the single guarded UPDATE from migration 360's header; the in-flow
    confirm admits `sending` unconditionally (it just observed transport
    acceptance) while the operator confirm passes `require_stale=True`;
    revoke admits `pending` always and `sending` only once `claimed_at`
    is at least 15 minutes old (the documented operator recovery, now
    provably unable to race an active send) and refuses `sent`; edit
    admits only `pending` and clears the blocker exactly when a recipient
    is set; every zero-row outcome re-reads the row and maps to
    404/409/idempotent-200 truth rather than a generic failure.
- Boundary path/seam: _reject_blank
  - Replaced-path behaviors: no draft edit model existed; nothing rejected
    whitespace-only subject or body values.
  - Guard-relevant fields: `subject`, `body` on the draft edit request
    (bounded length at the field level; this validator adds the non-blank
    admission on top).
  - Caller x input shape: the office edit PATCH; a whitespace-only value
    rejects 422 before any CRM call, so a draft can never be edited into
    an unsendable blank snapshot.
- Boundary path/seam: _validate_recipient
  - Replaced-path behaviors: no recipient edit existed; the only recipient
    source was the enqueue-time latest-intake projection.
  - Guard-relevant fields: `recipient_email` on the draft edit request,
    validated against the same conservative pattern the public intake
    boundary uses (`_EMAIL_RE` in atlas_brain/api/leads.py), stripped
    before matching, bounded 3-254 characters at the field level -- the
    intake boundary's exact cap, so a corrected address can never clear
    the blocker only for the transport to reject it after the claim.
  - Caller x input shape: the office edit PATCH; an invalid shape rejects
    422 before any CRM call, and a valid value is what clears the
    `no_email` blocker downstream, so an office-corrected address can
    never be looser than an intake-submitted one.
- Boundary path/seam: atlas_brain/eom_api/funnel_store.py
  - Replaced-path behaviors: the shared datastore guard admitted the
    onboarding drafts table on column presence alone, so a canonical
    store carrying migration 360 without 361 was treated as ready and a
    signed-64 actor id would fail in Postgres only after the claim.
  - Guard-relevant fields: `approved_by_employee_id` attribute type
    (`atttypid = 'bigint'::regtype`) alongside the existing
    relation/column/role readiness predicates.
  - Caller x input shape: both startup guards (full app and slim profile)
    run the same readiness SQL; an INTEGER approver column now fails
    readiness closed with the existing controlled error, and the slim
    profile -- which only applies receivables migrations by design --
    refuses to serve draft routes until the canonical store has the
    widened column.
- Boundary path/seam: atlas_brain/services/eom_onboarding_drafts.py
  - Replaced-path behaviors: new module; previously nothing could send.
  - Guard-relevant fields: `settings.email.enabled` + `settings.email.api_key`
    transport preflight, the deterministic
    `eom-onboarding-draft:<draft_id>` idempotency key, Resend response
    status/error-name classification (2xx accepted, 409
    invalid_idempotent_request = already delivered, anything else =
    transport failure).
  - Caller x input shape: the funnel route is the only caller; preflight
    is reached before any claim so an unconfigured deployment rejects 503
    with zero state change; transport failure propagates 502 while the
    row stays `sending`; evidence writers run only after confirmed
    delivery and log-and-continue on their own failures.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `settings.email.enabled`
  (`ATLAS_EMAIL_ENABLED`) defaults to false and `settings.email.api_key`
  (`ATLAS_EMAIL_API_KEY`) defaults to unset, so a fresh deployment rejects
  approve-send 503 before claiming anything; the funnel API keeps its
  existing disabled-by-default token config, unchanged.
- Explicit value probe: unit tests exercise the enabled path through the
  injected sender seam, and the sender itself against a fake Resend client
  asserting that `settings.email.api_key` reaches the Authorization header
  alongside the Idempotency-Key header.
- Absent value probe: a dedicated unit test forces
  `settings.email.enabled` to false (with `settings.email.api_key` unset
  by default in the test environment) and proves 503 with zero claim
  calls and the draft row untouched in its pre-approval state.
- Default-session/default-context probe: route tests call the isolated
  router with explicit dependency overrides, so neither
  `settings.email.enabled` nor `settings.email.api_key` is read from any
  ambient session and no ambient operator context is trusted; the
  slim-profile subprocess test proves the module chain imports without
  `atlas_brain.config` loading at import time at all.
- Side-effect ordering: the `settings.email.enabled` and
  `settings.email.api_key` preflight precedes the claim; the claim
  precedes the transport call; delivery confirmation and both evidence
  writers are reached only after transport acceptance; a transport
  failure changes nothing except leaving the already-claimed row in
  `sending`; tests assert the call order and every failure branch.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_onboarding_drafts.py`
- `atlas_brain/storage/migrations/361_eom_onboarding_draft_actor_bigint.sql`
- `atlas_brain/storage/repositories/email.py`
- `plans/PR-EOM-Onboarding-Draft-Approval.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`

## Mechanism

The provider methods are the migration-360 state machine, statement for
statement. Claim is the header's exact guarded UPDATE: `pending ->
'sending'` with `claimed_at`/`approved_by_*` stamped, admitted only when
`blocker IS NULL AND recipient_email IS NOT NULL`, `RETURNING *` so the
winner carries the snapshot it must send. Zero updated rows are never a
generic error: the provider re-reads the row and returns the idempotent
already-sent replay, or raises the precise 404/409 (in-flight, revoked,
blocked, recipient-less) the office can act on. Confirm admits only
`sending`, with a `require_stale` split: the in-flow confirm the service
issues immediately after transport acceptance runs unguarded, while the
operator reconciliation route additionally demands the claim be at least
15 minutes old (`claimed_at <= NOW() - make_interval(mins => ...)`), so a
human cannot mark a just-claimed send as delivered while its worker may
still be mid-transport. Revoke admits `pending` plus *stale* `sending`
under the same threshold -- migration 360's documented operator recovery,
now gated so an active claim cannot be revoked out from under its sender
-- and refuses `sent` outright, so delivery evidence can never be
un-recorded; a fresh `sending` row answers both reconciliation actions
with the same in-flight 409. Migration 361 widens
`approved_by_employee_id` to BIGINT in one atomic, value-preserving ALTER
so the column matches the signed-64 actor ids the funnel auth boundary
already admits and the handoff table already stores. Edit admits only
`pending`, builds its SET clause from the provided fields, and clears
`blocker` exactly when a recipient is set, which is what makes a
`no_email` draft claimable at all. The list projection joins the contact
name, filters by a status allowlist, and pages by the same
`(created_at, id)` keyset the lead review queue uses.

The service executes the protocol around those methods. Approve first
runs the transport preflight -- `settings.email.enabled` and the Resend
API key -- and rejects 503 BEFORE claiming, because a claim that the
deployment cannot send would manufacture a wedged `sending` row out of
pure configuration. It then claims, sends outside any transaction, and
confirms. The sender is a direct Resend POST over httpx with
`Idempotency-Key: eom-onboarding-draft:<draft_id>`: the key is derived
purely from the draft id so any retry of the same draft reuses it, and
Resend deduplicates identical keys server-side for 24 hours. A 409
invalid_idempotent_request response is positive proof the original send
was accepted, so the service confirms delivery and flags the transport
replay instead of failing -- the same semantics
`content_ops_deflection_delivery.py` proved for paid report delivery. Any
other transport failure leaves the row in `sending` and returns 502
naming reconciliation: the send outcome is unknown, so neither a silent
retry nor a rollback to `pending` is safe, and the row is the operator's
evidence. Deflection spends its 15-minute threshold on auto-reclaim; here
the same threshold instead gates when the operator reconciliation actions
admit the row at all, keeping migration 360's operator-in-the-loop
recovery while closing the fresh-claim race. The direct
sender exists because the generic email port offers no idempotency key
and its send() imports the `atlas_brain.tools` registry, whose
dependencies (`dateparser` et al.) are deliberately absent from the slim
EOM Render profile; `httpx` is already a slim-profile dependency, and the
module defers its `atlas_brain.config` import into the functions that
need it so the slim profile's import-isolation contract holds.

After confirmed delivery the service writes secondary evidence: a
sent_emails history row (`business_context_id='effingham_maids'`,
`template_type='onboarding_welcome'`, the transport message id, contact
and draft ids in metadata) and a CRM interaction on the contact. Both
mirror the intake acknowledgement's ordering rule -- delivery already
succeeded, so history failures log a warning and never flip the outcome.
Revoke and confirm-sent record the acting employee the same way, giving
reconciliation actions provenance without a schema change.

Codex round 2 tightened three admissions and one binding. The claim now
carries the booking family's contact-activity predicate (`EXISTS` on an
active `effingham_maids` contact), so archiving a lead after its draft
was enqueued blocks approval with a precise 409 while the queue still
lists the draft for the office to revoke. The shared funnel datastore
guard requires the approver column to be BIGINT, closing the deployment
window where a canonical store carries migration 360 without 361: the
slim profile applies only receivables migrations by design, so the guard
-- not a migration-list change -- is the seam that keeps it fail-closed.
The default evidence writers bind to the CRM provider's own pool: the
provider exposes a public `pool` property, `EmailRepository` gains the
same pool-override shape the provider already uses, and
`log_interaction` writes through the provider pool, so in the split-DB
Render shape (funnel provider on `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`,
global pool elsewhere or uninitialized) the sent_emails history and CRM
interaction land in the store that owns the draft instead of being
silently swallowed or misfiled. And the recipient edit cap drops from
320 to the intake boundary's 254.

The routes are verbatim siblings of the existing funnel surface: same
bearer + actor dependencies, same closed camelCase projections, same
201-fresh / 200-idempotent convention, same domain-error-to-HTTPException
mapping. The draft-action routes intentionally take no Idempotency-Key
header: booking and handoff use that header to CREATE their operation
identity, but a draft's identity already exists, and the status machine
makes every action either a single winner or an explicit idempotent
replay. The recipient edit validates against the same conservative email
pattern the public intake boundary uses, so an office-corrected address
can never be looser than an intake-submitted one.

## Intentional

- One migration only (361): an atomic, value-preserving INT -> BIGINT
  widening of the draft approver id, aligning it with the funnel actor
  boundary and the handoff table; everything else executes A2's schema as
  shipped.
- Stuck `sending` rows require explicit operator action (confirm-sent or
  revoke). Auto-reclaim was considered and rejected: migration 360 records
  the operator-in-the-loop policy, and the deterministic transport key
  already makes an operator-driven retry safe without background
  machinery.
- The direct Resend sender does not extend the generic email provider
  port. Threading an idempotency key through the port would touch every
  adapter (Gmail cannot honor it) and still leave the slim-profile tools
  registry import; the repo's established pattern for idempotent delivery
  paths (deflection delivery, campaign sender) is a direct sender, and
  the EOM funnel already forces Resend for its intake acknowledgement.
- Revoke from `sending` does not verify the transport log itself; that
  verification is the operator's step (query Resend by the draft-id key).
  The confirm path exists precisely so a verified send is recorded as
  sent rather than revoked.
- `revoked_by`/`confirmed_by` columns were not added; the CRM interaction
  note carries reconciliation provenance without schema churn. If A4+
  needs queryable provenance, that is a one-column additive migration
  then.
- Migration 361 is gated by the funnel datastore guard, not by enrolling
  it in the slim profile's startup migration set: that set is closed over
  the receivables readiness contract by design, and the guard is the
  existing seam that keeps the funnel fail-closed until the canonical
  store (migrated by the main brain) is ready.
- The queue list still shows drafts whose contact has been archived:
  visibility is what lets the office revoke them, and only the claim --
  the decision to send -- carries the contact-activity admission.
- Only `log_interaction` moves to the provider-pool seam in this slice;
  it is the one provider method on the draft evidence path. The remaining
  legacy `DatabaseCRMProvider` methods that still read the global pool
  directly are named in Deferred.

## Deferred

- A4: the tokenized customer-facing link.
- Tracker UI wiring for the approval queue screen (W-lane).
- Draft body re-rendering from an updated template (edits are per-draft;
  template changes are code changes).
- Sent-draft resend/duplicate flows (a sent draft is terminal; a repeat
  email would be a new product decision).
- Automated stuck-`sending` alerting (the queue list exposes the state;
  alerting is hardening once real volume exists).
- Estimate/first-clean reschedule-cancel lifecycle (unchanged from A1/A2
  deferrals).
- Migrating the remaining `DatabaseCRMProvider` methods that still call
  `get_db_pool()` directly (contact CRUD/search/list, service tickets,
  `get_interactions`, appointment operations) onto `self._get_pool()`.
  None of them is on a funnel route today; when one is wired into a
  funnel path it must move to the provider-pool seam in that slice.

Parking predicate: hardening narrower than one draft approval request, or
requiring a new table/subsystem/dependency, is parked unless it can send
email without office approval, double-send, or record unsent email as
sent.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py atlas_brain/services/eom_onboarding_drafts.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_render_profile.py` -- passed.
- ASCII scan of every touched Python file -- no non-ASCII bytes.
- `python -m pytest tests/test_eom_lead_conversion.py tests/test_migrations_runner.py -q` -- 185 passed, 1 skipped (including the migration-361 shape test, the fresh-claim in-flight 409 coverage, the archived-contact approval refusal, the 255-character recipient rejection, and the default-history provider-pool binding proof); 3 pre-existing torch-import failures reproduced identically on the unmodified base.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://postgres@localhost:5433/atlas_migration_tests python -m pytest tests/test_eom_lead_conversion_integration.py -q` -- 40 passed against disposable Postgres 16 (migrations through 361 plus sent_emails history 016/349), including the approval pipeline from first-clean booking through edited approved send with idempotent replay and the sent_emails + interaction evidence landing in the provider's own store while the global pool stays uninitialized, the two-session single-winner provider claim, blocker resolution through edit, stuck-`sending` reconciliation that first proves fresh claims answer confirm-sent and revoke with the in-flight 409 and only backdated (20-minute-old) claims are admitted, the revoked-while-sending zombie-confirm regression, the archived-contact claim refusal with restore-then-claim, the datastore guard failing closed on an INTEGER approver column, and the list projection; 3 pre-existing torch-import failures.
- `python3 scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json` -- ratchet gate passed (send evidence is exercised through the injected history seam, not by patching the repository module).
- `python -m pytest tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package tests/test_eom_render_profile.py::test_shared_eom_funnel_datastore_guard_keeps_missing_relations_in_verdict -q` -- 2 passed; the slim profile exposes all five draft routes with its import-isolation contract intact.
- `python -m pytest -q tests/test_audit_plan_doc.py tests/test_audit_plan_code_consistency.py tests/test_audit_pr_plan_presence.py tests/test_check_diff_budget.py` -- 103 passed.
- `python scripts/check_boundary_change_enumeration.py --base origin/main --strict` -- OK.
- `python scripts/check_deployed_config_probing.py --base origin/main --strict` -- OK.
- `bash scripts/local_pr_review.sh --current-pr-body-file <pr-body-file> --pr-author canfieldjuan origin/main` -- local PR review passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 4 |
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/eom_api/funnel.py` | 303 |
| `atlas_brain/eom_api/funnel_store.py` | 15 |
| `atlas_brain/services/crm_provider.py` | 350 |
| `atlas_brain/services/eom_onboarding_drafts.py` | 263 |
| `atlas_brain/storage/migrations/361_eom_onboarding_draft_actor_bigint.sql` | 23 |
| `atlas_brain/storage/repositories/email.py` | 34 |
| `plans/PR-EOM-Onboarding-Draft-Approval.md` | 665 |
| `tests/test_eom_lead_conversion.py` | 870 |
| `tests/test_eom_lead_conversion_integration.py` | 491 |
| `tests/test_eom_render_profile.py` | 5 |
| `tests/test_migrations_runner.py` | 27 |
| **Total** | **3052** |

## Cold diff reconstruction

Gaps first: no contract gaps found in the current diff.

Change-by-change reconstruction against the contract:

- The provider gains the six draft state-machine methods, each a thin
  guarded statement with zero-row outcomes mapped to precise truth
  (idempotent replay, 404, or an actionable 409); the claim is migration
  360's exact statement with the readiness predicate inline and actor
  stamping. Citation: `atlas_brain/services/crm_provider.py:2372`.
- The service module orchestrates preflight -> claim -> send -> confirm,
  carries the injectable sender seam, derives the deterministic transport
  key, classifies Resend's 409 as delivered, keeps transport failures in
  `sending` with a 502, and writes post-acceptance evidence that never
  flips the outcome; `atlas_brain.config` is imported lazily to preserve
  the slim profile's import isolation. Citation:
  `atlas_brain/services/eom_onboarding_drafts.py:1`.
- The funnel router gains the closed edit-request and projection models,
  the recipient pattern shared with intake, the sender test seam, and the
  five routes on the existing auth stack with the 201/200 convention;
  revoke and confirm-sent log actor provenance through the CRM
  interaction writer. Citation: `atlas_brain/eom_api/funnel.py:132`,
  `atlas_brain/eom_api/funnel.py:441`.

Codex round 1 (three fixes in this diff; the fourth finding is waived in
the PR body):

- Reconciliation staleness gate: `_EOM_ONBOARDING_SENDING_STALE_AFTER_MINUTES = 15`
  is the single threshold; revoke admits `sending` only when
  `claimed_at <= NOW() - make_interval(mins => $2)`, and confirm gains the
  `require_stale` split -- the operator route passes True, the in-flow
  service confirm after transport acceptance passes False. A fresh claim
  answers both operator actions with the in-flight 409. Citation:
  `atlas_brain/services/crm_provider.py:37`,
  `atlas_brain/services/crm_provider.py:2581`,
  `atlas_brain/services/crm_provider.py:2638`,
  `atlas_brain/eom_api/funnel.py:611`.
- Migration 361 atomically widens `approved_by_employee_id` from INTEGER
  to BIGINT (value-preserving, no CONCURRENTLY, no DROP), matching the
  signed-64 funnel actor boundary and the migration-353 handoff column;
  the shape test pins the marker, the ALTER, and the rollback evidence.
  Citation: `atlas_brain/storage/migrations/361_eom_onboarding_draft_actor_bigint.sql:1`,
  `tests/test_migrations_runner.py:403`.
- CI-lane enrollment: the new service module is added to the invoicing
  workflow's path filters (that lane runs the slim-profile guard in
  `tests/test_eom_render_profile.py`) alongside the EOM lead-pipeline
  lane, and migration 361 joins the lead-pipeline filters, so edits to
  either surface re-run the guards that own them. Citation:
  `.github/workflows/atlas_invoicing_checks.yml:27`,
  `.github/workflows/atlas_eom_lead_pipeline_checks.yml:58`.

Codex round 2 (four fixes in this diff):

- The claim carries the booking family's contact-activity admission: an
  `EXISTS` predicate on an active `effingham_maids` contact inside the
  guarded UPDATE, with the zero-row re-read mapping the archived-contact
  case to its own 409 after blocker and recipient truth. Citation:
  `atlas_brain/services/crm_provider.py:2553`,
  `tests/test_eom_lead_conversion_integration.py:3319`.
- The shared funnel datastore guard requires
  `approved_by_employee_id` to be `bigint`, so a canonical store with
  migration 360 but not 361 fails readiness closed instead of failing a
  valid signed-64 approval after the claim; proven both directions
  (INTEGER refuses, BIGINT admits) against real Postgres. Citation:
  `atlas_brain/eom_api/funnel_store.py:89`,
  `tests/test_eom_lead_conversion_integration.py:1948`.
- The default evidence writers bind to the CRM provider's own pool: the
  provider exposes a public `pool` property, `EmailRepository` gains the
  provider's pool-override shape, the service builds the default history
  writer from `crm.pool`, and `log_interaction` writes through
  `self._get_pool()`; the real-Postgres pipeline proof now asserts the
  sent_emails row and interaction land in the provider's store while the
  global pool stays uninitialized. Citation:
  `atlas_brain/services/crm_provider.py:337`,
  `atlas_brain/services/crm_provider.py:2896`,
  `atlas_brain/storage/repositories/email.py:27`,
  `atlas_brain/services/eom_onboarding_drafts.py:171`.
- The recipient edit cap drops from 320 to the intake boundary's 254
  characters, so an office correction can never clear the `no_email`
  blocker only for the transport to reject the over-long address after
  the claim. Citation: `atlas_brain/eom_api/funnel.py:144`.
- Unit tests extend the `_CRM` fake with an in-memory mirror of the
  provider state machine and cover the route projections, edit admission
  and validation, the full approve failure matrix (blocked,
  recipient-less, in-flight, revoked, replay, transport failure, Resend
  replay, unconfigured transport), revoke/confirm reconciliation with
  provenance, HTTP guard ordering, and the raw sender's header and
  response mapping. Citation: `tests/test_eom_lead_conversion.py:2481`.
- Integration tests prove the same protocol against real Postgres: the
  end-to-end pipeline with edit and idempotent replay, the two-session
  single-winner claim, blocker resolution, stuck-`sending` recovery by
  confirm and by revoke including the revoked-while-sending race, and the
  list projection with pagination and status filters. Citation:
  `tests/test_eom_lead_conversion_integration.py:3103`.
- The slim Render profile test asserts all five routes are exposed and
  the import-isolation ledger is unchanged. Citation:
  `tests/test_eom_render_profile.py:227`.
- The EOM lead pipeline workflow filters gain the new service path.
  Citation: `.github/workflows/atlas_eom_lead_pipeline_checks.yml:28`.

Scope check:

- Everything changed traces to the contract: provider state machine,
  send orchestration and transport, funnel routes, evidence writers,
  tests, and workflow filters.
- Everything the contract required appears in the diff: queue list,
  pending edit with blocker clearing, two-phase approve-send with
  deterministic transport idempotency, single-winner claim, refusal
  matrix, stuck-`sending` operator recovery, revoked-while-sending
  truth, evidence writers, slim-profile exposure, and regression tests.
- No declared out-of-scope module moved: no migration, no email provider
  port change, no A2 enqueue/booking/handoff change, no public route, no
  tracker UI, no auto-retry machinery.
