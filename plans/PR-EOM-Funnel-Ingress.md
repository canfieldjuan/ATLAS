# PR-EOM-Funnel-Ingress

## Why this slice exists

Issue #2188 is the umbrella only; its onboarding-email and first-clean flow is
superseded by [Juan's 2026-07-26 office-controlled conversion decision](https://github.com/canfieldjuan/ATLAS/issues/2188#issuecomment-5085162036). This
slice implements the prerequisite from the accepted EOM funnel plan: every new
EOM web, call, or SMS identity remains a lead until a later Juan-only conversion
action. It also persists ad-attribution input at the only point where it exists.

Diff-budget override: this is one indivisible inbound-lead boundary. Its shared
resolver, every pre-existing ingress adapter, database lifecycle guard and
ledger, public attribution payload, CI enrollment, and regression proof must
land together: splitting any one of them would leave an admitted ingress able
to create, match, or mutate contacts under a different lifecycle rule.

### Problem-derived contract

- Root cause: EOM's direct form, Web3Forms relay, call, SMS, and legacy
  estimate paths do not share one concurrency-safe, active-only, phone-first
  lead boundary. The first atomic-resolver implementation also replaced the
  prior `find_or_create_contact` path without enumerating its behavioral
  contract, so idempotent relay delivery, same-type contact merges, and the
  post-create reasoning event were omitted. Generic CRM updates can also mutate
  a claimable legacy row before it enters EOM, and ingress can run even if the
  new lifecycle ledger migration did not install. This can create or mutate a
  customer from untrusted inbound data, attach an inbound lead to an archived or
  wrong identity, duplicate a new identity under concurrent delivery, or leave
  lead evidence incomplete. The website also does not send per-lead ad
  identifiers to Atlas. Current-head review also found that this same admission
  seam treated any caller `source_ref` as a unique replay key, that attribution
  dedupe changed legacy empty-key bytes and lowercased opaque identifiers, and
  that the lifecycle ledger's row trigger did not defend against table
  truncation. The current head also resolves an inbound Gmail delivery by its
  relay identity but logs its interaction without that identity, so a retry on
  a later day can duplicate the interaction; and EOM call/SMS adapters prefer
  a partial extracted phone over a usable transport caller number, so a valid
  inbound lead can now be rejected. The MCP update guard also checks only an
  explicitly supplied tenant, so a default-EOM lead-stage request can claim a
  NULL-context legacy row before its lifecycle transition is rejected. The
  lower generic CRM provider also allows a caller to reassign an EOM row's
  `business_context_id` without entering its lifecycle guard, so sequential
  generic updates can attempt to leave EOM before changing lifecycle state.
  Finally, an EOM appointment has a durable database ID but does not pass that
  ID as the resolver's trusted event identity; a local-only phone and no email
  therefore leaves the calendar booking confirmed but drops its CRM lead. The
  resolver's active-only match query also reads without a row lock, so a
  concurrent soft archive can commit after the active predicate is evaluated
  but before the resolver returns that row. Separately, inbound SMS carries a
  provider `MessageSid`, but the background pipeline currently receives only
  its optional local SMS row ID; a persistence failure loses the durable ID and
  makes the interaction fall back to an unanchored daily dedupe key on retry.
  The current review loop exposed the shared execution seam behind those
  examples: contact resolution, generic ownership validation, and inbound
  interaction logging are independently committed. A selected row can be
  archived after resolver commit but before the later interaction insert, a
  generic ownership update can validate a stale NULL owner and then overwrite a
  concurrent EOM claim, and an identity-bearing relay retry bypasses its trusted
  event key because that key is consulted only for name-only deliveries. These
  are one transaction-boundary decision, not four unrelated missing guards.
- Correct fix must touch/change: centralize EOM inbound identity resolution so
  unmatched non-spam web/relay/calls/SMS/legacy estimate bookings become
  `lead/new` and any matching contact is read-only; resolve only active contacts
  with phone priority across EOM and claimable-legacy populations; serialize
  asserted inbound identities before lookup/insert, including a stable relay
  event identity when no phone or email is present; preserve idempotent
  same-type generic merges while blocking real EOM lifecycle transitions; emit
  the established contact-created reasoning event only after a successful atomic
  insert and without making that secondary delivery fatal; prevent generic
  lifecycle writes from bypassing the rule before a row is claimable into EOM;
  require the lifecycle table and trigger before atomic ingress inserts; add a
  durable lifecycle creation record; extend the public intake payload and
  interaction metadata for attribution without discarding a changed snapshot;
  and cover the real public route/core call paths plus database behavior with
  focused tests. A replay key must be explicit trusted relay-event evidence,
  not caller metadata; direct public intake without email or a full phone must
  be rejected before CRM/email side effects. Attribution must retain the exact
  legacy no-attribution key and opaque value bytes. Ledger immutability must
  include `TRUNCATE`. Every inbound delivery with a stable Gmail, call, or SMS
  event identifier must pass it to the interaction's recognized anchor metadata;
  adapters must use a full extracted phone when present, otherwise a full
  authoritative transport number, before EOM admission. A default effective
  EOM context must reject an MCP lead-stage mutation before the legacy
  claim-on-write step. Generic provider updates must reject an actual ownership
  transition into or out of EOM before their SQL write, leaving the dedicated
  claim operation as the only way to acquire EOM ownership. Once an EOM
  appointment is durably recorded, its namespaced appointment ID must anchor an
  otherwise identityless resolver call without treating a partial phone as a
  matching identity. The atomic resolver must lock each selected active match
  until its transaction completes, so a competing archive either waits for
  that resolution or commits first and forces the resolver to create an active
  replacement. The inbound-SMS webhook must carry `MessageSid` independently
  of optional persistence through the background pipeline, resolver provenance,
  and `crm_event_id` interaction anchor; only callers without a provider ID may
  retain the legacy local-row fallback. The repair must make an EOM inbound
  delivery that writes a CRM interaction one database command: under a
  transaction-scoped, sorted set of asserted identity and trusted-delivery
  advisory locks, resolve/create the contact, lock the selected active row, and
  insert/dedupe the interaction before commit. Every explicit trusted delivery
  ID must be checked and persisted as an EOM delivery receipt keyed by
  `(source, delivery_id)` before channel lookup, whether or not phone/email is
  also present. The receipt must be independent of `contacts.source_ref`, which
  cannot represent more than one delivery for a matched contact. A replay of
  that receipt must not create a new interaction against an archived original
  row. Generic ownership validation must lock the target row and
  perform its permitted update in that same transaction, so its decision is
  based on the owner that it writes. The EOM call/SMS and Gmail adapters must
  use the combined command; the scheduler has no inbound CRM interaction and
  remains resolver-only. Finally, non-EOM SMS contact provenance must retain
  its established local-row ID whenever one exists rather than replacing it
  with a provider ID.
- Must not change: existing customer identity, non-EOM CRM behavior, the
  current public intake response/CORS/email acknowledgement semantics, Google
  Calendar booking behavior, Customer/Site onboarding, jobs, payments,
  first-clean/card-on-file work, and adjacent PRs #2195/#2200. Existing
  idempotent generic enrichment of an EOM contact whose type/stage is already
  unchanged must continue to work. Full-phone-only form intake and
  email-backed partial-phone form intake retain their current behavior. Existing
  interaction type/intent mapping, non-EOM call/SMS phone selection, and
  ordinary daily interaction dedupe without a stable inbound event remain
  unchanged. Default-scoped non-stage MCP edits retain their existing
  claim-on-write behavior. Non-EOM ownership reassignment, EOM same-context
  updates, appointment/calendar creation, direct-form identity admission, and
  non-blocking CRM-link failure behavior remain unchanged. This repair must not
  make a generic caller-provided `source_ref` a trusted replay key, change the
  public intake response or acknowledgement behavior, alter non-EOM call/SMS
  phone selection, or change a scheduler booking into an inbound-interaction
  command.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 22

1. Make all new EOM web/Web3Forms relay/call/SMS/legacy estimate-booking inbound identities `lead/new`, preserve
   matching contacts unchanged, and block EOM lead-stage edits through the
   generic CRM MCP surface.
2. Add the immutable EOM lifecycle-event schema for inbound lead creation and
   extend public lead intake metadata with UTM/click-ID/landing/referrer input.
3. Add focused unit/route coverage for source normalization, preservation,
   attribution persistence, and the protected generic stage path.
4. Repair the current-head CI and review findings only where they are required
   to make this inbound boundary true: active-only, phone-first resolution;
   claimable-legacy lifecycle protection; lifecycle-migration readiness; and
   exact CI enrollment/baseline evidence.
5. Close the atomic-resolver decision seam as one fix: preserve the predecessor
   path's relay idempotency, actual-transition-only lifecycle guard, and
   post-create event behavior, with the schema-isolated sent-email route test
   applying the lifecycle prerequisite it now exercises.
6. Make identityless admission provenance-safe: only a trusted relay event ID
   may become a replay key; a direct form with no email and fewer than ten phone
   digits is rejected before CRM or acknowledgement side effects.
7. Preserve interaction-dedupe compatibility and evidence fidelity by using the
   exact legacy key basis when attribution is absent, retaining opaque value
   case when it is present, and rejecting lifecycle-ledger `TRUNCATE` in the
   same immutable policy as row mutation.
8. Carry Gmail, call, and SMS delivery identifiers into recognized interaction
   anchor metadata so a retry does not depend on the UTC-day fallback key.
9. For EOM call/SMS only, prefer a full extracted phone; when extraction is
   partial, fall back to a full authoritative transport caller number before
   shared lead admission.
10. Reject a default-EOM MCP lead-stage request before any NULL-context legacy
    claim, while preserving claim-on-write for non-stage edits.
11. Close the generic ownership-bypass and internal-booking identity seams:
    reject actual provider ownership transitions into or out of EOM before
    writing, and use a persisted EOM appointment's namespaced ID as a trusted
    replay identity only when full phone/email identity is absent.
12. Close the selected-contact archive race by holding a row lock for each
    active atomic-resolver match until the resolution transaction commits; an
    archive that wins first must cause normal unmatched-lead creation instead.
13. Preserve the provider `MessageSid` through the inbound-SMS background and
    fallback paths even if no local SMS row exists, and use it as the EOM
    resolver provenance and recognized interaction anchor with the local ID
    only as backward-compatible fallback.
14. Close the execution seam rather than adding another selected-row guard:
    every EOM inbound path that records an interaction must resolve/create its
    contact and insert/dedupe that interaction under one PostgreSQL transaction;
    a replayed archived trusted delivery is a no-op interaction replay, not a
    replacement contact or a new interaction on the archived row.
15. Treat every explicitly supplied trusted delivery ID as an EOM replay key,
    regardless of channel identity: take its advisory lock, check and persist a
    dedicated globally unique `(source, delivery_id)` receipt before phone/email
    lookup, and retain that receipt for future replays. EOM call/SMS adapters
    must pass their provider IDs explicitly; generic `source_ref` remains
    evidence, not authority.
16. Serialize generic EOM ownership validation with its SQL update by locking
    the target contact inside one transaction; preserve the dedicated
    compare-and-set claim operation as the only NULL-to-EOM acquisition path.
17. Restore non-EOM SMS `source_ref` to the local SMS UUID when it exists;
    retain provider-ID provenance only for EOM or when no local row exists.

### Review Contract

- Acceptance criteria:
  1. The public `POST /api/v1/leads/intake` route still returns its existing
     success envelope while persisting the supplied attribution snapshot in the
     `web_form` interaction; route tests settle the behavior.
  2. A new EOM call/SMS identity is created with `contact_type=lead` and
     `lead_stage=new`; a matching EOM lead or customer is returned unchanged;
     focused ingress tests settle both branches.
  3. An EOM lead cannot be retyped through the generic CRM merge or have its
     stage changed through generic MCP `update_contact`; provider/MCP tests
     settle those boundaries.
  4. Migration 351 creates an append-only lifecycle event record for new EOM
     leads, and migration 352 creates a globally unique trusted-delivery
     receipt in the same database transaction as the resolved contact and any
     inbound interaction; real PostgreSQL tests settle creation, replay, and
     immutability where applicable.
  5. Existing public CORS, acknowledgement-send, and non-EOM CRM behavior are
     covered by the existing intake/provider tests.
  6. Atomic EOM ingress fails before a contact write if migration 351's ledger
     table/trigger or migration 352's receipt table is absent; real-provider
     coverage settles each prerequisite.
  7. Resolution never returns an archived contact and prefers a phone match
     over an email match across EOM and claimable-legacy populations;
     real-PostgreSQL cases settle both properties.
  8. A generic caller `source_ref` cannot admit an identityless lead: direct
     public intake without email or a ten-digit phone fails before CRM/email
     side effects, while an explicit trusted relay-event key is replay-safe
     under concurrent delivery.
  9. An interaction with no attribution retains its predecessor dedupe key;
     opaque attribution values remain case-sensitive; and migration 351
     rejects ledger `TRUNCATE` in PostgreSQL.
  10. Reprocessing the same Gmail, call, or SMS delivery supplies the same
      recognized interaction anchor metadata, independent of its processing
      date; adapter tests settle the boundary.
  11. An EOM call or SMS with a partial extracted number and a full transport
      caller number links through the full transport number; a full extracted
      number remains preferred and non-EOM selection is untouched.
  12. With a default EOM context and no explicit tenant argument, a legacy
      lead-stage update returns the funnel-transition error without calling
      `claim_contact` or `update_contact`; a non-stage default-scoped edit
      remains claimable.
  13. A generic EOM ownership reassignment fails before the update, so the
      following attempted lifecycle write remains protected; non-EOM ownership
      reassignment and EOM same-context enrichment remain permitted.
  14. A persisted EOM appointment with only a partial phone and no email uses
      its namespaced appointment ID as the resolver replay anchor, creates or
      returns a `lead/new` contact, and still confirms the booking if the CRM
      link fails.
  15. A concurrent archive cannot commit between an active atomic-resolver
      selection and that resolver's transaction completion: the real-PostgreSQL
      interleaving holds the selected row lock, then releases the archive only
      after the resolver's transaction completes its active-match decision.
  16. An inbound SMS that reaches processing without a local SMS row still
      passes its Twilio `MessageSid` from the webhook to the EOM resolver and
      `crm_event_id` interaction metadata; a retry therefore retains the same
      dedupe anchor. Direct callers with no provider ID retain their local-ID
      anchor fallback.
  17. The documented PostgreSQL execution model holds for every admitted
      resolver/archive/replay/ownership interleaving: a combined EOM inbound
      command commits a selected active contact and its interaction together;
      an archive before its lock produces a replacement, an archive after its
      lock waits until that committed interaction, and a replay receipt found
      on an archived original performs no new interaction write. The real
      PostgreSQL integration tests exercise representative orderings without
      defining the model by enumeration.
  18. A generic ownership update and `claim_contact` serialize on the same row:
      either the generic update commits while the row is non-EOM and the claim
      fails its compare-and-set, or the claim commits first and the generic
      update rejects EOM reassignment. Focused real-PostgreSQL tests prove both
      orders and same-context/non-EOM permitted cases.
  19. EOM and non-EOM SMS provenance diverge only at the declared boundary:
      EOM uses the provider MessageSid when supplied, while a non-EOM local SMS
      row continues to send its UUID to `find_or_create_contact`; focused
      adapter tests settle both outcomes.
- Reachability proof: FastAPI `POST /api/v1/leads/intake` is exercised via
  TestClient and the injectable intake core; call/SMS use their real
  `_link_to_crm` paths with only CRM/transport boundaries faked.
- Affected surfaces: `atlas_brain/api/leads.py`, CRM provider/MCP writes,
  call/SMS intelligence, public website intake contract, and migration startup.
- Risk areas: tenant scoping, untrusted inbound identity data, accidental
  customer demotion/promotion, stage bypass, migration idempotency, and public
  form compatibility.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/comms/webhooks.py`
- `atlas_brain/api/leads.py`
- `atlas_brain/autonomous/tasks/gmail_digest.py`
- `atlas_brain/comms/call_intelligence.py`
- `atlas_brain/comms/sms_intelligence.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_lead_ingress.py`
- `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql`
- `atlas_brain/storage/migrations/352_eom_inbound_delivery_receipts.sql`
- `atlas_brain/tools/scheduling.py`
- `plans/PR-EOM-Funnel-Ingress.md`
- `tests/maturity_sweep/baseline_atlas_brain_mcp.json`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/maturity_sweep/baseline_atlas_brain_tools.json`
- `tests/test_crm_read_scoping.py`
- `tests/test_eom_lead_ingress.py`
- `tests/test_eom_lead_pipeline_integration.py`
- `tests/test_eom_sent_email_tenant_scope.py`
- `tests/test_leads_intake.py`
- `tests/test_tenant_stamping.py`

## Mechanism

`eom_lead_ingress` becomes the single inbound resolver for direct web intake,
the Web3Forms relay, call, SMS (including the webhook fallback), and legacy
estimate booking. It looks up the same EOM/claimable-legacy populations as
public intake, returns matching contacts untouched, and creates only unmatched
identities as `lead/new`. The database provider serializes each asserted email
or phone identity under transaction-scoped advisory locks before its
exact lookup/insert. Generic provider type/stage writes reject EOM leads, so
MCP and approved call-plan writes cannot become a lifecycle back door.

The atomic resolver checks the migration-351 lifecycle ledger and the
migration-352 delivery-receipt table inside the transaction before any identity
lookup or write. If either is absent, that inbound path fails closed rather than
creating an unledgered or unreplayable EOM lead. Its
lookup locks an active selected row until transaction completion, admits only
active rows, and resolves phone across both eligible populations before
attempting email. Thus a soft archive serializes either before selection (which
creates an active replacement) or after the resolver has committed its
active-match decision; it cannot commit between predicate evaluation and that
transaction completion.

Migration 351 adds `eom_lead_lifecycle_events` and an EOM-only contact trigger:
a new `lead/new` contact records one immutable `lead_created` event in its
insert transaction. The trigger records system actor/source/operation metadata;
the later booking/conversion slice will add human-authored transition events.
Migration 352 adds the distinct, globally unique inbound-delivery receipt that
records every trusted EOM `source`/delivery-ID pair with its resolved contact
and optional interaction. That table is intentionally not an alternative
lifecycle history: it is the replay identity the contact provenance column
cannot hold.

The public request accepts bounded attribution fields and writes a non-empty
snapshot only to the intake interaction metadata. The interaction identity
includes that snapshot, so a later submission with a different click or UTM
value is retained. It does not change contact identity or the response envelope.
Generic lifecycle edits reject EOM rows and claimable-legacy rows, and generic
ownership writes reject actual transitions into or out of EOM. The later
constrained transition service is therefore the only stage/type writer, while
the existing dedicated claim path remains the only ownership-acquisition path.

The trusted Web3Forms relay, call, SMS, and persisted appointment paths pass a
separate replay-event key when they possess one; a generic `source_ref` is
metadata and may be constant, so it cannot attest a distinct identity. The
receipt table, rather than that one contact column, maps each trusted delivery
to its resolved contact and optional interaction. Direct form intake therefore
requires email or a ten-digit phone.
Interaction dedupe keeps the predecessor byte basis for no-attribution records
and appends an exact submitted attribution snapshot only when present. The
ledger's immutable policy covers table truncation as well as row mutation.

Every inbound Gmail, call, and SMS delivery that has a stable provider event ID
records a receipt under that ID and writes the same ID into recognized
interaction-anchor metadata, so contact replay and interaction replay have the
same durability boundary. The SMS webhook
carries `MessageSid` separately from its optional local row through both the
intelligence and fallback adapters; the local row remains a compatibility
fallback only when the provider did not supply an ID. For EOM call/SMS,
phone extraction is enrichment: a full extracted number wins, a partial one
falls back to the full transport caller number, and neither rule changes
non-EOM caller behavior.

### Cold diff reconstruction: current-head repair

- `atlas_brain/api/leads.py` preserves the existing seven-digit validation for
  email-backed intake but rejects a no-email, sub-ten-digit phone before CRM or
  acknowledgement side effects.
- `atlas_brain/autonomous/tasks/gmail_digest.py` is the sole admitted
  name-only relay adapter: it passes its Web3Forms message ID through the
  separate `relay_event_id` argument while retaining that same ID as the
  stored `source_ref` provenance value.
- `atlas_brain/services/eom_lead_ingress.py` and
  `atlas_brain/services/crm_provider.py` use an explicit trusted delivery ID,
  not generic caller `source_ref`, for identityless admission and delivery
  replay. `crm_provider.py` locks and records each trusted ID in migration
  352's receipt ledger before it consults phone/email identity, leaving
  identity-bearing contact provenance unchanged.
- `atlas_brain/services/crm_provider.py` retains the predecessor interaction
  dedupe-byte basis when attribution is absent, adds attribution only when it
  exists, preserves opaque submitted values, and moves its existing postcommit
  reasoning emission behind an overrideable provider seam so its committed-row
  and non-fatal behavior can be proven without a direct producer mock.
- `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql` adds the
  statement-level `BEFORE TRUNCATE` ledger guard alongside the existing row
  mutation guard.
- `atlas_brain/autonomous/tasks/gmail_digest.py` writes each stable Gmail relay
  ID to `gmail_message_id` interaction metadata. `atlas_brain/comms/call_intelligence.py`
  and `atlas_brain/comms/sms_intelligence.py` write their stable provider IDs
  to `crm_event_id`, so all three adapters select the established anchor-key
  dedupe branch on retry.
- `atlas_brain/services/eom_lead_ingress.py` centralizes EOM-only full-phone
  preference; call/SMS use a complete extracted number when available and
  otherwise a complete transport number, while their non-EOM branches retain
  the prior extracted-or-transport selection.
- `atlas_brain/mcp/crm_server.py` derives the effective default business
  context before its EOM lead-stage guard. A default-EOM stage request on a
  NULL-context legacy lead now returns the transition-service error before
  `_claim_if_legacy`; an ordinary non-stage edit still reaches that existing
  claim-on-write path.
- `atlas_brain/services/crm_provider.py` fetches the existing ownership for a
  generic context update as well as lifecycle updates, and rejects an actual
  transition whose source or target is EOM before generating the SQL `UPDATE`.
  It leaves same-context EOM enrichment and non-EOM reassignment on their
  existing update paths.
- `atlas_brain/tools/scheduling.py` passes `appointment:<persisted-id>` as the
  separate trusted replay anchor only after the EOM appointment row exists.
  The shared resolver consequently creates a `lead/new` booking contact from a
  local-only phone without using that fragment as a matching identity; the
  existing catch still confirms a booking when CRM linking fails.
- `atlas_brain/services/crm_provider.py` takes `FOR UPDATE` locks on active
  EOM and claimable-legacy identity matches before completing its resolution
  transaction, so a concurrent `delete_contact` update cannot archive that
  selected row before its resolver transaction commits.
- `atlas_brain/api/comms/webhooks.py` passes the provider `MessageSid` into
  background intelligence and fallback handling even when SMS persistence
  failed. `atlas_brain/comms/sms_intelligence.py` prefers that stable provider
  ID for EOM delivery receipts and `crm_event_id`; non-EOM source provenance
  keeps the local SMS ID whenever it exists, retaining the provider ID only
  when no local row exists.
- The focused ingress, public-route, and real-PostgreSQL tests prove direct
  partial-phone rejection, trusted concurrent relay replay, committed/nonfatal
  postcommit emission, legacy/case-sensitive dedupe behavior, `TRUNCATE`
  rejection, interaction event anchors, partial-extraction transport fallback,
  default-EOM pre-claim stage rejection with preserved non-stage claiming,
  generic ownership-bypass rejection, and persisted-appointment replay
  anchoring.

Contract reconciliation: every changed production path traces to Scope items
4 through 17 and Review Contract criteria 8 through 19; every new contract
requirement has a focused regression proof. No Customer/Site, jobs, calendar,
payment, first-clean, non-EOM CRM, or public success-envelope behavior is
touched. No untraced change or unmet contract item remains in this repair.

### Decision-seam analysis: fix

The one decision under review is the atomic resolver's replacement of the
general contact-resolution path. The first version correctly changed EOM
lifecycle ownership, but it silently shed predecessor behavior because it had no
explicit inventory. This repair closes that seam structurally: all inbound
identity shapes select one of three admitted paths before insert -- asserted
phone/email identity, a stable relay-event identity, or explicit rejection --
and the provider compares requested lifecycle values with the stored row before
blocking a real transition. Creation-only reasoning notification happens after
the transaction commits and cannot turn a committed contact insert into an
apparent ingress failure.

### Decision-seam analysis: EOM contact ownership and delivery execution

This is the §3k.2 repair. Finding 2 is the third ownership-bypass report on
this surface, and the current thread set names fresh adjacent evidence rather
than independent defects. The one decision is **which committed EOM contact an
inbound delivery and a generic ownership write are allowed to act on**. The
current split resolver/interaction and read/validate/update sequences answer
that question in more than one transaction, so each local lock has protected
only one side of the decision.

The selected closed-surface component is PostgreSQL: transaction-scoped
advisory locks for stable delivery/identity keys plus `FOR UPDATE` row locks for
the selected contact. No application lease, retry map, or hand-rolled
coordination is introduced. For one combined inbound command, the execution
model is:

1. Normalize asserted phone/email, `source`, and an explicitly supplied trusted
   delivery ID. Reject an identityless command without that explicit ID before
   opening a write path. Generic `source_ref` never becomes a delivery ID.
2. In one transaction, acquire the sorted, de-duplicated advisory locks for all
   asserted identity keys and the trusted delivery key. Check the EOM
   globally unique delivery receipt first, including its archived contact, with
   row locks. That receipt is the durable replay record for every explicit
   trusted delivery ID, not only name-only Web3Forms traffic; `source_ref`
   remains contact provenance and is never overloaded as the receipt ledger.
3. If that mapping already has its anchored interaction, return the prior
   contact and interaction as a replay without an insert. If it is archived,
   the command deliberately performs no new interaction write: a delivery must
   not be duplicated or newly attached to an archived original. For historical
   mappings without an interaction, an active row may receive the missing write
   in this transaction; an archived row remains read-only and is reported as a
   safe replay rather than manufacturing a replacement.
4. With no delivery mapping, resolve only active EOM/claimable-legacy
   phone-first then email candidates under `FOR UPDATE`, or insert an EOM
   `lead/new`. Insert/dedupe the inbound interaction against that same locked
   active contact before the transaction commits. Thus an archive that locked
   first is observed as archived and causes normal active replacement; an
   archive that arrives later waits until contact and interaction commit.
5. Emit reasoning notifications only after commit. A cancellation, exception,
   or process loss before commit leaves neither a new contact nor its inbound
   interaction; after commit, notification remains secondary and cannot change
   the committed delivery decision. The existing delivery adapters use this
   combined command only where they already record an inbound interaction;
   scheduler booking has no such interaction and remains resolver-only.
6. For generic `update_contact` ownership/lifecycle requests, select the target
   row `FOR UPDATE`, decide the EOM transition from that locked state, and issue
   the allowed `UPDATE` before ending the same transaction. `claim_contact` is
   its existing compare-and-set update; PostgreSQL row locking serializes it
   against the generic operation. Therefore either the generic non-EOM update
   commits first and the claim returns no row, or the claim commits first and
   generic validation rejects the now-EOM reassignment. It cannot validate NULL
   and subsequently overwrite a committed claim.

The invariant is that no combined EOM inbound command commits an interaction for
a contact that was archived before the command acquired its row lock, and no
generic update commits an EOM ownership transition based on a stale owner. The
model admits arbitrary archive, retry, generic-update, and claim scheduling;
the transaction/row-lock serialization supplies the order rather than a test
fixture list. It assumes every EOM adapter that writes `contact_interactions`
uses the combined ingress helper, and that the existing database transaction
and row-lock semantics remain available. Those callers are enumerated and
tested in this slice; a newly added adapter must use the same helper before it
may write an EOM inbound interaction.

### Replaced-path behavior inventory

This inventory is derived from `DatabaseCRMProvider.create_contact` and
`find_or_create_contact` on `origin/main`, not from the review findings.

| Previous behavior | Disposition in the EOM atomic boundary |
|---|---|
| Build a contact request from the supplied name, truthy phone/email, and caller extras. | Preserved for asserted identity values; EOM still fixes its own context, type, stage, source, and tags. |
| Normalize a submitted email to lowercase before lookup and insert. | Preserved. |
| Prefer phone resolution before email resolution. | Preserved across both EOM and claimable-legacy populations. |
| Ignore archived contacts during ordinary search. | Preserved by every atomic lookup. |
| Scope a tenant lookup to its own contact before a NULL-context legacy candidate. | Preserved for each identity channel. |
| Claim a matching NULL-context row and merge inbound fields into it. | Intentionally changed: EOM inbound returns every matching legacy row untouched so untrusted intake cannot claim or mutate it. |
| Merge non-null name/address/source/tag fields into an existing contact. | Intentionally changed for EOM inbound: matching contacts remain evidence-only; generic non-EOM merge behavior stays unchanged. |
| Allow `merge_existing=False` to return a match without claiming or updating it. | Preserved in effect: the EOM resolver always returns a matching contact without mutation. |
| Treat a sub-10-digit phone as an ordinary raw lookup input. | Intentionally changed: it is not an asserted identity in EOM ingress; email or a stable relay-event identity must anchor the request. |
| Admit a name-only relay submission. | Preserved only with an explicit trusted relay-event key; otherwise rejected before insert rather than creating duplicates. |
| Carry caller `source_ref` metadata. | Preserved for identity-bearing intake. Identityless intake stores its explicit trusted event as provenance because no caller metadata may attest identity; every trusted delivery is additionally recorded in the separate receipt ledger, so a later matched contact does not overwrite its own provenance. |
| Log an inbound interaction after linking its contact. | Preserved with its stable Gmail/call/SMS provider ID in recognized interaction metadata; only delivery retries change from UTC-day fallback dedupe to event-anchor dedupe. |
| Prefer an extracted call/SMS phone whenever it is truthy. | Intentionally changed for EOM only: a partial fragment yields to a full authoritative transport caller number; a full extraction still wins and non-EOM selection is unchanged. |
| Return `_was_created` so downstream callers can distinguish a new row. | Preserved. |
| Emit `crm.contact_created` after the contact path creates a row. | Preserved for atomic inserts after transaction commit; event-delivery failure is logged and non-fatal. |
| Validate that lead pipeline fields require a lead contact type. | Preserved: atomic creates are always `lead/new`; generic writes still validate pipeline requests. |
| Permit a merge whose requested contact type equals the stored type. | Preserved: the EOM lifecycle guard blocks actual type/stage transitions, not a same-value enrichment request. |
| Claim a NULL-context legacy row from a default-scoped MCP update. | Preserved for ordinary edits. Intentionally interrupted for an effective-default EOM `lead_stage` transition so an invalid lifecycle write cannot acquire tenant ownership before rejection. |
| Reassign a contact's generic `business_context_id`. | Preserved for transitions that neither enter nor leave EOM and for EOM same-context enrichment. Intentionally blocked for actual EOM ownership changes; `claim_contact`, not generic update sequencing, remains the acquisition path. |
| Link a persisted estimate appointment to CRM after calendar booking. | Preserved as non-blocking. When no full phone/email identity exists, the durable appointment ID is now the trusted replay anchor; the partial phone is not a matching identity. |
| Carry `tags` as supplied and leave metadata outside the generic merge list. | Preserved for new atomic leads; matching EOM rows remain unmodified, while generic merge semantics remain unchanged. |

## Intentional

- Existing call-action and voice booking controls remain available until the
  portal booking surface lands. This slice preserves their calendar behavior,
  removes their EOM customer stamping, and does not claim they advance
  `estimate_booked`.
- Attribution is stored as intake evidence only; campaign dashboards and Ads
  conversion uploads are deliberately not added.
- Existing mismatched `customer + lead_stage` rows are not changed here; Juan's
  review queue belongs with the authenticated conversion UI.
- Until Slice 3's constrained transition service ships, no surface can change
  an EOM contact's `lead_stage` or `contact_type`, including marking it lost.
  `lead_owner` and `next_follow_up_at` remain writable for daily triage.
- Generic updates also cannot move a contact into or out of EOM. The existing
  dedicated claim path owns only the claimable-legacy-to-EOM acquisition case;
  broader reassignment needs a later explicit ownership workflow.
- A local-only phone on an internally persisted EOM estimate booking creates a
  lead anchored to that appointment. It is evidence, not a cross-contact
  matching identity.
- Lifecycle evidence intentionally blocks hard deletion of a contact that owns
  it. Runtime deletion is soft archive; a statutory purge needs a future,
  explicit retention/deletion policy rather than an accidental cascade.
- The maturity baseline refresh records scanner state already present in the
  touched MCP/tool/storage paths (including real-entrypoint test seams); it
  does not waive a new production behavior or error path.

## Deferred

- Slice 2: service-authenticated canonical estimate booking, calendar
  projection/retry, and the portal Leads tab; then route EOM notification
  booking actions there.
- Slice 3: Juan-only conversion, historical review queue, and idempotent
  Customer-draft handoff to the time tracker.
- First-clean lifecycle, card-on-file/Stripe, customer emails, lost outcomes,
  and attribution reporting remain separate operator-approved phases.

Parked hardening: none.

## Rollback

Migrations 351 and 352 are ingress prerequisites. To roll them back safely,
first deploy an application version that does not require the ledger or delivery
receipts (or disable EOM inbound intake) so no new inbound request reaches a
schema that is being removed. Then, in one maintenance transaction, drop
`eom_inbound_delivery_receipts`, then drop `trg_record_eom_lead_created` from
`contacts`, drop `record_eom_lead_created()`, drop
`trg_prevent_eom_lead_lifecycle_event_mutation` from
`eom_lead_lifecycle_events`, drop
`trg_prevent_eom_lead_lifecycle_event_truncate` from
`eom_lead_lifecycle_events`, drop
`prevent_eom_lead_lifecycle_event_mutation()`, and drop
`eom_lead_lifecycle_events`. Finally remove the migrations' applied records only
if the deployment system requires it before a later reapply. Never drop the
ledger while the new resolver is serving: it intentionally fails closed.

## Verification

- Earlier current-head repair: Python compile check and the exact EOM lead-pipeline
  workflow test-file list passed against a fresh PostgreSQL 16 database after
  rebasing the scoped-Gmail credential lane: **224 passed**. This includes the
  schema-isolated sent-email route proof, trusted relay replay,
  lifecycle-ledger `TRUNCATE` proof, stable interaction anchors, EOM
  transport-phone fallback, default-EOM pre-claim stage rejection, and the
  upstream scoped-Gmail regression cases now enrolled beside this lane.
- Passed the unit ratchet with the checked-out and origin/main baselines.
- Passed maturity sweeps for atlas_brain/mcp, atlas_brain/tools, and
  atlas_brain/storage against their corresponding baselines; the three accepted
  baseline changes only capture existing test seams used by ingress reachability
  tests.
- Passed the exact current-head maturity ratchets for
  `atlas_brain/reasoning`, `atlas_brain/security`, and
  `atlas_brain/storage`; the repair adds no new direct producer mock or storage
  test seam.
- Passed the exact maturity ratchets for `atlas_brain/autonomous` and
  `atlas_brain/comms`; the stable event anchors and EOM-only phone fallback add
  no new brittleness above their baselines.
- Current review repair: the exact rebased EOM workflow list passed **224
  tests** after the ownership and appointment-anchor changes. The
  `atlas_brain/tools` and `atlas_brain/storage` maturity ratchets also passed
  with no baseline increase.
- Passed the exact `atlas_brain/reasoning`, `atlas_brain/security`, and
  `atlas_brain/storage` maturity ratchets after folding non-EOM preservation
  assertions into existing adapter tests; the storage baseline remains at 168
  with 41 internal mocks, so this repair introduces no new ratchet count.
- Passed the plan audit, plan-sync check, Python compile check, and diff check.
- Current archive/SMS review repair: `tests/test_eom_lead_ingress.py` passed
  **17 tests** and the real-PostgreSQL
  `tests/test_eom_lead_pipeline_integration.py` passed **13 tests**. The latter
  executes six selected-match/archive interleavings across EOM and
  claimable-legacy phone/email identity branches; the former proves stable
  provider anchoring with and without a local SMS row and through the
  intelligence and fallback adapters.
- Current rebased head: the exact EOM lead-pipeline workflow list passed **231
  tests** against PostgreSQL 16 after rebasing onto `origin/main` at
  `9b983be1f`. The `atlas_brain/api` and `atlas_brain/comms` maturity ratchets
  also passed with no baseline update.
- Current §3k.2 execution-model repair: the exact current-head EOM workflow
  list passed **234 tests** against PostgreSQL 16. Its real-PostgreSQL proofs
  hold six selected-contact rows through interaction commit, replay an
  identity-bearing Web3Forms delivery after archival without overwriting an
  existing contact's `source_ref`, and serialize generic ownership update with
  `claim_contact` in both orders.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 15 |
| `atlas_brain/api/comms/webhooks.py` | 71 |
| `atlas_brain/api/leads.py` | 125 |
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 41 |
| `atlas_brain/comms/call_intelligence.py` | 70 |
| `atlas_brain/comms/sms_intelligence.py` | 85 |
| `atlas_brain/mcp/crm_server.py` | 18 |
| `atlas_brain/services/crm_provider.py` | 580 |
| `atlas_brain/services/eom_lead_ingress.py` | 180 |
| `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql` | 98 |
| `atlas_brain/storage/migrations/352_eom_inbound_delivery_receipts.sql` | 21 |
| `atlas_brain/tools/scheduling.py` | 29 |
| `plans/PR-EOM-Funnel-Ingress.md` | 671 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 6 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 4 |
| `tests/test_crm_read_scoping.py` | 41 |
| `tests/test_eom_lead_ingress.py` | 522 |
| `tests/test_eom_lead_pipeline_integration.py` | 908 |
| `tests/test_eom_sent_email_tenant_scope.py` | 2 |
| `tests/test_leads_intake.py` | 56 |
| `tests/test_tenant_stamping.py` | 34 |
| **Total** | **3581** |
