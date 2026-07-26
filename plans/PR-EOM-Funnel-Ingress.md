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
  inbound lead can now be rejected.
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
  authoritative transport number, before EOM admission.
- Must not change: existing customer identity, non-EOM CRM behavior, the
  current public intake response/CORS/email acknowledgement semantics, Google
  Calendar booking behavior, Customer/Site onboarding, jobs, payments,
  first-clean/card-on-file work, and adjacent PRs #2195/#2200. Existing
  idempotent generic enrichment of an EOM contact whose type/stage is already
  unchanged must continue to work. Full-phone-only form intake and
  email-backed partial-phone form intake retain their current behavior. Existing
  interaction type/intent mapping, non-EOM call/SMS phone selection, and
  ordinary daily interaction dedupe without a stable inbound event remain
  unchanged.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 21

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
     leads in the same database transaction as contact insertion; real
     PostgreSQL tests settle creation, exact-once identity, and immutability.
  5. Existing public CORS, acknowledgement-send, and non-EOM CRM behavior are
     covered by the existing intake/provider tests.
  6. Atomic EOM ingress fails before a contact write if migration 351's ledger
     table or contacts trigger is absent; real-provider coverage settles the
     prerequisite.
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

The atomic resolver checks the migration-351 table and contacts trigger inside
the transaction before any identity lookup or write. If either is absent, that
inbound path fails closed rather than creating an unledgered EOM lead. Its
lookup admits active rows only and resolves phone across both eligible
populations before attempting email.

Migration 351 adds `eom_lead_lifecycle_events` and an EOM-only contact trigger:
a new `lead/new` contact records one immutable `lead_created` event in its
insert transaction. The trigger records system actor/source/operation metadata;
the later booking/conversion slice will add human-authored transition events.

The public request accepts bounded attribution fields and writes a non-empty
snapshot only to the intake interaction metadata. The interaction identity
includes that snapshot, so a later submission with a different click or UTM
value is retained. It does not change contact identity or the response envelope.
Generic lifecycle edits reject EOM rows and claimable-legacy rows, so the later
constrained transition service is the only stage/type writer.

The trusted Web3Forms relay alone passes a separate replay-event key; a generic
`source_ref` is metadata and may be constant, so it cannot attest a distinct
identity. Direct form intake therefore requires email or a ten-digit phone.
Interaction dedupe keeps the predecessor byte basis for no-attribution records
and appends an exact submitted attribution snapshot only when present. The
ledger's immutable policy covers table truncation as well as row mutation.

Every inbound Gmail, call, and SMS delivery that has a stable provider event ID
writes that ID into recognized interaction-anchor metadata, so contact replay
and interaction replay have the same durability boundary. For EOM call/SMS,
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
  `atlas_brain/services/crm_provider.py` use that explicit relay-event value,
  not generic caller `source_ref`, for identityless admission, locking, lookup,
  and persistence. They continue to use asserted email/full-phone identities
  for all other inbound lead paths.
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
- The focused ingress, public-route, and real-PostgreSQL tests prove direct
  partial-phone rejection, trusted concurrent relay replay, committed/nonfatal
  postcommit emission, legacy/case-sensitive dedupe behavior, `TRUNCATE`
  rejection, interaction event anchors, and partial-extraction transport
  fallback.

Contract reconciliation: every changed production path traces to Scope items
4 through 9 and Review Contract criteria 8 through 11; every new contract
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
| Carry caller `source_ref` metadata. | Preserved for identity-bearing intake. Intentionally changed for identityless intake: caller metadata is not stored or assumed unique; only the relay adapter supplies the stored replay-event provenance. |
| Log an inbound interaction after linking its contact. | Preserved with its stable Gmail/call/SMS provider ID in recognized interaction metadata; only delivery retries change from UTC-day fallback dedupe to event-anchor dedupe. |
| Prefer an extracted call/SMS phone whenever it is truthy. | Intentionally changed for EOM only: a partial fragment yields to a full authoritative transport caller number; a full extraction still wins and non-EOM selection is unchanged. |
| Return `_was_created` so downstream callers can distinguish a new row. | Preserved. |
| Emit `crm.contact_created` after the contact path creates a row. | Preserved for atomic inserts after transaction commit; event-delivery failure is logged and non-fatal. |
| Validate that lead pipeline fields require a lead contact type. | Preserved: atomic creates are always `lead/new`; generic writes still validate pipeline requests. |
| Permit a merge whose requested contact type equals the stored type. | Preserved: the EOM lifecycle guard blocks actual type/stage transitions, not a same-value enrichment request. |
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

Migration 351 is an ingress prerequisite. To roll it back safely, first deploy
an application version that does not require the ledger (or disable EOM inbound
intake) so no new inbound request reaches a schema that is being removed. Then,
in one maintenance transaction, drop `trg_record_eom_lead_created` from
`contacts`, drop `record_eom_lead_created()`, drop
`trg_prevent_eom_lead_lifecycle_event_mutation` from
`eom_lead_lifecycle_events`, drop
`trg_prevent_eom_lead_lifecycle_event_truncate` from
`eom_lead_lifecycle_events`, drop
`prevent_eom_lead_lifecycle_event_mutation()`, and drop
`eom_lead_lifecycle_events`. Finally remove the migration's applied record only
if the deployment system requires it before a later reapply. Never drop the
ledger while the new resolver is serving: it intentionally fails closed.

## Verification

- Current-head repair: Python compile check and the exact EOM lead-pipeline
  workflow test-file list passed against a fresh PostgreSQL 16 database:
  **150 passed**. This includes the schema-isolated sent-email route proof,
  trusted relay replay, lifecycle-ledger `TRUNCATE` proof, stable interaction
  anchors, and EOM transport-phone fallback.
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
- Passed the plan audit, plan-sync check, Python compile check, and diff check.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 17 |
| `atlas_brain/api/comms/webhooks.py` | 28 |
| `atlas_brain/api/leads.py` | 117 |
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 27 |
| `atlas_brain/comms/call_intelligence.py` | 47 |
| `atlas_brain/comms/sms_intelligence.py` | 47 |
| `atlas_brain/mcp/crm_server.py` | 15 |
| `atlas_brain/services/crm_provider.py` | 346 |
| `atlas_brain/services/eom_lead_ingress.py` | 137 |
| `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql` | 98 |
| `atlas_brain/tools/scheduling.py` | 28 |
| `plans/PR-EOM-Funnel-Ingress.md` | 395 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 6 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 4 |
| `tests/test_crm_read_scoping.py` | 8 |
| `tests/test_eom_lead_ingress.py` | 460 |
| `tests/test_eom_lead_pipeline_integration.py` | 495 |
| `tests/test_eom_sent_email_tenant_scope.py` | 1 |
| `tests/test_leads_intake.py` | 56 |
| `tests/test_tenant_stamping.py` | 31 |
| **Total** | **2367** |
