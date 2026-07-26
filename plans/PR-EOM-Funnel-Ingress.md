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
  identifiers to Atlas.
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
  focused tests.
- Must not change: existing customer identity, non-EOM CRM behavior, the
  current public intake response/CORS/email acknowledgement semantics, Google
  Calendar booking behavior, Customer/Site onboarding, jobs, payments,
  first-clean/card-on-file work, and adjacent PRs #2195/#2200. Existing
  idempotent generic enrichment of an EOM contact whose type/stage is already
  unchanged must continue to work.

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

### Cold diff reconstruction: current-head repair

- `atlas_brain/services/crm_provider.py` adds a normalized, lock-protected
  `(business_context_id, source, source_ref)` replay identity only when phone
  and email are both absent; it returns an existing replay row unchanged,
  persists that normalized key on a new row, permits same-value type/stage
  enrichment, and emits `crm.contact_created` only after the insert transaction
  has committed. It catches secondary event failure after that commit.
- `atlas_brain/services/eom_lead_ingress.py` rejects an identityless call that
  has no stable relay key before either the atomic provider or test-double
  fallback can create a row, and passes normalized relay metadata downstream.
- `atlas_brain/autonomous/tasks/gmail_digest.py` supplies the message ID as the
  relay key when it has one, while skipping name-only relay email that has no
  such key. Email- or phone-identified relay email remains admitted without an
  ID.
- `tests/test_eom_lead_pipeline_integration.py` proves five concurrent replay
  identities create exactly five contacts/events, proves the reasoning event
  observes a committed contact and cannot make a later insert fail, and proves
  the generic `create_contact` same-type EOM merge remains available.
- `tests/test_eom_lead_ingress.py` proves direct identityless rejection and the
  Gmail no-anchor skip. `tests/test_eom_sent_email_tenant_scope.py` applies
  migration 351 in its isolated schema so the public route reaches the intended
  ingress contract.

Contract reconciliation: every changed production path traces to Scope item 5
or the existing fail-closed ingress prerequisite in Scope item 4; every new
contract requirement has a focused regression proof. No Customer/Site, jobs,
calendar, payment, first-clean, or non-EOM CRM path is touched. No untraced
change or unmet contract item remains in this repair.

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
| Admit a name-only relay submission. | Preserved only with a non-empty, stable `source + source_ref` replay key; otherwise rejected before insert rather than creating duplicates. |
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
`prevent_eom_lead_lifecycle_event_mutation()`, and drop
`eom_lead_lifecycle_events`. Finally remove the migration's applied record only
if the deployment system requires it before a later reapply. Never drop the
ledger while the new resolver is serving: it intentionally fails closed.

## Verification

- Current-head repair: Python compile check and the exact EOM lead-pipeline
  workflow test-file list passed against a fresh PostgreSQL 16 database:
  **146 passed**. This includes the schema-isolated sent-email route proof.
- Passed the unit ratchet with the checked-out and origin/main baselines.
- Passed maturity sweeps for atlas_brain/mcp, atlas_brain/tools, and
  atlas_brain/storage against their corresponding baselines; the three accepted
  baseline changes only capture existing test seams used by ingress reachability
  tests.
- Passed the plan audit, plan-sync check, Python compile check, and diff check.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 17 |
| `atlas_brain/api/comms/webhooks.py` | 28 |
| `atlas_brain/api/leads.py` | 113 |
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 19 |
| `atlas_brain/comms/call_intelligence.py` | 34 |
| `atlas_brain/comms/sms_intelligence.py` | 34 |
| `atlas_brain/mcp/crm_server.py` | 15 |
| `atlas_brain/services/crm_provider.py` | 319 |
| `atlas_brain/services/eom_lead_ingress.py` | 111 |
| `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql` | 91 |
| `atlas_brain/tools/scheduling.py` | 28 |
| `plans/PR-EOM-Funnel-Ingress.md` | 310 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 6 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 4 |
| `tests/test_crm_read_scoping.py` | 8 |
| `tests/test_eom_lead_ingress.py` | 373 |
| `tests/test_eom_lead_pipeline_integration.py` | 494 |
| `tests/test_eom_sent_email_tenant_scope.py` | 1 |
| `tests/test_leads_intake.py` | 40 |
| `tests/test_tenant_stamping.py` | 31 |
| **Total** | **2080** |
