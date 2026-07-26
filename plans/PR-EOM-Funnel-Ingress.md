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
  lead boundary. Generic CRM updates can also mutate a claimable legacy row
  before it enters EOM, and ingress can run even if the new lifecycle ledger
  migration did not install. This can create or mutate a customer from
  untrusted inbound data, attach an inbound lead to an archived or wrong
  identity, duplicate a new identity under concurrent delivery, or leave lead
  evidence incomplete. The website also does not send per-lead ad identifiers
  to Atlas.
- Correct fix must touch/change: centralize EOM inbound identity resolution so
  unmatched non-spam web/relay/calls/SMS/legacy estimate bookings become
  `lead/new` and any matching contact is read-only; resolve only active contacts
  with phone priority across EOM and claimable-legacy populations; serialize
  asserted inbound identities before lookup/insert; prevent generic lifecycle
  writes from bypassing the rule before a row is claimable into EOM; require the
  lifecycle table and trigger before atomic ingress inserts; add a durable
  lifecycle creation record; extend the public intake payload and interaction
  metadata for attribution without discarding a changed snapshot; and cover the
  real public route/core call paths plus database behavior with focused tests.
- Must not change: existing customer identity, non-EOM CRM behavior, the
  current public intake response/CORS/email acknowledgement semantics, Google
  Calendar booking behavior, Customer/Site onboarding, jobs, payments,
  first-clean/card-on-file work, and adjacent PRs #2195/#2200.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice
Max files: 20

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

- Passed focused ingress suite against isolated PostgreSQL 16 (75 passed):
  `pytest -q` followed by the four focused test paths in this PR's Files
  touched list.
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
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 12 |
| `atlas_brain/comms/call_intelligence.py` | 34 |
| `atlas_brain/comms/sms_intelligence.py` | 34 |
| `atlas_brain/mcp/crm_server.py` | 15 |
| `atlas_brain/services/crm_provider.py` | 254 |
| `atlas_brain/services/eom_lead_ingress.py` | 100 |
| `atlas_brain/storage/migrations/351_eom_lead_lifecycle_events.sql` | 91 |
| `atlas_brain/tools/scheduling.py` | 28 |
| `plans/PR-EOM-Funnel-Ingress.md` | 228 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 6 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/maturity_sweep/baseline_atlas_brain_tools.json` | 4 |
| `tests/test_crm_read_scoping.py` | 8 |
| `tests/test_eom_lead_ingress.py` | 330 |
| `tests/test_eom_lead_pipeline_integration.py` | 386 |
| `tests/test_leads_intake.py` | 40 |
| `tests/test_tenant_stamping.py` | 31 |
| **Total** | **1763** |
