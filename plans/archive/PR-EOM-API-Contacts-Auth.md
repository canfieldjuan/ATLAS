# PR-EOM-API-Contacts-Auth

## Why this slice exists

Issue #2170 records a production security gap deferred from #2157: the
operator-facing contact timeline and call-search HTTP routes are reachable
without an authentication dependency and read across every CRM tenant. The
routes serve data for both `effingham_maids` and `churnsignals`, so silently
applying the EOM MCP default would hide valid Churn Signals data rather than
authorize the caller's requested page. This production-hardening slice is
allowed ahead of further product work because it closes a concrete
authorization and cross-tenant privacy risk on existing reachable endpoints.

Diff-budget override: the platform-admin dependency, exact provider/repository
predicates, route wiring, CI enrollment, and real-PostgreSQL cross-tenant
entrypoint proof are one indivisible security repair. Splitting the proof from
the enforcement would temporarily ship either an unverified guard or a test
with no protected production path.

### Problem-derived contract

- Root cause: `atlas_brain/api/contacts.py` mounts both routes without
  `require_auth`, accepts no tenant selector, fetches timeline contacts
  unscoped, and calls repository search without a tenant predicate. Any caller
  can therefore enumerate contact history and call metadata across business
  contexts; filtering after a page limit would not repair that leak.
- Correct fix must touch/change: require a valid platform-admin dashboard
  identity on both existing routes; require and normalize an explicit
  `business_context_id`; apply exact tenant equality to the contact lookup and
  every timeline child query and call-transcript search in SQL before
  ordering/limit; revalidate exact ownership after the final child await so a
  concurrent reassignment cannot serialize a stale 200; build the timeline
  only from tenant-addressable interactions, appointments, and calls; exclude
  email sources that still lack tenant keys; retain the existing response
  event shape while flagging the scoped email omission; and prove the real
  FastAPI entrypoints against PostgreSQL with same-keyword foreign rows that
  would otherwise win the page and a deterministic ownership race.
- Must not change: do not bind SaaS `account_id` UUIDs to CRM business-context
  strings without an accepted mapping model; do not default these two-tenant
  routes to EOM; do not claim or include NULL legacy rows on this authenticated
  HTTP surface; do not lift the email/B2B omissions owned by #2171; do not
  change CRM MCP scoping semantics; do not alter contact, appointment, call,
  invoice, or service records; and do not modify any frontend product shape or
  unrelated API route.

## Scope (this PR)

Ownership lane: eom-crm/api-auth
Slice phase: Production hardening

1. Gate the contact timeline and call-search endpoints to platform admins and
   make explicit strict tenant selection mandatory.
2. Add provider/repository query support that enforces exact tenant scope
   before page limits while leaving existing unscoped and MCP callers intact.
3. Add real-entrypoint PostgreSQL proof for authentication, tenant isolation,
   page-starvation resistance, and safe timeline source selection.

### Review Contract

- Acceptance criteria:
  1. Missing/invalid auth returns 401 and an authenticated non-platform user
     returns 403 on both routes.
  2. Missing, blank, or overlong `business_context_id` is rejected.
  3. Timeline lookup for a foreign contact reads as 404; a same-tenant contact
     returns only tenant-addressable events and identifies email omission.
  4. Call search binds exact tenant equality inside SQL before `ORDER BY` and
     `LIMIT`, so newer foreign rows cannot starve the requested tenant's page.
  5. A contact reassigned to foreign or NULL ownership between the initial
     guard and child reads produces 404 with no stale events.
  6. Existing repository callers that omit tenant scope preserve their legacy
     SQL behavior, and the CRM MCP's tenant-plus-legacy semantics do not
     change.
- Reachability proof: FastAPI `GET /contacts/{id}/timeline` and
  `GET /comms/calls/search` execute through their real route/dependency
  functions against isolated PostgreSQL rows; observable HTTP status/payload
  assertions prove auth refusal, foreign-as-not-found behavior, and a
  tenant-only search result despite a newer foreign match, plus a
  deterministic post-guard ownership reassignment.
- Affected surfaces: `atlas_brain/api/contacts.py`,
  `DatabaseCRMProvider.get_contact`,
  `CallTranscriptRepository.search`/`get_by_contact_id`, the EOM PostgreSQL CI
  lane, and focused route/repository tests.
- Risk areas: treating authentication as authorization, accepting an
  untrusted/default tenant implicitly, filtering after `LIMIT`, leaking
  unaddressable email data, changing NULL-legacy MCP behavior, archived
  timeline compatibility, and auth-disabled local-development behavior.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R7, R8, R10, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/contacts.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/repositories/call_transcript.py`
- `plans/INDEX.md`
- `plans/PR-EOM-API-Contacts-Auth.md`
- `plans/archive/PR-EOM-Recurring-Appointments.md`
- `tests/test_crm_read_scoping.py`
- `tests/test_eom_contacts_api_tenant_scope.py`

## Mechanism

The router composes `require_auth` into a platform-admin dependency and requires
an explicit bounded business-context query parameter. Timeline handling first
performs an exact scoped contact lookup, then gathers only interactions,
appointments, and strictly tenant-matched call transcripts. Each child query
reasserts exact current contact ownership, and a final exact contact read after
the gather rejects any ownership transition before synchronous serialization.
Existing per-source fail-open behavior remains, but unaddressable email sources
are not queried or serialized under scope.

The CRM provider adds an optional exact-scope predicate to `get_contact`;
call-transcript search adds an optional exact-scope predicate before ordering
and limit. Linked interaction, appointment, and transcript reads gain opt-outs
from their existing tenant-plus-NULL compatibility modes so this HTTP route
can be strict while MCP callers retain current behavior.

## Intentional

- Platform-admin authorization is deliberate: these routes expose cross-tenant
  operator CRM data and an ordinary authenticated SaaS account must not gain
  access merely by guessing `business_context_id`.
- Tenant selection remains caller-supplied because the same operator UI serves
  two contexts and no accepted SaaS-account-to-business-context mapping exists.
- Strict HTTP scoping excludes NULL legacy calls. The MCP compatibility surface
  retains tenant-plus-NULL behavior through the repository's default.
- Scoped timelines omit sent/inbox email events and report that omission until
  #2171 makes the stores tenant-addressable.
- Archived same-tenant contacts remain readable as history; this slice changes
  authorization, not retention.
- No in-repo Intel UI caller for either path was found, so this slice repairs
  the server contract without inventing a frontend selector or changing UI
  product shape.

## Deferred

- #2171 owns tenant-addressable sent/inbox email and B2B enrichment stores and
  lifting the scoped omissions after those stores can prove ownership.
- Any durable mapping from authenticated SaaS accounts/users to allowed CRM
  `business_context_id` values requires its own accepted data/authorization
  model. Until then these cross-tenant operator routes remain platform-admin
  only.

Parked hardening: none.

## Verification

- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://... python -m pytest
  tests/test_eom_contacts_api_tenant_scope.py -q` - passed, 1 test.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://... python -m pytest
  tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py
  tests/test_eom_contacts_api_tenant_scope.py
  tests/test_eom_lead_pipeline_integration.py
  tests/test_eom_recurring_appointments_integration.py
  tests/test_leads_intake.py tests/test_migrations_runner.py -q` - passed,
  111 tests.
- `python scripts/maturity_sweep.py atlas_brain/storage ...` - passed with no
  new brittleness above the stored baseline.
- `python scripts/maturity_sweep.py atlas_brain/api ...` - passed with no new
  brittleness above the stored baseline.
- `python -m ruff check atlas_brain/api/contacts.py
  atlas_brain/services/crm_provider.py
  atlas_brain/storage/repositories/call_transcript.py
  tests/test_eom_contacts_api_tenant_scope.py --ignore F841` - passed.
- `python -m py_compile` on the three changed runtime modules and focused test
  - passed.
- Plan/body audits and `git diff --check` - passed. The managed pre-push review
  remains the publication gate.
- Independent judgment review round one found four blockers. All four classes
  were repaired; an exact-head delta review is required before publication.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/api/contacts.py` | 133 |
| `atlas_brain/services/crm_provider.py` | 62 |
| `atlas_brain/storage/repositories/call_transcript.py` | 32 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-API-Contacts-Auth.md` | 185 |
| `plans/archive/PR-EOM-Recurring-Appointments.md` | 0 |
| `tests/test_crm_read_scoping.py` | 11 |
| `tests/test_eom_contacts_api_tenant_scope.py` | 480 |
| **Total** | **913** |
