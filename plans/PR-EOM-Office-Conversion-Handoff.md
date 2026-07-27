# PR-EOM-Office-Conversion-Handoff

## Why this slice exists

This is the next EOM funnel slice after merged inbound preservation. A completed
estimate normally gives Juan the customer name/contact/address plus the initial
site's per-visit price, service scope, and preferred cleaning schedule. Juan
approves that completed estimate in the office; it should create the operational
Customer and first Site there, not turn a field action or Google Calendar entry
into a customer.

### Problem-derived contract

- Root cause: the only current EOM lifecycle record is the Atlas contact, while
  the completed-estimate facts that make someone a real customer belong to the
  time tracker Customer/Site model. Generic CRM writes are correctly frozen
  after #2216, but there is no idempotent, office-authenticated handoff joining
  one approved EOM lead to one tracker Customer and initial Site. Splitting a
  blind CRM promotion from later operational creation would recreate the
  duplicate-system and partial-retry debt this funnel is meant to remove.
- Correct fix must touch/change: make the time tracker the command owner,
  because it authenticates the named office employee and owns Customer/Site,
  rates, and service schedule. Its Juan-only approval command must create or
  return exactly one Customer draft and one initial Site keyed by the Atlas
  contact and a stable approval key. It then calls an Atlas service-authenticated
  finalization endpoint. Atlas must verify the active EOM `lead/new` contact,
  write one durable handoff record containing only cross-system IDs and actor
  evidence, atomically transition that contact to `customer` only after the
  tracker write exists, and append the lifecycle event. Both sides must return
  the prior result for the same key and make the tracker-to-Atlas finalization
  retryable after a network failure.
- Must not change: public website intake; inbound identity/receipt semantics
  from #2216; field/Google Calendar imports; estimate appointment booking;
  generic non-EOM CRM; payroll, QR, jobs, first-clean, card collection,
  receivables, customer-facing website onboarding, and EOM Website PR #70.
  Atlas must not store the estimate price, rate type, frequency, schedule, or
  service scope as a second operational source of truth. A declined estimate,
  reschedule/cancel, additional sites, self-service onboarding, and later
  customer updates are separate commands.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Add the Atlas half of the office-approved customer handoff: dedicated
   service authentication, a durable idempotency/link record, a constrained
   `lead/new -> customer` finalization transaction, and an append-only
   lifecycle event. The companion time-tracker PR owns the authenticated Juan
   approval endpoint and Customer/Site transaction; neither half is published
   as merge-ready until the other has the matching contract and proof.
2. Add real PostgreSQL and HTTP-boundary proof for duplicate finalization,
   foreign/non-lead/non-new rejection before writes, tracker-first/Atlas-retry
   recovery, and the invariant that no Atlas lifecycle promotion exists without
   a matching tracker Customer/Site identifier.

### Review Contract

1. A tracker request from the configured Juan employee ID creates or returns
   one Customer draft and one initial Site for one Atlas contact; the completed
   estimate's price, rate type, service scope, and preferred schedule are
   stored only with that Customer/Site. The companion's database integration
   proof settles this local transaction.
2. The Atlas finalization route accepts only its dedicated enabled service token
   and a validated tracker-issued handoff payload. Missing/invalid token, blank
   actor evidence, foreign contact, inactive contact, non-lead, or a lead not
   in `new` fails before contact/lifecycle/handoff writes. HTTP tests settle
   the route's fail-closed boundary.
3. The Atlas handoff table has a unique contact/key invariant and immutable
   Customer/Site IDs. A same-key retry returns the original transition without
   another lifecycle event; a different key for an already converted contact
   is rejected. Real PostgreSQL tests settle these execution properties.
4. Atlas moves `contact_type` from `lead` to `customer` and clears the lead
   stage only in the finalization transaction after storing the tracker link.
   Generic provider/MCP paths remain unable to make that lifecycle change;
   integration tests settle the transition and blocked bypasses.
5. If the tracker has committed its Customer/Site but the Atlas call fails, its
   persisted approval operation retries the same request/key. Atlas recovers
   exactly one finalization; no second Customer/Site or lifecycle event is
   produced. Cross-repository contract tests settle the recovery rule.
- Reachability proof: authenticated time-tracker admin request -> configured
  Juan-only guard -> durable tracker Customer/Site operation -> Atlas
  service-authenticated finalization -> Atlas handoff/lifecycle rows. Each
  repository's route tests exercise its real dependency chain; a joint smoke
  request verifies the stable JSON request/response contract without exposing
  the Atlas service token to a browser.
- Affected surfaces: Atlas settings/startup validation, EOM funnel router and
  service auth, CRM lifecycle transition, handoff migration, EOM pipeline CI,
  and the companion time-tracker admin API/customer-onboarding flow.
- Risk areas: mistaken actor authority, credential exposure, contact tenant
  escape, duplicate Customer/Site creation, tracker/Atlas partial failure,
  lifecycle bypass, and operational-fact duplication.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/__init__.py`
- `atlas_brain/api/eom_lead_funnel.py`
- `atlas_brain/api/eom_lead_funnel_auth.py`
- `atlas_brain/config.py`
- `atlas_brain/main.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_lead_conversion.py`
- `atlas_brain/storage/migrations/353_eom_customer_handoffs.sql`
- `plans/PR-EOM-Office-Conversion-Handoff.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

The browser never calls Atlas. The time tracker verifies the existing employee
session, compares its stable employee ID with a deployment-configured Juan
approver ID, validates the completed-estimate fields, and writes its Customer,
initial Site, and approval operation in one local transaction. `atlas_contact_id`
and a per-approval idempotency key make that result recoverable after a retry.

The tracker sends only the opaque contact/customer/site IDs, approval key, and
trusted actor evidence to Atlas using a dedicated server-side service token.
Atlas locks the contact and handoff-key rows; validates EOM ownership,
`lead/new`, and the immutable tracker IDs; records the handoff; changes the
contact to `customer` with no lead stage; and appends the matching lifecycle
event in the same transaction. A duplicate request returns the recorded
handoff. Atlas never receives or persists the per-visit price, schedule,
frequency, or service details.

## Intentional

- Customer/Site creation precedes Atlas finalization because the tracker owns
  the actual customer, site, rate, and schedule. A failed callback is visible
  as a retryable pending handoff, not silently inferred from an Atlas status.
- Juan-only authority is enforced where the end-user session is verifiable: the
  time tracker. Atlas additionally accepts only its dedicated server token and
  records the actor evidence, but does not mistake a caller-controlled display
  name for authentication.
- One initial Site is created on approval. A multi-site customer is an explicit
  later office action, not an estimate-time guess.

## Deferred

- A declined/non-customer outcome and explicit reopen command.
- Canonical estimate booking, reschedule/cancel, and calendar projection.
- Customer-draft completion, first-clean, card collection, emails, and
  attribution reporting.
- Backfill/linking for existing Customers and more than one initial Site.

Parked hardening: none.

## Verification

- Before implementation: create and approve the matching time-tracker contract
  and its clean worktree; do not code the Atlas half against an unreviewed
  companion shape.
- Before push: focused Atlas route/service/real-PostgreSQL tests, matching
  companion tests, joint contract smoke, plan synchronization, local review,
  and the unit-gate ratchet.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `atlas_brain/api/__init__.py` | 2 |
| `atlas_brain/api/eom_lead_funnel.py` | 85 |
| `atlas_brain/api/eom_lead_funnel_auth.py` | 79 |
| `atlas_brain/config.py` | 17 |
| `atlas_brain/main.py` | 2 |
| `atlas_brain/services/crm_provider.py` | 140 |
| `atlas_brain/services/eom_lead_conversion.py` | 44 |
| `atlas_brain/storage/migrations/353_eom_customer_handoffs.sql` | 20 |
| `plans/PR-EOM-Office-Conversion-Handoff.md` | 174 |
| `tests/test_eom_lead_conversion.py` | 153 |
| `tests/test_eom_lead_conversion_integration.py` | 105 |
| **Total** | **823** |
