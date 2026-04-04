# Vendor Intelligence Watchlists & Accounts-in-Motion Feed

Date: 2026-04-04

## Goal

Build the highest-priority Churn Signals end product:

`Vendor Intelligence Watchlists & Accounts-in-Motion Feed`

This product should turn existing churn and displacement artifacts into a daily-use monitoring surface for:

- RevOps
- product marketing
- competitive intelligence
- strategic AEs

It should not depend on fresh synthesis at page load.

---

## Product Definition

### Primary user job

"Show me meaningful vendor movement and account-level displacement signals for the vendors I care about, with enough evidence to trust the signal and decide whether to act."

### Core UX

The product combines three things:

1. Watchlist management
2. Vendor movement feed
3. Accounts-in-motion drilldown

### Required trust layer

This product must include evidence drilldown from v1.

If the user cannot inspect:

- source provenance
- witness-backed quote context
- timing and urgency clues
- reasoning summary freshness

then the feed becomes another black-box signal board, which is the wrong product.

---

## What Already Exists

### Existing backend building blocks

Tracked vendors:

- [b2b_tenant_dashboard.py](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_tenant_dashboard.py)
  - `GET /b2b/tenant/vendors`
  - `POST /b2b/tenant/vendors`

Slow-burn tracked-vendor feed:

- [b2b_tenant_dashboard.py](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_tenant_dashboard.py)
  - `GET /b2b/tenant/slow-burn-watchlist`

Tracked vendor detail:

- [b2b_tenant_dashboard.py](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_tenant_dashboard.py)
  - `GET /b2b/tenant/signals/{vendor_name}`

Persisted accounts-in-motion:

- [b2b_dashboard.py](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_dashboard.py)
  - `GET /b2b/dashboard/accounts-in-motion`
  - `GET /b2b/dashboard/accounts-in-motion/live`

### Existing frontend building blocks

Vendor overview surface:

- [Dashboard.tsx](/home/juan-canfield/Desktop/Atlas/atlas-churn-ui/src/pages/Dashboard.tsx)

Vendor table:

- [Vendors.tsx](/home/juan-canfield/Desktop/Atlas/atlas-churn-ui/src/pages/Vendors.tsx)

Existing client hooks:

- [client.ts](/home/juan-canfield/Desktop/Atlas/atlas-churn-ui/src/api/client.ts)
  - `fetchSignals(...)`
  - `fetchSlowBurnWatchlist(...)`
  - `fetchHighIntent(...)`

### Important implication

The product is not greenfield.

What exists now:

- tracked-vendor CRUD
- vendor-level slow-burn feed
- vendor detail
- per-vendor accounts-in-motion

What does not exist yet as a unified product:

- one tenant-wide movement feed across all watched vendors
- one coherent watchlist page tying vendor movement to account-level opportunity cards
- embedded evidence drawer for trust and actionability

---

## Gaps To Close

### Gap 1: Feed shape is vendor-level, not movement-card-level

Current `slow-burn-watchlist` returns vendor summary rows.

That is useful, but the benchmark product needs actual movement cards, not just ranked vendors.

Need:

- vendor-level feed rows
- plus account-level movement rows where persisted accounts-in-motion exists

### Gap 2: Accounts-in-motion is only per vendor

Current accounts-in-motion API requires `vendor_name`.

Need a tenant-scoped aggregation path so the feed can show the top account movement across all tracked vendors without N client round-trips.

### Gap 3: Trust layer is fragmented

Current data surfaces expose:

- vendor summaries
- account lists
- report details

But the benchmark product needs a consistent evidence drawer that can appear from feed cards and account rows.

### Gap 4: Freshness and coverage need to be explicit

The feed must tell users:

- when the signal was last computed
- whether it comes from persisted report vs live review fallback
- whether evidence is thin or strong

Without those markers, the product risks overclaiming precision.

---

## Product Scope

### V1 in scope

1. Watchlist home page
2. Saved vendor watchlists
3. Vendor movement feed using tracked-vendor and slow-burn endpoints
4. Accounts-in-motion section using persisted accounts-in-motion data
5. Evidence drawer from any row/card
6. Freshness and confidence markers
7. Saved filters and alert thresholds

### V1 out of scope

1. Fresh synthesis at read time
2. Live cross-vendor recomputation
3. Full CRM sync
4. Full campaign generation from inside the feed
5. Public SEO packaging
6. Arbitrary account lookup beyond existing data coverage

---

## Recommended Implementation Shape

### Backend

#### Reuse unchanged

- tracked vendor CRUD
- `GET /b2b/tenant/slow-burn-watchlist`
- `GET /b2b/tenant/signals/{vendor_name}`
- `GET /b2b/dashboard/accounts-in-motion?vendor_name=...`

#### Add

1. Tenant-scoped accounts-in-motion feed endpoint

Suggested shape:

- `GET /b2b/tenant/accounts-in-motion-feed`

Behavior:

- scopes to tracked vendors for the current account
- pulls latest persisted accounts-in-motion artifacts for those vendors
- returns top account movement cards across the watchlist
- includes data source and freshness markers

2. Feed item evidence payload

Either:

- extend the new feed endpoint with compact evidence preview fields

or:

- add a small detail endpoint for card drilldown

Suggested shape:

- `GET /b2b/tenant/feed-item-detail`

This should return only stored evidence and enrichment details, not trigger synthesis.

#### Data contract goals

Every movement/feed row should include:

- vendor name
- product category
- urgency
- last seen or last computed timestamp
- data source
- top quote or witness excerpt
- buying stage when available
- contract timing when available
- confidence or evidence coverage posture
- link target for vendor detail or account detail

### Frontend

#### Preferred route strategy

Do not create a disconnected new shell.

Prefer either:

1. a dedicated `Watchlists` page added to the current churn UI

or:

2. evolve `Dashboard` into a real watchlist home and keep `Vendors` as the exploration table

Recommended:

- new page: `Watchlists.tsx`
- keep `Dashboard` for broad summary
- make `Watchlists` the habitual working surface

#### V1 page sections

1. Watchlist summary bar
   - watched vendors
   - high-urgency vendors
   - active accounts in motion
   - freshest update time

2. Vendor movement feed
   - one card per tracked vendor movement
   - urgency, archetype/wedge, freshness, summary evidence

3. Accounts in motion
   - ranked table across tracked vendors
   - top company opportunities

4. Evidence drawer
   - top quote
   - source
   - timing signal
   - witness and reasoning references where available

5. Saved filters
   - category
   - urgency threshold
   - vendor subset
   - source subset

---

## Build Phases

### Phase 1: Productize what already exists

Goal:

- assemble a usable watchlist page with existing vendor feed + existing per-vendor details

Work:

- add `Watchlists` page
- add tracked-vendor list fetch and CRUD wiring
- render `slow-burn-watchlist`
- deep-link to vendor detail
- add freshness markers

Success:

- users can manage tracked vendors and monitor them from one page

### Phase 2: Add tenant-wide accounts-in-motion aggregation

Goal:

- show account movement across the whole watched portfolio

Work:

- add tenant-scoped aggregated accounts-in-motion endpoint
- render account movement table under the vendor feed
- support filters and top-N ranking

Success:

- users no longer need one API call per vendor to see the top moving accounts

### Phase 3: Add evidence drawer and trust posture

Goal:

- make each signal inspectable

Work:

- expose evidence preview fields or lightweight card-detail endpoint
- add evidence drawer component
- show source, quote, timing, and trust markers

Success:

- every important card/row can answer "why is this here?"

### Phase 4: Alerts and saved views

Goal:

- make the product habitual

Work:

- saved filters
- urgency thresholds
- watchlist-level notification preferences

Success:

- product becomes a recurring operational feed rather than a static dashboard

---

## Risks

### Product risk

If the page leans too hard on named accounts or inferred buyer identity, users will over-trust sparse fields.

Mitigation:

- confidence-tiered UI
- clear evidence posture
- explicit freshness/source markers

### Technical risk

Naively aggregating accounts-in-motion client-side will create too many round-trips across tracked vendors.

Mitigation:

- add one tenant-scoped aggregation endpoint

### Trust risk

If evidence drilldown is postponed, the product will look like a black-box alert board.

Mitigation:

- treat evidence drawer as required v1 trust infrastructure

---

## Recommended Next Implementation Order

1. Add product logging docs and freeze benchmark
2. Add `Watchlists` UI route and basic page shell
3. Wire tracked vendor CRUD + `slow-burn-watchlist`
4. Add tenant-scoped accounts-in-motion feed endpoint
5. Add evidence drawer and confidence/freshness markers

---

## Success Metrics

Initial product success should be measured by:

- watchlist creation rate
- repeat weekly usage of the watchlist page
- clickthrough from movement cards to evidence/details
- number of account movement cards reviewed
- number of vendors with fresh monitored data
- internal analyst trust in the feed

Do not use:

- raw model-call volume
- page views alone

The product is successful when it becomes a trusted daily operating surface.
