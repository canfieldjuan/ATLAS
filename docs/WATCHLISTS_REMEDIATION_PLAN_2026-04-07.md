# Watchlists Remediation Plan

Date: 2026-04-07
Surface: Vendor Intelligence Watchlists and Accounts-in-Motion Feed
Primary UI: `atlas-churn-ui/src/pages/Watchlists.tsx`

## Scope

This plan turns the current watchlists audit into an execution order.

The goal is not to patch cosmetic gaps. The goal is to make the watchlists
surface production-safe, evidence-backed, and aligned with the intended Churn
Signals product:

- saved watchlists
- ranked vendor movement feed
- accounts-in-motion feed
- evidence-backed event cards
- freshness markers
- confidence-tiered account presentation
- practical filtering and saved views

## Current State

What is already in place:

- tracked vendor CRUD
- watchlists route and sidebar entry
- slow-burn vendor feed
- tenant-wide persisted accounts-in-motion aggregation
- competitive sets controls
- backend test coverage for tenant feed aggregation

What is not yet complete:

- frontend build is currently broken
- no inline evidence drawer for account rows
- account confidence is not surfaced
- filter and saved-view experience is missing
- freshness is inconsistent across sections
- account feed is backed by per-vendor report fan-out, which is acceptable for
  early tenants but not the final scaling shape

## Phase 0: Ship Blockers

### 0.1 Restore Green Frontend Build

Problem:

- `atlas-churn-ui` does not compile today because of unused-symbol TypeScript
  errors in unrelated shared report files.

Files:

- `atlas-churn-ui/src/components/ReportActionBar.tsx`
- `atlas-churn-ui/src/pages/ReportDetail.tsx`

Required change:

- remove or use dead imports/state so `npm --prefix atlas-churn-ui run build`
  succeeds again

Acceptance:

- root Vercel build path passes
- local `npm --prefix atlas-churn-ui run build` passes
- no watchlists work ships on top of a broken compile

### 0.2 Add Watchlists Page Test Coverage

Problem:

- the backend feed has tests, but the page itself has no test file
- current regression escaped from shared frontend code proves page-level and
  route-level safety is too weak

Files:

- add `atlas-churn-ui/src/pages/Watchlists.test.tsx`

Required coverage:

- initial load fetches tracked vendors, slow-burn feed, and accounts feed
- empty states
- row navigation
- stale report label rendering
- confidence-tiered account rendering once added

Acceptance:

- watchlists page has direct component coverage
- shared-report regressions do not silently block deploys again

## Phase 1: Trust Contract

### 1.1 Expand Accounts-in-Motion Feed Contract

Problem:

- the current feed only carries one short evidence string and drops the witness
  trail that makes this product defensible

Current shaping path:

- `atlas_brain/api/b2b_dashboard.py`
- `atlas_brain/api/b2b_tenant_dashboard.py`
- `atlas-churn-ui/src/api/client.ts`

Required change:

- extend the persisted row shaping contract for watchlist consumption
- preserve account-level evidence metadata instead of collapsing to
  `top_quote` only

Target additions:

- `review_ids`
- `witness_ids`
- `reference_ids`
- `evidence_items[]`
- `evidence_count`
- `quote_match_type`
- `confidence_band`
- `source_distribution`
- `report_date`
- `stale_days`
- `is_stale`

Important rule:

- do not synthesize new evidence at read time
- read only from persisted accounts-in-motion artifacts and their linked
  evidence metadata

Acceptance:

- a watchlist account row can answer "why does this exist?" without redirecting
  to another page

### 1.2 Add Inline Evidence Drawer

Problem:

- current account row click only navigates to vendor detail
- the watchlists page itself states that the embedded trust layer is still
  missing

Files:

- `atlas-churn-ui/src/pages/Watchlists.tsx`
- likely shared UI support under `atlas-churn-ui/src/components/`
- optional reuse of `atlas-churn-ui/src/pages/EvidenceExplorer.tsx` patterns

Required change:

- clicking an account row should open a drawer or side panel
- drawer should show:
  - account name
  - vendor
  - urgency
  - confidence
  - evidence quotes
  - source mix
  - freshness
  - witness/reference links where available

Acceptance:

- watchlists becomes a real inspection surface instead of a redirect table

### 1.3 Confidence-Tier Account Presentation

Problem:

- unknown and low-confidence accounts are shown almost the same as strong rows

Files:

- `atlas-churn-ui/src/pages/Watchlists.tsx`
- `atlas-churn-ui/src/api/client.ts`

Required change:

- render confidence explicitly on account rows
- visually distinguish:
  - named high-confidence rows
  - named medium-confidence rows
  - anonymous or low-confidence rows
- stop presenting `Unknown account` rows as equivalent to named-account claims

Recommended rule:

- anonymous rows remain visible but clearly labeled as anonymous signal clusters,
  not specific accounts

Acceptance:

- page presentation matches the stated product constraint around named-account
  certainty

## Phase 2: Product Controls

### 2.1 Add Feed Filters

Problem:

- the current page fetches both feeds with default params only
- target product shape calls for filters by vendor, category, and source

Backend:

- slow-burn feed already supports `vendor_name` and `category`
- accounts-in-motion feed currently supports only `min_urgency`,
  `per_vendor_limit`, and `limit`

Required change:

- add watchlists controls for:
  - vendor
  - category
  - urgency threshold
  - source
  - stale/fresh only

Backend work:

- extend `/accounts-in-motion-feed` to support at least:
  - `vendor_name`
  - `category`
  - `source`
  - `include_stale`

Acceptance:

- users can narrow the feed without leaving the watchlists surface

### 2.2 Add Saved Views

Problem:

- no saved views exist today, despite being part of the intended product shape

Required change:

- persist named view presets per account
- preset fields should include:
  - selected vendors
  - category
  - urgency threshold
  - source filter
  - stale/fresh mode

Suggested backend:

- create a dedicated tenant-scoped saved-view table rather than overloading
  tracked vendors or report subscriptions

Acceptance:

- watchlists can support habitual use instead of one transient default view

### 2.3 Add Alert Thresholding

Problem:

- today there is no watchlist-specific threshold state beyond raw feed data

Required change:

- allow a view or watchlist to define thresholds such as:
  - minimum urgency
  - new named accounts only
  - only changed wedges
  - only non-stale reports

Important boundary:

- alert delivery does not have to ship first
- threshold persistence and filtering should ship before notifications

Acceptance:

- the product supports "show me what matters" before it tries to notify

## Phase 3: Freshness Contract

### 3.1 Normalize Freshness Handling

Problem:

- tracked vendors use a page-local `24h` freshness rule
- accounts-in-motion already has backend `report_date`, `stale_days`,
  and `is_stale`

Files:

- `atlas-churn-ui/src/pages/Watchlists.tsx`
- `atlas_brain/api/b2b_tenant_dashboard.py`

Required change:

- move freshness semantics to backend-backed fields where possible
- tracked vendors should expose:
  - latest snapshot timestamp
  - latest reasoning timestamp
  - latest report timestamp or freshness status

Acceptance:

- one freshness model across watchlists sections
- no UI-only freshness heuristics for backend-derived rows

## Phase 4: Scaling and Query Shape

### 4.1 Replace Per-Vendor Fan-Out for Larger Tenant Watchlists

Problem:

- `/accounts-in-motion-feed` currently does one persisted-report read per
  tracked vendor
- that is acceptable for small watchlists, but not the final scaling model

Current path:

- `atlas_brain/api/b2b_tenant_dashboard.py`

Required change:

- keep current behavior for now if needed
- plan a second-generation query path that reads the newest scoped
  `accounts_in_motion` artifacts in one backend pass

Acceptance:

- tenant watchlists do not degrade linearly with tracked vendor count

## Build Order

1. Phase 0.1 Restore green frontend build
2. Phase 0.2 Add watchlists page tests
3. Phase 1.1 Expand accounts-in-motion feed contract
4. Phase 1.2 Add evidence drawer
5. Phase 1.3 Add confidence-tier account treatment
6. Phase 2.1 Add filters
7. Phase 2.2 Add saved views
8. Phase 2.3 Add alert thresholds
9. Phase 3.1 Normalize freshness contract
10. Phase 4.1 Replace per-vendor feed fan-out when scale requires it

## Validation Checklist

Backend:

- `pytest tests/test_b2b_tenant_data_freshness.py tests/test_b2b_dashboard_accounts_in_motion.py -q`
- add new tests for filter params and evidence contract expansion

Frontend:

- `npm --prefix atlas-churn-ui run build`
- add watchlists page test coverage

Integration:

- verify `/watchlists` renders tracked vendors, vendor feed, and account feed
- verify account row opens evidence drawer
- verify anonymous/low-confidence rows are clearly downgraded
- verify filters affect both watchlist feeds correctly
- verify stale markers are consistent across sections

## Non-Goals

- no live read-time re-synthesis
- no account certainty inflation
- no workaround that hides weak rows without surfacing confidence
- no notification system before thresholding and saved views exist
