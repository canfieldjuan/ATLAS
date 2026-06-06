# PR: Content Ops Usage Cards

## Why this slice exists

Content Ops now records hosted LLM usage and exposes a tenant-scoped
`/content-ops/usage/summary/tenant` read path. The next product-facing step is
to show that usage inside the Content Ops run screen so operators can see recent
spend, calls, failures, and cache signals before they run more generation.

This keeps the cost/caching lane source-first: record real usage, expose a
tenant-safe API, then surface the current account's usage in the UI before
adding budget gates or cache controls.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add typed frontend API support for the tenant usage summary route.
2. Add domain types and a wire-to-domain mapper for the usage summary payload.
3. Show a compact read-only 7-day usage card on the Content Ops new-run page.
4. Add a focused Node test that pins the route URL/query contract and mapper
   shape.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Usage-Cards.md` | Plan doc for tenant usage UI cards. |
| `atlas-intel-ui/package.json` | Add a focused usage-summary test script. |
| `atlas-intel-ui/scripts/content-ops-usage-summary.test.mjs` | Cover the API route/query contract and usage-summary mapper. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add tenant usage summary wire types and fetch wrapper. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add camelCase domain types for usage summary cards. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map usage summary wire payloads into domain shape. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Export usage summary types and mapper from the domain barrel. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Fetch and render the 7-day usage summary card. |

## Mechanism

The API adapter adds `fetchContentOpsTenantUsageSummary`, which calls
`GET /api/v1/content-ops/usage/summary/tenant` with optional query params for
`days`, `asset_type`, `run_id`, and `request_id`. The response stays in snake
case at the API layer.

The domain mapper translates the response into camelCase, copies breakdown
arrays, and preserves nullable fields such as `latest_call_at`. The new-run page
uses the existing `useApiData` hook to fetch the 7-day tenant summary separately
from the catalog so a usage-card error does not block the run form.

The UI card is read-only and intentionally compact: total spend, calls,
failures, cache-hit calls, token totals, latest call time, and the top asset-type
breakdown. Budget enforcement and cache controls remain separate slices.

## Intentional

- This does not add budget gates. The card is visibility only.
- This does not call the operator-only global usage route. Tenant UI must use
  the account-scoped route.
- This does not expose prompts, responses, or support-ticket bodies. The route
  returns aggregate usage only.
- This does not add charting. The first UI surface should stay compact until we
  know which usage dimensions matter most.

## Deferred

- Future PR: wire BudgetGate against account-scoped Content Ops usage.
- Future PR: add exact-cache controls with explicit support-ticket privacy
  policy and account scoping.
- Future PR: add deeper usage drilldowns if operators need run/model/detail
  filters in the UI.
- Parked hardening: none planned.

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-usage-summary — 2 passed.
- npm --prefix atlas-intel-ui run lint — passed.
- npm --prefix atlas-intel-ui run build — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| API wire types/fetcher | ~65 |
| Domain types/mapper/barrel | ~105 |
| UI card | ~120 |
| Test script/package hook | ~160 |
| **Total** | **~540** |

This is over the 400 LOC soft cap because the UI needs a typed route wrapper,
domain mapping, visible rendering, and a focused executable test in the same
slice to prove the end-to-end frontend contract.
