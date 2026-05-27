# PR: Content Ops Cache Savings UI

## Why this slice exists

PR-Content-Ops-Cache-Savings-Rollup added `total_cache_savings_usd` to the
tenant usage summary backend. The Content Ops run screen still only shows
spend, calls, failures, cache hits, and tokens, so operators cannot see the
actual avoided spend from exact-cache hits.

This slice completes the small product-facing loop: the existing usage card
should display the backend-provided savings value without recomputing it in the
frontend.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add cache-savings fields to the Content Ops usage summary API/domain types.
2. Map `summary.total_cache_savings_usd` and breakdown `cache_savings_usd`.
3. Add a compact "Saved" metric to the existing 7-day usage card.
4. Extend the existing usage-summary UI contract test.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Savings-UI.md` | Plan doc for the UI display slice. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add cache-savings wire fields to usage summary types. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add cache-savings fields to domain types. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map cache-savings fields from wire to domain. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Show a 7-day saved-spend metric in the usage card. |
| `atlas-intel-ui/scripts/content-ops-usage-summary.test.mjs` | Pin the mapper contract for cache savings. |

## Mechanism

The backend already returns:

- `summary.total_cache_savings_usd`
- `by_model[].cache_savings_usd`
- `by_asset_type[].cache_savings_usd`

The frontend API shape and domain mapper now carry those fields into
camelCase domain names. `UsageSummaryCard` renders `summary.totalCacheSavingsUsd`
with the existing `formatUsd(...)` helper as a sixth metric in the existing
grid.

## Intentional

- This does not recompute savings in the UI; the backend owns pricing and
  savings logic.
- This does not add deeper cache drilldowns yet. The first UI step is a compact
  top-level saved-spend number beside spend and cache hits.
- This keeps the existing usage card and route; no new frontend fetch path.

## Deferred

- Future PR: add model/asset savings drilldowns if the compact usage card is
  not enough.
- Future PR: decide whether savings should be highlighted differently once real
  production traffic produces nontrivial values.
- Parked hardening: none. Root `HARDENING.md` was scanned; no cost-surfacing
  items are parked.

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-usage-summary — 2 passed.
- npm --prefix atlas-intel-ui run build — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| API/domain mapper fields | ~20 |
| Usage card display | ~10 |
| Test fixture/assertions | ~10 |
| **Total** | **~120** |

This stays below the 400 LOC soft cap.
