# PR: Content Ops Usage Budget UI

## Why this slice exists

PR-Content-Ops-Usage-Budget-Gate added a load-bearing backend gate for
`account_usage_budget_usd` and `account_usage_budget_days`, but the new-run UI
still cannot set those fields or show the account-period budget evaluation the
backend returns.

This slice closes the product-facing part of that contract: operators can add a
per-request account usage budget before previewing, see the backend's projected
usage verdict, and send the same fields through preview, plan, and execute.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add frontend wire/domain support for account usage budget request fields and
   preview `usage_budget` responses.
2. Add new-run form controls for the account budget cap and budget window.
3. Validate the budget inputs client-side before preview/plan/execute.
4. Render the backend budget evaluation in the preview verdict when present.
5. Render the refreshed plan-time budget evaluation when `/plan` returns one.
6. Add a focused Node test for the request mapper, default budget fields,
   preview budget evaluation mapper, and plan-panel budget verdict wiring.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Usage-Budget-UI.md` | Plan doc for the account usage budget UI slice. |
| `atlas-intel-ui/package.json` | Add a focused usage-budget UI contract test script. |
| `atlas-intel-ui/scripts/content-ops-usage-budget-ui.test.mjs` | Cover request mapping, default budget fields, preview budget evaluation mapping, and plan-panel budget verdict wiring. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add wire fields for account usage budgets and preview budget evaluation. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add camelCase budget request/evaluation domain fields. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map account budget fields both directions and preview evaluation responses. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Export the budget evaluation domain type. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Add budget inputs, validation, request wiring, and preview/plan budget rendering. |

## Mechanism

The API adapter keeps the backend contract in snake case:
`account_usage_budget_usd`, `account_usage_budget_days`, and preview
`usage_budget`. The domain mapper translates those to `accountUsageBudgetUsd`,
`accountUsageBudgetDays`, and `usageBudget`, while `toWireRequest` sends them
back through the same request body used by preview, plan, and execute.

`ContentOpsNewRun` keeps the account budget USD and days controls as string
drafts, matching the existing max-cost input pattern so decimal entry is not
coerced mid-keystroke. Submit-time parsing allows a blank budget cap, requires a
positive cap when supplied, and keeps the days value inside the backend's
1-90-day range. The preview verdict shows current usage, estimated run cost,
projected usage, and budget cap when the backend includes an evaluation.

The focused Node test transpiles the existing TypeScript modules directly. It
pins mapper behavior so the new request fields and preview evaluation cannot
silently drop at the frontend contract layer. It also pins that the plan panel
uses `plan.preview.usageBudget`, because `/plan` re-evaluates account usage and
can return a fresher budget verdict than the earlier preview response.

## Intentional

- This does not add persisted per-account default budgets. The backend contract
  is still explicit per request.
- This does not change the existing `max_cost_usd` per-run cap. The new fields
  are account-period budget controls.
- This does not add charting or model-level budget controls. The UI only shows
  the backend's current budget verdict.
- This does not add exact-cache controls. That stays in the cost/caching lane
  after the usage and budget surfaces are proven.

## Deferred

- Future PR: persisted per-account budget defaults if operators need an
  always-on tenant cap instead of per-request caps.
- Future PR: exact-cache controls with support-ticket privacy policy and
  account scoping.
- Future PR: richer usage drilldowns if the compact usage card and budget
  verdict are not enough for operators.
- Parked hardening: none planned.

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-usage-budget-ui — 4 passed.
- npm --prefix atlas-intel-ui run build — passed.
- npm --prefix atlas-intel-ui run lint — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| API/domain type and mapper updates | ~70 |
| New-run UI controls and preview rendering | ~120 |
| Focused test script/package hook | ~140 |
| **Total** | **~425** |

This is slightly over the 400 LOC soft cap because the frontend request,
response, visible controls, and executable contract test need to land together
for the budget UI path to be usable end to end.
