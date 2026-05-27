# PR: Content Ops Cache Policy UI

## Why this slice exists

PR-Content-Ops-Cache-Policy-Request added the backend
`content_ops_cache_policy` contract and routed it into the existing Content Ops
exact-cache policy. The New Run UI still cannot set that field, so operators
cannot intentionally request exact cache or explicit no-store behavior from the
product surface.

This slice closes the UI half of that contract without changing backend cache
behavior.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add frontend wire/domain support for `content_ops_cache_policy`.
2. Preserve the field through request mapping and preview normalized-request
   mapping.
3. Add a compact cache-policy select to the existing New Run Options panel.
4. Keep the default safe: no explicit cache request unless the operator selects
   one.
5. Add a focused Node test for request round-trip, preview mapping, and UI
   control wiring.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Policy-UI.md` | Plan doc for the frontend cache-policy controls. |
| `atlas-intel-ui/package.json` | Add a focused cache-policy UI contract test script. |
| `atlas-intel-ui/scripts/content-ops-cache-policy-ui.test.mjs` | Cover request mapping, preview mapping, and New Run control wiring. |
| `atlas-intel-ui/src/api/contentOps.ts` | Add wire request field. |
| `atlas-intel-ui/src/domain/contentOps/types.ts` | Add cache policy domain type and request field. |
| `atlas-intel-ui/src/domain/contentOps/fromWire.ts` | Map cache policy in request and preview normalized-request flows. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Export the cache policy domain type. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Add the Options-panel cache policy select. |

## Mechanism

The frontend keeps the backend's normalized values: `exact`, `no_store`, or
`null`. `fromWireRequest` and `fromWirePreview` read
`content_ops_cache_policy`; `toWireRequest` sends the camelCase domain field
back as the backend snake-case field.

`ContentOpsNewRun` renders a select in the existing Options grid. The blank
option leaves `contentOpsCachePolicy` as `null`, preserving current behavior.
Choosing no-store or exact sends the explicit backend value through the same
preview/plan/execute request object.

## Intentional

- This does not change backend cache eligibility or support-ticket/customer-data
  no-store rules.
- This does not add tenant-level persisted defaults.
- This does not expose namespace or customer-data cache toggles. The UI only
  sends the existing safe request policy field.
- This keeps the control in the existing Options panel instead of adding a new
  section.

## Deferred

- Future PR: persisted tenant cache defaults if operators need always-on
  posture instead of per-run choices.
- Future PR: richer cache diagnostics if operators need to see why a request was
  no-store after execution.
- Parked hardening: none planned.

## Verification

- npm --prefix atlas-intel-ui run test:content-ops-cache-policy-ui — 4 passed.
- npm --prefix atlas-intel-ui run build — passed.
- npm --prefix atlas-intel-ui run lint — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file /tmp/atlas_pr_cache_policy_ui_body.md — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~95 |
| API/domain mapping | ~20 |
| UI control | ~30 |
| Test/package hook | ~85 |
| **Total** | **~240** |
