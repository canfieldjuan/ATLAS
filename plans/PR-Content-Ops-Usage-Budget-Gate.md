# PR: Content Ops Usage Budget Gate

## Why this slice exists

Content Ops now records hosted LLM usage, exposes tenant-scoped usage summaries,
and shows recent usage in the run UI. The next production step is a load-bearing
budget gate that uses the same account-scoped usage data before generation runs.

This slice keeps the old `max_cost_usd` request cap intact as a per-run estimate
cap, and adds a separate account-period budget contract for cumulative tenant
spend.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add explicit `account_usage_budget_usd` and `account_usage_budget_days`
   request fields.
2. Add pure budget evaluation against current account spend plus the estimated
   request cost.
3. Wire the budget evaluation into `/preview`, `/plan`, and `/execute` when the
   account budget field is provided.
4. Keep budget enforcement fail-closed when the usage pool or tenant account
   scope is unavailable.
5. Add focused tests for request validation, preview/plan response shaping, and
   execute blocking before generation services run.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Usage-Budget-Gate.md` | Plan doc for the usage-backed budget gate. |
| `extracted_content_pipeline/control_surfaces.py` | Add account-period budget fields and pure budget evaluation. |
| `extracted_content_pipeline/api/control_surfaces.py` | Apply account-scoped usage budget checks to preview, plan, and execute routes. |
| `tests/test_extracted_content_control_surfaces.py` | Cover request validation and budget evaluation. |
| `tests/test_extracted_content_control_surface_api.py` | Cover API preview/plan/execute budget gate wiring. |

## Mechanism

`ContentOpsRequest` gains `account_usage_budget_usd` and
`account_usage_budget_days`. `max_cost_usd` remains a per-request estimate cap;
the new fields mean "current account usage over N days plus this request's
estimated cost must stay under this account-period cap."

The API layer reads the current tenant usage through
`summarize_content_ops_llm_usage` with the authenticated scope account_id. It
then evaluates projected spend with the pure control-surface helper and attaches
an `usage_budget` payload to preview/plan output. If projected spend exceeds the
budget, preview returns `can_run: false`, plan returns `can_execute: false`, and
execute raises a 400 before resolving or calling generation services.

## Intentional

- This does not overload `max_cost_usd`; existing per-run behavior remains.
- This does not add UI controls yet. The backend contract lands first so the
  UI can call a stable field name in a follow-up.
- This does not update frontend request controls or TypeScript domain fields in
  this slice. The backend accepts and enforces the new fields now; the UI
  control surface is the named follow-up.
- This only enforces when `account_usage_budget_usd` is supplied. Accounts with
  no explicit budget keep today's behavior.
- The execute route checks the budget before LLM generation, not after usage is
  already spent.

## Deferred

- Future PR: expose account usage budget controls in the Content Ops run UI.
- Future PR: persist per-account default budget settings if product needs
  always-on tenant caps instead of per-request caps.
- Future PR: add exact-cache controls with explicit support-ticket privacy
  policy and account scoping.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py::test_preview_generation_route_blocks_account_usage_budget tests/test_extracted_content_control_surface_api.py::test_plan_generation_route_marks_steps_blocked_when_account_budget_exceeds tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_blocks_account_usage_budget_before_generation -q — 35 passed.
- python -m compileall -q extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_control_surfaces.py tests/test_extracted_content_control_surface_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Control-surface request/evaluation | ~85 |
| API route wiring | ~140 |
| Tests | ~140 |
| **Total** | **~455** |

This is over the 400 LOC soft cap because preview, plan, and execute all need to
use the same budget contract in one slice for the gate to be load-bearing.
