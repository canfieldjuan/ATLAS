# PR: Content Ops Tenant Usage Summary

## Why this slice exists

Content Ops hosted generation now writes usage rows, and those rows carry
trusted account_id/user_id metadata from the authenticated execution scope. The
operator-only usage summary can show global spend, but tenants still do not have
a scoped read path for their own Content Ops usage.

This slice adds the thin tenant read path before UI cards, budget gates, or
cache controls build on top of usage data.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Production hardening

1. Add an account_id filter to the shared Content Ops usage summary query.
2. Add a tenant-scoped usage route that resolves the authenticated
   TenantScope.account_id and filters at SQL time.
3. Keep the existing operator /usage/summary route global and
   platform-admin-only.
4. Add focused tests for account filtering and hosted route wiring.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Tenant-Usage-Summary.md` | Plan doc for tenant-scoped usage summary. |
| `extracted_content_pipeline/content_ops_usage_summary.py` | Add account_id filtering to the shared usage summary query. |
| `extracted_content_pipeline/api/control_surfaces.py` | Add tenant-scoped usage summary route using the resolved scope. |
| `tests/test_extracted_content_ops_usage_summary.py` | Pin SQL-level account filtering behavior. |
| `tests/test_atlas_content_ops_generated_assets_api.py` | Pin hosted tenant usage route auth/scope/pool wiring. |

## Mechanism

summarize_content_ops_llm_usage accepts an optional account_id and includes
metadata ->> 'account_id' in the SQL WHERE clause when present. The new
/usage/summary/tenant route resolves the same usage pool as the operator route,
then resolves the request scope from the host scope provider. If no account_id
is available, the route returns 400 instead of falling back to global data.

The existing /usage/summary route is unchanged and keeps the operator dependency
for global spend analysis.

## Intentional

- This does not add frontend cards yet. It creates the tenant-safe API surface
  those cards can call.
- This does not add budget enforcement. BudgetGate should consume this scoped
  usage once the read path is stable.
- This does not wire exact-cache controls. Cache policy needs a separate slice
  because it has support-ticket privacy and account-scoping implications.
- The tenant route filters in SQL rather than filtering the returned payload so
  cross-account rows never enter the tenant summary calculation.

## Deferred

- Future PR: add UI/control-surface cards for tenant usage.
- Future PR: wire BudgetGate against account-scoped usage.
- Future PR: add exact-cache integration with explicit support-ticket privacy
  policy and account scoping.
- Parked hardening: none planned.

## Verification

- python -m pytest tests/test_extracted_content_ops_usage_summary.py tests/test_atlas_content_ops_generated_assets_api.py -q — 10 passed, 1 skipped, 1 warning.
- python -m compileall -q extracted_content_pipeline/content_ops_usage_summary.py extracted_content_pipeline/api/control_surfaces.py tests/test_extracted_content_ops_usage_summary.py tests/test_atlas_content_ops_generated_assets_api.py — passed.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <body> — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~75 |
| Usage summary query | ~20 |
| Control-surface route | ~30 |
| Tests | ~80 |
| **Total** | **~205** |

Under the 400 LOC soft cap.
