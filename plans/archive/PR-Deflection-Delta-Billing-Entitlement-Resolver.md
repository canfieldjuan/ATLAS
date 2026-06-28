# PR-Deflection-Delta-Billing-Entitlement-Resolver

## Why this slice exists

#1873 replaces the monthly deflection delta config allowlist with Stripe Billing subscription entitlements. Root cause: `content_ops_deflection_delta_automation` currently treats `ATLAS_DEFLECTION_DELTA_ENTITLED_ACCOUNT_IDS` as the entitlement source of truth, so there is no durable account-level Billing entitlement record for Stripe webhook lifecycle handlers to grant/revoke later. This fixes the upstream seam for the next webhook slice by adding a DB-backed entitlement table/resolver and routing delta generation/delivery through that resolver.

This PR fixes the root storage/resolution gap, but it intentionally does not claim full #1873 completion. The Stripe webhook writer is deferred so the first slice stays reviewable and so the resolver can be proven fail-closed before webhook events mutate it.

Diff-size note: this is over the 400 LOC soft target because the safe first slice needs the migration, the real store-adapter contract, task wiring, adapter tests, task fail-closed tests, and the real-DB migration apply guard in one reviewable contract.

## Scope (this PR)

Ownership lane: deflection/delta-billing-entitlements
Slice phase: Production hardening

1. Add a `content_ops_deflection_delta_entitlements` migration for account-level monthly delta subscription entitlement records.
2. Add the Billing-backed entitlement resolver to the real deflection report store adapters, returning only active/trialing, non-revoked entitlement accounts with the existing config allowlist as a rollout fallback for accounts with no Billing row yet.
3. Route delta automation generation, delivery drain, and pending-count paths through the Postgres store adapter instead of parsing the env allowlist inline.
4. Prove active/trialing rows allow delivery, canceled/past_due/unpaid rows do not, and blank DB+config state still fails closed.

### Review Contract

- Delta automation must never scan globally when both DB entitlements and config fallback are empty.
- Active/trialing, non-revoked Billing entitlement rows must feed the existing `entitled_account_ids` filters for compute, drain, and pending-count paths.
- Inactive Billing statuses (`canceled`, `past_due`, `unpaid`, `incomplete`, `incomplete_expired`, `paused`) must not grant delivery.
- The existing config allowlist remains fallback-only and is explicitly temporary for rollout safety; it must not revive an account with a known inactive/revoked Billing row.
- Entitlement resolution must be bounded by the task account scan window.
- No Stripe Checkout or webhook lifecycle mutation is added in this slice.
- Reviewer rules triggered: R1, R2, R4, R6, R8, R10, R14.

### Files touched

- `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py`
- `atlas_brain/storage/migrations/343_content_ops_deflection_delta_entitlements.sql`
- `extracted_content_pipeline/deflection_report_access.py`
- `plans/PR-Deflection-Delta-Billing-Entitlement-Resolver.md`
- `tests/maturity_sweep/deflection_product_surface_manifest.json`
- `tests/test_content_ops_deflection_delta_persistence.py`
- `tests/test_deflection_delta_automation_task.py`
- `tests/test_deflection_migrations_apply.py`

## Mechanism

Migration 343 creates `content_ops_deflection_delta_entitlements` with one row per Stripe subscription/account mapping, constrained subscription status values, and an active-entitlement index that excludes soft-revoked rows. `DeflectionReportArtifactStore` grows a monthly delta entitlement-listing method, implemented by both `InMemoryDeflectionReportArtifactStore` and `PostgresDeflectionReportArtifactStore`, so entitlement reads stay in the same adapter layer as report/delta persistence.

`content_ops_deflection_delta_automation.run()` creates the Postgres store once, asks it for entitled accounts bounded by the task account limit, and then passes that tuple through the existing compute/delivery filters. The resolver returns active/trialing non-revoked Billing accounts plus fallback accounts only when that account has no Billing row yet, so canceled/unpaid/revoked rows cannot be revived by config fallback.

## Intentional

- `active` and `trialing` are the only granting statuses in this slice. `past_due`/`unpaid` are fail-closed until product explicitly chooses a grace policy.
- The config allowlist stays as fallback to avoid breaking the MVP before Stripe webhook lifecycle sync exists; it is not treated as the durable source of truth and cannot override known inactive/revoked Billing rows.
- No `payment_method_types`, Checkout Session creation, or subscription Price wiring is added here; that belongs with the customer-facing subscription purchase slice.

## Deferred

- Follow-up #1873 slice: Stripe webhook lifecycle handler upserts/revokes `content_ops_deflection_delta_entitlements` for the monthly delta Price IDs.
- Follow-up #1873 slice: subscription Checkout Session / customer portal surface once monthly product and Price IDs are finalized.

Parked hardening: none.

## Verification

- PASS: focused pytest bundle covering the delta automation task, deflection delta persistence adapters, deflection migration apply guard, and product-surface manifest (71 passed, 2 skipped).
- PASS: Python compile check for the delta automation task and deflection report access adapter modules.
- PASS: repo local PR review bundle.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/autonomous/tasks/content_ops_deflection_delta_automation.py` | 20 |
| `atlas_brain/storage/migrations/343_content_ops_deflection_delta_entitlements.sql` | 38 |
| `extracted_content_pipeline/deflection_report_access.py` | 106 |
| `plans/PR-Deflection-Delta-Billing-Entitlement-Resolver.md` | 77 |
| `tests/maturity_sweep/deflection_product_surface_manifest.json` | 1 |
| `tests/test_content_ops_deflection_delta_persistence.py` | 126 |
| `tests/test_deflection_delta_automation_task.py` | 192 |
| `tests/test_deflection_migrations_apply.py` | 85 |
| **Total** | **645** |
