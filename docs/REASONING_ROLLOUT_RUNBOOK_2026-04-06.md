# Reasoning Rollout Runbook

## Scope
This runbook covers the additive Stage 5 rollout that introduced:
- witness-first vendor synthesis
- competitive-set scoped synthesis
- `scope_manifest`
- `reasoning_atoms`
- `reasoning_delta`
- competitive-set run history

## Required Migrations
Apply these migrations before enabling the new scheduled/operator paths:
- `245_cross_vendor_reasoning_synthesis`
- `247_b2b_vendor_witness_packets`
- `261_b2b_competitive_sets`
- `262_b2b_competitive_set_runs`
- `263_b2b_competitive_set_run_constraints`

Why they matter:
- `245` enables the canonical cross-vendor synthesis table
- `247` persists witness-backed packet artifacts
- `261` adds the scoped competitive-set control plane
- `262` stores scoped run history for previews and operator visibility
- `263` hardens `262` with FK and enum-like constraints

## Backfill Requirements
No blocking DB backfill is required to deploy the new additive reasoning contract.

Why:
- `scope_manifest`, `reasoning_atoms`, and `reasoning_delta` are additive fields on new synthesis rows
- legacy and pre-atom rows still load through the shared synthesis reader
- delivery remains synthesis-first but compatibility-safe for older persisted rows

Optional backfills:
- normalize older reasoning payloads into current contract shape:
  - `python scripts/backfill_b2b_reasoning_contracts.py --apply`
- reconcile stale/legacy battle-card quality rows:
  - `python scripts/reconcile_battle_card_quality.py --write`

These improve consistency and admin/operator surfaces, but they are not a schema prerequisite.

## Safe Rollout Order
1. Apply migrations.
2. Verify readiness:
   - `python scripts/check_reasoning_rollout_readiness.py`
3. Keep local/prod synthesis disabled until readiness passes:
   - `ATLAS_B2B_CHURN_REASONING_SYNTHESIS_ENABLED=false`
   - `ATLAS_B2B_CHURN_CROSS_VENDOR_SYNTHESIS_ENABLED=false`
4. Set scheduled strategy explicitly:
   - `ATLAS_B2B_CHURN_REASONING_SYNTHESIS_SCHEDULED_SCOPE_STRATEGY=competitive_sets`
   - use `full_universe` only if you intentionally want the legacy scheduled behavior
5. Re-enable synthesis.
6. Run one bounded competitive-set smoke run.
7. Confirm previews, run history, and hash-reuse behavior before allowing broader scheduled runs.

## Readiness Criteria
The rollout is ready when all of these are true:
- required migrations are recorded in `schema_migrations`
- competitive-set tables exist
- `b2b_reasoning_synthesis` task exists
- latest synthesis rows are present with `schema_version LIKE '2.%'`
- at least some fresh synthesis rows carry:
  - `scope_manifest`
  - `reasoning_atoms`
  - `reasoning_delta`

Warnings that are acceptable during rollout:
- old synthesis rows missing atoms/delta
- zero competitive-set rows before operators create sets
- old battle-card/blog rows that predate the new quality/tracing logic

## Operational Verification
Use:
- `python scripts/check_reasoning_rollout_readiness.py`
- `python scripts/smoke_test_reasoning.py`

Recommended bounded smoke:
- create one competitive set
- focal vendor plus 1-2 competitors
- run manual preview
- run manual changed-only execution
- verify:
  - no full-universe expansion
  - only focal plus explicit competitors run
  - run history row is written
  - preview and actual reuse behavior agree

## Rollback
If rollout needs to be paused:
1. disable synthesis flags
2. set:
   - `ATLAS_B2B_CHURN_REASONING_SYNTHESIS_SCHEDULED_SCOPE_STRATEGY=full_universe`
   if you need to avoid scoped scheduling behavior without reseeding the task row
3. do not remove tables or additive fields

The schema changes are additive; rollback should be behavioral, not destructive.
