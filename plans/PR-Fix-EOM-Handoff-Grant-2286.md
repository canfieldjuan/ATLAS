# PR-Fix-EOM-Handoff-Grant-2286

## Why this slice exists

Migration `354_eom_customer_handoff_privileges.sql` transfers ownership of `eom_customer_handoffs` to the guard role `atlas_eom_handoff_owner`, and — before the transfer — grants the runtime login its DML back ("Preserve Atlas finalization access") so the app can keep finalizing handoffs. That self-grant is issued while the runtime still **owns** the table, where an owner's privileges are implicit, so it never materializes a durable ACL entry and is lost the instant ownership moves to the guard role.

The funnel had never been enabled, so this stayed latent until #2254 S0. The first real enable left the app login with **zero** privileges on `eom_customer_handoffs`, which 500s the direct `INSERT` in `finalize_eom_customer_handoff` (`atlas_brain/services/crm_provider.py`) and any funnel read touching handoff state. The boot gate `require_eom_funnel_data_store` does not check the runtime's own privileges, so it passed while the write path was broken. Tracked in #2286.

### Problem-derived contract

Restore the runtime's handoff DML so it **persists** past both the ownership transfer and the DBA's post-commit membership revoke, without granting `atlas_nocodb` anything and without changing the boot-gate contract. It must add no new step to the enablement runbook.

## Scope (this PR)

Ownership lane: eom-lead-funnel-handoff-privileges
Slice phase: production hardening

- Edit `354_...sql`: after the `ALTER TABLE ... OWNER TO atlas_eom_handoff_owner`, re-grant the runtime DML **as the guard owner** via `SET LOCAL ROLE atlas_eom_handoff_owner`, so the grant is made by the new owner and survives the transfer and the post-commit revoke. The runtime still holds admin membership at that point in the migration, so the `SET ROLE` is permitted.
- Extend `test_privilege_migration_runs_from_a_real_non_superuser_login` to assert the runtime keeps `SELECT/INSERT/UPDATE/DELETE` on `eom_customer_handoffs` after the membership revoke — the assertion the readiness guard cannot make.

### Review Contract

- Reviewer rules triggered: R2, R3, R4.
- **R4 (migration safety):** additive, forward-only, non-destructive. The runner keys applied migrations by filename stem with no checksum, so DBs that already ran 354 (only the ts.net host — hand-patched during #2254 S0) do not re-run it; fresh/CI DBs get the corrected grant. No rollback needed; the grant is scoped to the runtime login and idempotent under re-issue. The pre-transfer self-grant is left in place because it correctly handles `eom_lead_lifecycle_events`, whose ownership does not move.
- **R3 (authorization):** the added grant targets only the runtime login and is issued as the guard owner; `atlas_nocodb` privileges and the non-membership gate clause are unchanged.
- **R2 (migration/rollback test):** the regression runs 354 from a real non-superuser login, revokes the guard membership, and asserts the runtime's handoff DML persists — red on the pre-fix migration, green after.

### Files touched
- `atlas_brain/storage/migrations/354_eom_customer_handoff_privileges.sql`
- `tests/test_eom_lead_conversion_integration.py`
- `plans/PR-Fix-EOM-Handoff-Grant-2286.md`

### Boundary-change enumeration
No module/ownership boundary changes: the edit stays inside migration 354 and its existing non-superuser proof. No new imports, callers, or public surfaces.

### Deployed-config probing
No deployed-config or env changes. The migration is applied by the existing startup runner; no new env var, secret, or blueprint slot.

## Mechanism

A table owner's privileges are implicit, so `GRANT ... TO <runtime>` while `<runtime>` owns the table records no ACL entry; after `ALTER TABLE ... OWNER TO atlas_eom_handoff_owner` the runtime is no longer owner and holds nothing. The fix issues the grant **after** the transfer, under `SET LOCAL ROLE atlas_eom_handoff_owner`, so PostgreSQL records the grant with the guard role as grantor. That ACL entry is independent of the runtime's membership in the guard role, so the DBA's post-commit `REVOKE atlas_eom_handoff_owner FROM <runtime>` does not remove it.

## Intentional
- The pre-transfer self-grant block is retained (it correctly grants `eom_lead_lifecycle_events`, still owned by the runtime).
- `TRUNCATE` is included to match the migration's original stated intent for the handoff table.

## Deferred
- Longer term, `finalize_eom_customer_handoff` could move to a `SECURITY DEFINER` function owned by the guard role so the runtime needs no direct table grant. Out of scope here; the grant-persistence fix is the minimal correct change.

## Verification

Mechanism proven against real PostgreSQL (throwaway roles, rolled back): the buggy self-grant-then-transfer leaves the runtime with `INSERT = false`; granting as the owner after transfer yields `true`; and it stays `true` after the membership revoke. The extended integration test exercises the full path (non-super login applies 354, membership revoked) and asserts the DML persists; it is red against the pre-fix migration and green after.

## Estimated diff size

| Change | LOC |
|---|---|
| migration 354 post-transfer grant block | ~27 |
| non-superuser regression assertion | ~15 |
| this plan doc | ~65 |
| **Total** | ~107 |

## Cold diff reconstruction
- `354_...sql`: after the three `ALTER ... OWNER TO atlas_eom_handoff_owner` statements, add a `DO` block that captures `current_user`, `SET LOCAL ROLE atlas_eom_handoff_owner`, `GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON <schema>.eom_customer_handoffs TO <runtime>`, then `RESET ROLE`.
- `test_eom_lead_conversion_integration.py`: in the post-revoke verifier block of `test_privilege_migration_runs_from_a_real_non_superuser_login`, assert `has_table_privilege(current_user, 'eom_customer_handoffs', p)` for `p` in `SELECT/INSERT/UPDATE/DELETE`.
