# PR-2318-Tenant-Existence-FK

## Why this slice exists

ATLAS #2318: close the tenant-EXISTENCE axis that D1 (#2317) deliberately
deferred. D1's `create_contact` guard validates tenant PRESENCE only, so a
non-blank-but-unknown `business_context_id` still reaches the provider and creates
an orphan row. D1 could not validate existence because `business_contexts` was
empty and there was no FK -- validating then would have rejected every real
tenant.

### Problem-derived contract

- **Root cause.** `contacts.business_context_id` is a bare `VARCHAR(64)` with no
  FK, and the registry table `business_contexts` is empty in prod, so nothing
  enforces that a stamped tenant is REAL. The D1 guard's own closure comment names
  this as tracked-in-#2318.
- **Correct fix touches.** (1) Seed `business_contexts` with the real tenants; (2)
  add the FK `contacts.business_context_id -> business_contexts.id` (the durable
  enforcement); (3) add a fail-safe existence net to `create_contact` for a clean
  typed refusal before the INSERT. Order is load-bearing: SEED before the FK, so
  prod's ~709 existing rows validate.
- **Must NOT change.** The presence guard and EOM guard behavior; the read path;
  `business_contexts`' voice-assistant config columns (registry seed only, no
  invented config); any tenant's data.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: production hardening

1. `migrations/365_contacts_business_context_registry_fk.sql`: seed
   `effingham_maids` + `churnsignals` as minimal registry rows (+ a dynamic
   backstop for any other stamped tenant), then add the FK idempotently.
2. `crm_server.create_contact`: after the EOM guard, reject a tenant absent from a
   POPULATED registry; degrade to presence-only when the registry is empty/
   unavailable (fail-safe).

### Files touched

- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/storage/repositories/business_context.py` (adds the admission_check method)
- `atlas_brain/storage/migrations/365_contacts_business_context_registry_fk.sql`
- `tests/test_crm_read_scoping.py`
- `tests/test_migration_365_business_context_fk.py` (real-postgres apply check)
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml` (enroll the pg test in a postgres-backed CI job)
- `tests/maturity_sweep/baseline_atlas_brain_storage.json` (accept the intentional unit-test mock of admission_check; score 8 >= 8)
- `plans/PR-2318-Tenant-Existence-FK.md`

### Review Contract

1. The FK is the durable enforcement: an unknown non-NULL tenant cannot be
   INSERTed. A NULL tenant is allowed by the FK (the D1 guard forbids NULL on the
   agent path).
2. The migration seeds BEFORE adding the FK, so existing rows validate; it is
   idempotent (ON CONFLICT DO NOTHING; FK add guarded by `pg_constraint`).
3. The runtime existence net rejects an unknown tenant ONLY when the registry is
   populated, and admits any tenant when it is empty -- so it is safe to deploy in
   any order relative to migration 365 and never rejects a real tenant because the
   table has not been seeded yet (this preserves D1's behavior on an empty DB and
   in the unit tests, where the pool is not initialized).
4. The seed is a REGISTRY seed, not voice-assistant config: `business_contexts` is
   also the voice product's config table, but it is empty in prod and its config
   columns stay NULL here.

- Reviewer rules triggered: R1, R2, R4, R5, R14. (R1: enforce existence at the DB
  root via the FK, not only a runtime string check. R2: the existence net is a
  membership guard; its closure is `admit iff registry-empty OR tenant in
  registry`, proven generatively by `test_existence_net_membership_property` over
  random (registry, candidate) pairs against a semantic oracle, plus three example
  tests. R4 (data & migration safety): migration 365 seeds `business_contexts` and
  adds the FK -- seed-before-FK ordering (so prod's ~709 rows validate), idempotent
  (`ON CONFLICT` + `pg_constraint` guard), fresh-DB no-op backstop, and NULL
  allowed by the FK (D1 forbids NULL on the agent path). R5 (backward
  compatibility): `create_contact`'s response contract gains a new refusal -- a
  create under a non-blank-but-unknown tenant now returns `{success:false, error}`
  when the registry is populated, where it previously returned a created contact.
  Deliberate; uses the tool's existing error convention; no persisted-data or
  wire-format compatibility impact, and the empty-registry fail-safe preserves
  today's behavior until the seed runs. R14: reviewer verdict discipline.)

**boundary-probe:** populated registry -> known tenant admits, unknown rejects;
empty registry -> fail-safe admits (both error directions covered).

**Mutation-probe (run, not asserted):** forcing `known_contexts = []` (neutering
the net) makes `test_create_contact_rejects_unknown_tenant_when_registry_populated`
fail, so the test is bound to the real enforcement.

## Mechanism

A seed+FK migration (the enforcement) plus a fail-safe membership check in
`create_contact` (the clean error). No read-path or other-tool change.

## Intentional

- **FK is the enforcement; the guard is the clean error.** The DB rejects an
  unknown tenant even if the runtime guard is bypassed.
- **Fail-safe on empty registry.** Deploy order between the migration and the code
  cannot break real tenants; seeding is what ACTIVATES the stricter check.
- **`business_contexts` as the registry.** The existing `BusinessContextRepository`
  already treats it as the tenant source of truth; introducing a second registry
  would create the drift this boundary exists to remove. Its voice-config role is
  noted as a consideration (see Deferred).

## Deferred

- **Whether the tenant registry should be a table separate from the voice-assistant
  config** (`business_contexts` serves both). Not changed here: the code already
  treats `business_contexts` as the registry, and a fresh table would fork the
  source of truth. Filed as a design note for the operator rather than resolved in
  this slice.
- Renaming the backstop-seeded display names (id-as-name) for any non-primary
  tenant.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_crm_read_scoping.py -q
264 passed        # 231 D1 + 3 existence examples + 30 generative membership

# Real-postgres apply check for the migration (tests/test_migration_365_*).
# Applied 040 + 035 + 365 to a throwaway postgres:16 container and asserted:
#   fresh DB   -> seeds effingham_maids/churnsignals with voice config NEUTRALIZED
#                 (scheduling_enabled/sms_enabled/take_messages FALSE; voice_name,
#                 greeting, hours, timezone NULL) + FK present, scoped to contacts;
#   prepopulated-> seed-before-FK validates existing rows; the dynamic backstop
#                 seeds a contacts-only tenant; unknown tenant is FK-rejected;
#                 NULL tenant allowed; reapply is idempotent (no dup constraint).
# 3 passed. The test SKIPS unless ATLAS_MIGRATION_TEST_DATABASE_URL is set (CI's
# migration-tests service DB sets it); it never touches prod.
```

**Operator / deploy note:** applying migration 365 to the live Atlas DB (seed +
FK) is a gated deploy step, not done by this PR. On the live DB it seeds the two
tenants and adds the FK against ~709 existing rows (all effingham_maids/
churnsignals, zero NULL -- verified), which validates.

## Rollback

Reverting merging this PR removes the migration and the guard, but on a DB where
migration 365 has already been APPLIED the FK and seed rows persist; to roll those
back:

1. **Remove the FK** (restores D1 presence-only behavior; the runtime guard's
   fail-safe also degrades once the registry looks unavailable):
   `ALTER TABLE contacts DROP CONSTRAINT IF EXISTS contacts_business_context_id_fkey;`
2. **Seed rows:** leave them. They are inert, neutralized registry rows
   (`enabled=TRUE`, all voice/scheduling/SMS config NULL/FALSE) and harmless once
   the FK is gone. Do NOT blanket-`DELETE` `business_contexts` rows: the voice
   product may own rows there, and any contact still references its tenant. If a
   specific seed row must go, delete it only after confirming no `contacts` row
   references it and the voice product does not own it.
3. **Verify:** `SELECT conname FROM pg_constraint WHERE conname =
   'contacts_business_context_id_fkey';` returns no row, and `create_contact`
   admits a previously-rejected tenant again.

The migration is idempotent and additive (seed + FK only), so a forward re-apply
after rollback is safe.

## Estimated diff size

| File | LOC (added) |
|---|---:|
| `atlas_brain/mcp/crm_server.py` | 57 |
| `atlas_brain/storage/repositories/business_context.py` | 32 |
| `atlas_brain/storage/migrations/365_contacts_business_context_registry_fk.sql` | 80 |
| `tests/test_crm_read_scoping.py` | 101 |
| `tests/test_migration_365_business_context_fk.py` | 151 |
| `plans/PR-2318-Tenant-Existence-FK.md` | 168 |
| **Total** | **589** |
