# PR-2318-Tenant-Existence-FK

## Why this slice exists

ATLAS #2318: close the tenant-EXISTENCE axis that D1 (#2317) deliberately
deferred. D1's `create_contact` guard validates tenant PRESENCE only, so a
non-blank-but-unknown `business_context_id` still reaches the provider and creates
an orphan row. D1 could not validate existence because `business_contexts` was
empty and there was no FK -- validating then would have rejected every real
tenant.

**Diff-budget overage (1094 LOC vs the 400 soft cap) -- why this slice is
indivisible.** Production code is ~132 LOC (MCP guard 74 + repo `admission_check` 42 +
the call-recording `_link_to_crm` guard 16); the migration is 83. The remaining ~879 is
review-mandated *evidence for this exact change*, not extra scope: the real-postgres
test file (340) that proves seed-before-FK ordering, FK enforcement, neutralization,
idempotence, and the concurrent-writer lock protocol; the generative membership unit
tests (102); the call-recording guard's boundary-probe tests (80); and this contract
plan itself (334, incl. rollback + the R8 execution-model criterion + the
affected-surfaces/risk declarations). Splitting the migration from its guard would ship
enforcement without its DB root (or vice-versa) for a window; splitting either from its
tests would orphan the acceptance evidence the reviewer required. Every LOC over the cap
is the seed+FK+guard triple (now incl. the recording-writer alignment) or the proof that
it is correct -- there is no independently-shippable sub-slice.

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

1. `migrations/365_contacts_business_context_registry_fk.sql`: in ONE DO block that
   holds `LOCK TABLE contacts IN SHARE MODE` (atomic on the autocommit runner),
   seed `effingham_maids` + `churnsignals` + `personal` (every canonical
   contact-writer context) plus a dynamic backstop, then add the FK idempotently.
2. `crm_server.create_contact`: after the EOM guard, reject a tenant absent from
   the registry ONLY once migration 365 has run (the FK exists); degrade to
   presence-only before then / when the registry is unavailable (fail-safe).

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/comms/call_intelligence.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/storage/migrations/365_contacts_business_context_registry_fk.sql`
- `atlas_brain/storage/repositories/business_context.py`
- `plans/PR-2318-Tenant-Existence-FK.md`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/test_call_intelligence.py`
- `tests/test_crm_read_scoping.py`
- `tests/test_migration_365_business_context_fk.py`

### Review Contract

**Affected surfaces (in bounds for this review):**
- **Runtime admission seam** -- `atlas_brain/mcp/crm_server.py::create_contact`, the
  only tool that runs the existence net; its sole direct caller is the MCP CRM
  server's create path. The presence/EOM guards and the read path are unchanged.
- **Repository behavior** -- `atlas_brain/storage/repositories/business_context.py`
  adds `admission_check`; no existing method's behavior changes.
- **Call-recording CRM link** -- `atlas_brain/comms/call_intelligence.py::_link_to_crm`
  gains a pre-write guard that skips an unresolved (`"unknown"`/blank) call context;
  resolved tenants are unaffected. The shared chokepoint guard is deferred (#2327).
- **Database schema/data** -- migration `365_...sql` seeds `business_contexts` and
  adds the `contacts.business_context_id` FK; no other table or migration is touched.
- **CI** -- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`, the postgres-backed
  job that executes the migration test.

**Risk areas:**
- **Migration/data safety (R4/R8)** -- seed-before-FK ordering against prod's ~709
  rows; the SHARE-lock DO block's concurrency invariant; idempotent re-apply.
- **Backward compatibility (R5)** -- the new unknown-tenant refusal in `create_contact`.
- **CI reachability (R12)** -- the postgres job must trigger on every file the
  migration test touches, including its prerequisites 035 + 040.
- **Deploy** -- applying 365 is a gated manual step (not done by merge); the runtime
  net is fail-safe until the FK exists.

**Reachability declaration (R12):** the migration test's prerequisites (035, 040) and
subject (365), the test file itself, the guard, and the repository method are all in
both path-filter blocks of `atlas_eom_lead_pipeline_checks.yml`, and the workflow
triggers on changes to its own file -- so any change that could regress the seed /
neutralization / FK behavior triggers the exact job that verifies it (no reachability
gap). The recording-writer guard is enrolled the same way: `call_intelligence.py` is
in both path-filter blocks and `tests/test_call_intelligence.py` is added to both the
filters AND the explicit pytest command, so inverting the `_link_to_crm`
unresolved-context rejection fails this required PR gate -- not only the schedule-only
repo-wide backstop. (The per-area workflows run hand-maintained test lists, so a test
no workflow enrolls can pass CI while never executing; this closes that gap for the
guard.)

1. The FK is the durable enforcement: an unknown non-NULL tenant cannot be
   INSERTed. A NULL tenant is allowed by the FK (the D1 guard forbids NULL on the
   agent path).
2. The migration seeds BEFORE adding the FK, so existing rows validate; it is
   idempotent (ON CONFLICT DO NOTHING; FK add guarded by `pg_constraint`).
3. The runtime existence net enforces ONLY once migration 365 has run (the FK
   `contacts_business_context_id_fkey` exists) -- NOT gated on table occupancy,
   because `business_contexts` pre-exists for the voice product and may hold
   unrelated rows. Before the FK exists it admits any tenant, so it is safe to
   deploy in any order relative to 365, never rejects a real tenant before the seed
   lands, and the runtime refusal exactly mirrors the FK (this preserves D1's
   behavior pre-migration and in the unit tests, where the pool is not initialized).
4. The seed is a REGISTRY seed, not voice-assistant config: `business_contexts` is
   also the voice product's config table, but it is empty in prod and its config
   columns stay NULL here.

- Reviewer rules triggered: R1, R2, R4, R5, R8, R12, R13, R14. (R1: enforce existence at the DB
  root via the FK, not only a runtime string check. R2: the existence net is a
  membership guard; its closure is `admit iff FK-not-yet-present OR tenant in
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
  today's behavior until the seed runs. R8 (concurrency): the migration's
  correctness under concurrent contact writes is a transaction-scoped invariant,
  not a single prose example. Execution model: the migration runner is autocommit
  and splits statements (so `CREATE INDEX CONCURRENTLY` can run), meaning each
  top-level statement commits independently; wrapping the seed and the FK add in
  one DO block makes them a single server-side transaction. Invariant across
  admitted interleavings: `LOCK TABLE contacts IN SHARE MODE`, taken first in that
  transaction, blocks every concurrent contact INSERT/UPDATE for the block's
  duration, so the set of tenants visible to the seed's snapshot is exactly the set
  present when the FK is added -- no writer can commit a new tenant between the
  snapshot and the ALTER. Interleavings: a writer that committed BEFORE the lock is
  captured by the dynamic backstop (`SELECT DISTINCT ... FROM contacts`); a writer
  that arrives DURING the block is queued and resumes after COMMIT against the
  now-present FK (a new tenant then correctly FK-fails at the app layer, which is
  the intended enforcement). Crash/cancellation assumption: if the migration
  transaction aborts mid-block, the whole DO block rolls back -- seed and FK both
  absent -- leaving the DB re-runnable and idempotent; SHARE (not ACCESS EXCLUSIVE)
  is deliberate so concurrent reads are never blocked. Driven end-to-end (not a
  hand-copied lock) by `test_migration_365_serializes_a_concurrent_writer`, which
  runs the real migration against an uncommitted-insert writer and asserts, via
  `pg_locks`, that the migration WAITS on a `ShareLock` on `contacts` before it can
  seed -- so moving the lock below the seed or out of the DO block fails the test.
  R12 (CI reachability): the migration test's prerequisites (035, 040) and subject
  (365), the test file, the guard, and the repository method are all in both
  path-filter blocks of the postgres-backed workflow, which also triggers on its own
  file -- so a change that could regress the seed/neutralization/FK behavior triggers
  the job that verifies it (see the Reachability declaration above). R13 (contact-writer
  consistency with the FK): the FK governs EVERY contacts writer, so the recording
  path's "unknown" sentinel (assigned when a call's context can't be resolved) is now
  rejected in `_link_to_crm` BEFORE `find_or_create_contact`, so an unattributable
  call skips CRM linking cleanly instead of manufacturing a mis-tenanted contact that
  would FK-violate; SMS resolves to the seeded `personal` context, and the shared
  provider chokepoint's uniform guard is tracked in #2327. Covered by
  `TestLinkToCrmUnresolvedContext` (unresolved -> skip; a resolved tenant -> writes).
  R14: reviewer verdict discipline.)

**Admission-boundary enumeration (R1).** The decision seam is `create_contact`,
immediately after the EOM guard and before the provider INSERT. Complete
enumeration by tenant-input shape (not just the example criteria below):

- **explicit known tenant** (e.g. `effingham_maids`, `churnsignals`, `personal`, a
  backstop-seeded id) -> admits; the normalized value is stamped so it matches the FK.
- **explicit unknown non-blank tenant** (typo / injected sentinel) -> REPLACED
  behavior: D1 admitted it (presence-only) and created an orphan row; now it is
  rejected `{success:false, error}` once the FK exists, and fail-safe-admitted only
  before the FK exists / when the registry is unavailable.
- **blank or whitespace-only tenant** -> unchanged: the D1 presence guard rejects it
  on the agent path; it never reaches the existence net.
- **NULL / absent tenant** -> unchanged: D1 forbids NULL on the agent path, while the
  FK itself permits NULL (so non-agent writers are not broken).
- **default / resolved tenant** (context resolution or the EOM guard supplies the
  value) -> evaluated as an explicit value at the seam; admitted iff it is in the
  registry, on the same FK-readiness gate.

Caller/input shape: the MCP `create_contact` tool is the only writer that runs this
runtime seam; the other contact writers (inbound SMS, webhooks, lead-ingress) are
governed by the FK directly at the DB -- which is why `personal` and the backstop
tenants are seeded, so those writers are not silently dropped. No input shape reaches
the provider INSERT without passing both the presence guard and, when enforced, the
existence net.

**boundary-probe:** FK present (365 ran) -> known tenant admits, unknown rejects;
FK absent (pre-migration) -> fail-safe admits any tenant even with unrelated voice
rows present (both error directions covered).

**Mutation-probe (run, not asserted):** forcing `known_contexts = []` (neutering
the net) makes `test_create_contact_rejects_unknown_tenant_when_registry_populated`
fail, so the test is bound to the real enforcement.

## Mechanism

A single DO-block migration -- `LOCK TABLE contacts IN SHARE MODE`, then seed,
then the FK, atomic w.r.t. contact writes on the autocommit runner (the
enforcement) -- plus a fail-safe membership check in `create_contact` (the clean
error). No read-path or other-tool change.

## Intentional

- **FK is the enforcement; the guard is the clean error.** The DB rejects an
  unknown tenant even if the runtime guard is bypassed.
- **Fail-safe until the FK exists.** The net enforces only once migration 365 adds
  the FK, not on table occupancy; deploy order cannot break real tenants, and an
  unrelated pre-existing voice row cannot trigger a false rejection.
- **`business_contexts` as the registry.** The existing `BusinessContextRepository`
  already treats it as the tenant source of truth; introducing a second registry
  would create the drift this boundary exists to remove. Its voice-config role is
  noted as a consideration (see Deferred).

## Deferred

**Parking predicate.** A finding is parked here iff it is a latent robustness gap
that does NOT change the slice's behavior for any tenant present in prod -- verified
read-only: `contacts.business_context_id` is `effingham_maids` (709) or
`churnsignals` (1), with zero NULL/whitespace/non-canonical keys. Anything that
alters a real tenant's behavior is NOT parked; it blocks. All three items below are
tracked in **#2327**, each with an unlock condition.

- **Legacy whitespace / non-canonical tenant-key normalization** (from the R5 P2
  review thread). The backstop seeds each existing `contacts.business_context_id`
  verbatim (so the FK validates existing rows), while `create_contact` now `btrim()`s
  the tenant before `admission_check` + persist -- so a hypothetical legacy key like
  `' acme '` would seed raw but be looked up trimmed, and a new create for it would be
  FK-rejected. Non-triggering now (prod has only the two clean canonical keys above;
  the canonical seed is hardcoded-clean). **Unlock (#2327):** the moment a whitespace
  or non-canonical key is introduced, backfill `contacts.business_context_id =
  btrim(...)` or seed both raw + trimmed forms in the backstop.
- **Whether the tenant registry should be a table separate from the voice-assistant
  config** (`business_contexts` serves both). Not changed here: the code already
  treats it as the registry, and a fresh table would fork the source of truth.
  **Unlock (#2327):** if the voice-config schema and the registry needs diverge enough
  that shared columns cause friction, add a dedicated `tenant_registry` table and move
  the FK to it.
- **Backstop-seeded display names (id-as-name) for any non-primary tenant.**
  **Unlock (#2327):** when a non-primary tenant becomes customer-visible and needs a
  real display name.

Parked hardening: none. (The above are DEFERRED design/robustness items with unlock
conditions, not parked hardening on the shipped code path -- the seed+FK+guard triple
is fully implemented and tested here.)

## Verification

```
$ python -m pytest tests/test_crm_read_scoping.py -q
264 passed        # 231 D1 + 3 existence examples + 30 generative membership

# Real-postgres apply check for the migration (tests/test_migration_365_*).
# Applied 040 + 035 + 365 to a throwaway postgres:16 container and asserted:
#   fresh DB   -> seeds effingham_maids/churnsignals/personal with voice config
#                 NEUTRALIZED (scheduling_enabled/sms_enabled/take_messages FALSE;
#                 voice_name, greeting, hours, timezone NULL) + FK present, scoped
#                 to contacts; real list_enabled() excludes the enabled=FALSE seeds;
#   prepopulated-> seed-before-FK validates existing rows; the dynamic backstop
#                 seeds a contacts-only tenant; unknown tenant is FK-rejected;
#                 NULL tenant allowed; reapply is idempotent (no dup constraint);
#   race       -> the REAL migration is run against an uncommitted-insert writer;
#                 pg_locks shows it WAITING on a ShareLock on contacts before it can
#                 seed, then (writer committed) it completes, backstop-seeds the
#                 raced tenant, and the FK lands with no orphan. Mutation-probed:
#                 removing the LOCK makes it wait on ShareRowExclusiveLock at the
#                 ALTER instead -> the test fails, so it is bound to the real lock.
# 4 passed. The test SKIPS unless ATLAS_MIGRATION_TEST_DATABASE_URL is set (CI's
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
2. **Reset the migration ledger** if you intend a later redeploy to re-apply 365.
   The repository runner (`atlas_brain/storage/migrations`) records each applied
   migration by name in `schema_migrations` and SKIPS any already recorded -- so
   dropping only the FK would leave `365` recorded and the FK **permanently absent**
   after the next deploy (`create_contact` would silently resume admitting unknown
   tenants). Reset the ledger entry so 365 re-runs:
   `DELETE FROM schema_migrations WHERE name = '365_contacts_business_context_registry_fk';`
   Conversely, to STAY rolled back, leave this row in place so the runner does not
   re-apply 365.
3. **Seed rows:** leave them. They are inert, neutralized registry rows
   (`enabled=FALSE`, all voice/scheduling/SMS config NULL/FALSE) and harmless once
   the FK is gone. Do NOT blanket-`DELETE` `business_contexts` rows: the voice
   product may own rows there, and any contact still references its tenant. If a
   specific seed row must go, delete it only after confirming no `contacts` row
   references it and the voice product does not own it.
4. **Verify:** `SELECT conname FROM pg_constraint WHERE conname =
   'contacts_business_context_id_fkey';` returns no row and `create_contact` admits a
   previously-rejected tenant again; if you reset the ledger (step 2), also confirm
   `SELECT 1 FROM schema_migrations WHERE name =
   '365_contacts_business_context_registry_fk'` returns no row.

The migration is idempotent and additive (seed + FK only), so a forward re-apply
after rollback re-seeds and re-adds the FK cleanly -- but ONLY once the ledger row
is reset (step 2). Without that reset the runner skips 365 and the FK never returns,
so "drop the FK" alone is a one-way rollback.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 12 |
| `atlas_brain/comms/call_intelligence.py` | 16 |
| `atlas_brain/mcp/crm_server.py` | 74 |
| `atlas_brain/storage/migrations/365_contacts_business_context_registry_fk.sql` | 83 |
| `atlas_brain/storage/repositories/business_context.py` | 42 |
| `plans/PR-2318-Tenant-Existence-FK.md` | 334 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 11 |
| `tests/test_call_intelligence.py` | 80 |
| `tests/test_crm_read_scoping.py` | 102 |
| `tests/test_migration_365_business_context_fk.py` | 340 |
| **Total** | **1094** |
