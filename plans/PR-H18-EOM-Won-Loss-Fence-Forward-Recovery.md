# PR-H18-EOM-Won-Loss-Fence-Forward-Recovery

## Why this slice exists

H-18 issue [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476) correctly
keeps migration `389_eom_missed_call_recovery` blocked by historical migration
evidence. Fresh, read-only target evidence shows that the recorded
`386_eom_won_loss_nocodb_fence` digest is the original NocoDB-only source,
while the packaged migration was later strengthened in place. The target's
function still predicates on `session_user = 'atlas_nocodb'`, its trigger only
watches `status`, and the canonical `crm_user` role has direct `contacts`
`UPDATE` permission. Thus the target does not have the direct-SQL boundary the
current package describes.

This is a `Production hardening` slice justified by a concrete safety and
deployment blocker: the ordinary runner cannot apply the amended historical
source, and blindly attesting the weak target would let unrelated pending SQL
run before the missing direct-write fence is restored. It adds one explicit,
forward-only recovery route under the existing migration lock; it does not
change a customer-facing workflow.

The slice exceeds the normal 400-LOC budget because its one deployment-safe
security boundary is indivisible: the forward-only SQL fence, exact catalog
attestation, selected-only runner admission, canonical completion sequencing,
and fake-runner plus disposable PostgreSQL evidence must ship together. Of the
current 2,072 changed lines, 712 are the two focused regression suites and 383
are the executable plan/evidence record; the small runtime compatibility change
is the 37-line in-transaction reorder in `crm_provider.py`. Splitting the
remaining migration/attestation/runner pieces would either publish a privileged
function without its admission proof, block the existing authorized won-loss
completion, or weaken the recovery path before its direct-SQL bypass tests
exist.

### Problem-derived contract

- Root cause: migration 386's source was strengthened after its original
  NocoDB-only form had already been recorded on the canonical target. The
  immutable ledger therefore prevents replaying the amended file, while the
  target's weak function leaves other direct database clients able to mutate a
  won EOM lead's `status` or `contact_type` during an unresolved Calendar
  cancellation. The generic integrity gate has no way to apply the required
  forward recovery before it admits later pending migrations.
- Correct fix must touch/change:
  1. Add one source-controlled, target-specific reconciliation record for
     historical migration 386. It must distinguish the exact legacy weak
     function/trigger catalog state from the recovered direct-SQL fence and
     never call the weak state fully attested.
  2. Add one atomic, additive migration
     `390_eom_won_loss_direct_sql_fence_recovery.sql` that replaces the
     function and trigger with the existing all-direct-SQL predicate, fixed
     schema-qualified lifecycle reads with `pg_temp` explicitly after the
     trusted schema, `SECURITY DEFINER`, every mutable field used by the
     EOM-won predicate (`status`, `contact_type`, `lead_stage`, and
     `business_context_id`) in the trigger update coverage, delete coverage,
     revoked public execute privilege, and
     an ownership transfer to the existing no-login
     `atlas_eom_handoff_owner` guard. The guard receives only `SELECT` on the
     lifecycle evidence needed by the predicate; the recovery must require a
     database administrator rather than granting that guard to the app login.
  3. Teach the existing migration runner to execute only that named recovery
     under its existing session advisory lock when the *only* remaining
     unresolved content evidence is the exact target-confirmed 386 legacy
     state. It must reserve every registered forward-recovery migration from
     the ordinary pending loop even when an explicit `only=` caller selected
     it, commit the attested recovery and its ledger receipt atomically,
     re-read the ledger and integrity report, then apply ordinary pending SQL
     only if the recovered target attests. Any other unresolved record,
     malformed legacy target, missing recovery source, or subset that omits
     the recovery remains a zero-pending-SQL failure.
  4. Include the named recovery in the closed EOM missed-call readiness set so
     its existing `only=` caller can select the prelude without the runner
     silently applying an unrequested migration. Add focused fake-runner,
     read-only-preflight, and disposable-PostgreSQL tests for legacy
     recognition, recovery ordering, recovery retry, failure closure, and the
     strong trigger's direct-SQL behavior, including an attacker-owned temp
     lifecycle relation and each mutable predicate field. Enroll every changed
     EOM entrypoint and test surface in the existing EOM pipeline workflow.
  5. Preserve the canonical authorized won-loss completion under the expanded
     trigger event set: within its existing database transaction, persist the
     matching cancellation-completed lifecycle evidence before its protected
     `lead_stage` update, then persist the loss event. A failed transition or
     later event write must roll the whole transaction back, so no committed
     completion evidence can authorize an uncommitted or failed loss.
- Must not change:
  1. The historical `386_eom_won_loss_nocodb_fence.sql` bytes,
     `schema_migrations` rows, or generic H-18 report semantics. No source
     rewrite, digest backfill, replay, or generic allowlist is permitted.
  2. Migration 379, other H-18 evidence records, external migration-evidence
     serialization, the migration runner's ordinary ordering for non-recovery
     migrations, and all commercial-billing work.
  3. Existing lead lifecycle, Calendar, CRM API, role grants, customer data,
     email delivery, configuration, Website, tracker, and user-facing product
   behavior, except for the single guard-only lifecycle `SELECT` needed by
   the recovered SECURITY DEFINER function and the internal, atomic order of
   already-required completion evidence before the protected lead-stage write.
  4. Live target migration execution. Verification uses catalog-only target
     reads and disposable test schemas only.

## Scope (this PR)

Ownership lane: h18-migration-content-integrity
Slice phase: Production hardening

Max files: 12

1. Record the exact legacy 386 evidence and its required recovered state in
   the existing reconciliation module.
2. Apply an atomic forward migration only through a named, fail-closed runner
   prelude, reserve it from ordinary pending SQL, then re-evaluate the same
   integrity gate.
3. Add the recovery to the existing closed EOM missed-call readiness set so
   its `only=` call preserves caller-selected migration semantics.
4. Prove both the semantic fence and the runner ordering with existing
   migration test surfaces, including the guarded function owner and its
   lifecycle read, temp-schema isolation, and full mutable-predicate trigger
   coverage; keep the current EOM pipeline coverage enrolled.
5. Reorder only the existing authoritative provider's in-transaction
   cancellation-completed event so the expanded fence preserves the canonical
   Calendar-confirmed won-loss completion without granting any direct writer a
   bypass.

### Review Contract

- Acceptance criteria:
  - [x] `386_eom_won_loss_nocodb_fence` is `recovery_required`, not
    `attested`, only when the read-only evidence matches its exact historical
    ledger identity and NocoDB-only function/trigger state; settled by focused
    `tests/test_migrations_runner.py` cases and the controlled target preflight.
  - [x] A target-shaped legacy 386 mismatch with the named 390 recovery pending
    executes that recovery under the normal runner's existing advisory-lock and
    atomic-bookkeeping path before a later pending migration; the runner
    re-reads evidence and records/apply later SQL only after the recovered
    function, trigger, and recovery ledger receipt attest; settled by focused
    `tests/test_migrations_runner.py` behavior assertions.
  - [x] A selected 390 recovery with no exact weak-386 precondition is not
    executed, does not record a ledger receipt or change function ownership,
    and does not prevent an ordinary selected migration from running; settled
    by the focused fake-runner case and a disposable-PostgreSQL selected-run
    assertion.
  - [x] A second unresolved record, altered legacy metadata (including a
    wrong historical version or trigger `WHEN` condition), recovery subset
    omission, or a recorded-but-weak recovery leaves
    every pending SQL statement and ledger insertion absent; settled by the
    negative runner/preflight tests.
  - [x] In a disposable PostgreSQL schema, the recovered trigger rejects a
    direct non-NocoDB `status`, `contact_type`, and delete mutation while
    unresolved cancellation evidence exists, but preserves an ordinary
    non-state update; settled by
    `tests/test_eom_lead_conversion_integration.py`.
  - [x] The recovered SECURITY DEFINER function is owned by the trusted,
    membership-isolated no-login guard, that guard has only the lifecycle read
    needed by the function, and the former direct CRM login cannot alter the
    function; a non-administrator cannot start the recovery. Settled by the
    same disposable-PostgreSQL test and the post-recovery catalog attestation.
  - [x] In a disposable PostgreSQL schema, an attacker-created temp
    `eom_lead_lifecycle_events` relation with a guard-role `SELECT` grant
    cannot hide unresolved cancellation evidence, and direct changes to each
    mutable EOM-won predicate field are rejected before they can make a later
    protected mutation eligible; settled by the real recovery runner and
    direct-connection regression assertions.
  - [x] The normal Calendar-confirmed won-loss completion still reaches `lost`
    under the expanded trigger, while its completion evidence, contact change,
    and loss event remain one rollback-safe transaction; settled by the same
    disposable-PostgreSQL end-to-end regression.
  - [x] The EOM lead pipeline workflow runs when this forward-recovery migration
    or its curated EOM profile contract changes, and it executes the
    profile-contract assertion; settled by
    `tests/test_eom_lead_conversion.py::test_eom_lead_pipeline_workflow_enrolls_won_loss_runtime_paths`.
- Reachability proof: `run_migrations()` is the production migration entrypoint.
  Focused runner tests invoke it through the real migration selection and
  observe the recovery ledger/SQL ordering; the real-PostgreSQL test invokes
  the same runner against a disposable schema and observes the trigger verdict.
- Affected surfaces: migration content admission, migration ledger recording,
  PostgreSQL function/trigger catalog, EOM migration CI enrollment.
- Risk areas: migration ordering, data safety, atomic recovery, direct SQL
  lifecycle bypass, retry/idempotency, backward compatibility.
- Reviewer rules triggered: R1, R2, R4, R5, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: `run_migrations()` content-evidence admission before its
  first pending SQL statement.
- Replaced-path behaviors:
  - no unresolved evidence: preserve the ordinary non-recovery pending path
    while every selected forward-recovery migration remains inert;
  - exact legacy 386 only: run only named 390 recovery first, then re-evaluate;
  - any other unresolved name or any unrecognized 386 target: reject before
    all pending SQL;
  - recovered 386: preserve ordinary pending-migration execution after the
    fresh evidence check.
- Guard-relevant fields: mismatch and missing-source names, exact historical
  ledger version/digest/timestamp, final and recovery package digests, function
  body, function security/search-path/ACL metadata, trusted guard role and
  ownership/lifecycle-read metadata, trigger events, `WHEN` condition, and columns,
  recovery ledger row, recovery-file presence in the selected pending set,
  SECURITY DEFINER search-path order, lifecycle relation identity, and every
  mutable input to the `OLD` EOM-won predicate.
- Caller x input shape:
  - full generic runner x legacy 386 plus later pending SQL;
  - `only=` runner including recovery x legacy 386 plus later pending SQL;
  - `only=` runner omitting recovery x legacy 386;
  - closed EOM missed-call readiness caller x legacy 386, where 390 must be
    explicit in its selected prerequisite set rather than implicitly escaping
    `only=`;
  - closed EOM missed-call readiness caller x fresh or already-strong 386,
    where 390 is selected but must not execute or create a receipt;
  - direct EOM writer x attacker-owned temp lifecycle relation that grants the
    guard `SELECT`, where the trusted schema relation must still decide the
    fence;
  - direct EOM writer x `lead_stage` or `business_context_id` mutation before
    a protected `status`/`contact_type` mutation;
  - either runner x a second unresolved H-18 record;
  - rerun after an atomic recovery receipt already exists.

### Deployed-config probing

- Deployed/default config values: N/A; this slice adds no configuration or
  fallback.
- Explicit value probe: N/A.
- Absent value probe: N/A.
- Default-session/default-context probe: the recovery uses PostgreSQL's
  `current_schema()` under the existing runner connection; disposable-schema
  coverage proves the function and trigger bind to that schema.
- Side-effect ordering: target evidence is read before recovery selection; the
  390 SQL and its ledger receipt commit as one atomic-bookkeeping unit; the
  runner re-reads integrity evidence before it can execute another pending
  migration.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/390_eom_won_loss_direct_sql_fence_recovery.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/PR-H18-EOM-Won-Loss-Fence-Forward-Recovery.md`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migration_content_integrity_preflight.py`
- `tests/test_migrations_runner.py`

## Mechanism

The reconciliation record treats the observed 386 target as a *recovery
precondition*, not a successful historical attestation. It accepts only one
known original ledger receipt and exact weak function/trigger metadata. The
same record can become `attested` only after the named 390 migration is
recorded with its packaged digest and the target has the stronger current
function/trigger catalog contract.

The recovered catalog additionally requires the existing
`atlas_eom_handoff_owner` role to remain a membership-isolated no-login guard,
to own the SECURITY DEFINER function, and to have only the lifecycle-evidence
read needed to evaluate it. Migration 390 is DBA-only and transfers ownership
inside its existing atomic transaction. This avoids granting the guard to the
normal Atlas login; a normal startup attempt fails before replacing the
function, trigger, grant, or ledger receipt.

The recovered function resolves lifecycle evidence only in its trusted active
schema: both lifecycle table references are schema-qualified by the migration's
quoted current schema, and its SECURITY DEFINER search path puts `pg_temp` after
that schema. Its trigger subscribes to every mutable input of the `OLD`
EOM-won predicate, so a direct writer cannot first move the row out of scope and
then change a formerly protected field. Ordinary unrelated updates remain
outside that closed event set.

The canonical Calendar-confirmed completion already owns the matching lifecycle
event, contact transition, and loss record in one database transaction. Its
completion evidence is written immediately before the protected `lead_stage`
transition, rather than afterward, so the fence sees the authorized evidence
without creating a committed bypass window. Any error in the contact or loss
write rolls the completion evidence back with the same transaction.

PostgreSQL reports a trigger's update columns in the table's physical-column
order rather than the `CREATE TRIGGER` declaration order. The recovered
contract therefore requires the exact four-column predicate set
`business_context_id`, `contact_type`, `lead_stage`, and `status`, with no
extras, rather than incorrectly depending on source text ordering.

`run_migrations()` already holds one session-level advisory lock across its
content report, migration SQL, and ledger writes. A closed, source-derived
registry marks named forward-recovery migrations as prelude-only: an explicit
`only=` caller can authorize 390, but the ordinary loop logs and skips it until
the exact legacy precondition selects it. When no other discrepancy is
unresolved and that exact precondition is present, the runner executes only the
named recovery through the existing atomic-bookkeeping implementation, then
recomputes applied rows, the content report, and reconciliation evidence before
entering the ordinary pending loop. Thus no later SQL can run on the weak
target, and a fresh/current target cannot create a 390 receipt or change
function ownership merely because its selected readiness set names 390. The
recovery is intentionally a forward migration numbered after the currently
pending 389 file; the named prelude is the explicit dependency edge, not a new
numeric-prefix collision or a generic ordering rule.

The EOM missed-call profile intentionally calls the runner with a closed
`only=` set. The new 390 name is added to that set in the same slice: the
profile still selects only its explicit prerequisites, while the runner may
choose 390 ahead of 389 solely for the exact legacy-386 recovery precondition.
It never discovers and executes an unselected migration behind the caller's
back.

Execution model: cooperating Atlas runners serialize on the existing session
advisory lock, and the recovery's PostgreSQL transaction makes its function,
trigger, and ledger receipt all-or-nothing. A queued cooperating runner
re-snapshots after the holder releases the lock. External privileged database
sessions are not serialized by this existing component; H-18's documented
external migration-evidence serialization item remains deferred, and normal
operation still requires a fresh read-only preflight if external mutation is
suspected. The DBA must invoke the existing selected migration runner rather
than executing the SQL file directly, so that same transaction records 390's
digest before 389 or any ordinary pending migration can be considered.

## Intentional

- The exact weak target remains visible as `recovery_required`; it is never
  declared equivalent to the strengthened historical source.
- The prelude is record-specific and only executable when it is the sole
  unresolved discrepancy. There is no reusable exception list, source replay,
  or generic “repair all mismatches” path.
- The EOM readiness set continues to name 390 as explicit caller authorization;
  it is not removed or made implicit. The runner's prelude-only registry is the
  single distinction between authorization and ordinary application.
- The 390 migration duplicates the required function/trigger definition rather
  than altering historical 386 bytes. This preserves immutable ledger meaning
  and lets the recovery commit with its own digest.
- Ordinary application rollback retains the new fence and immutable lifecycle
  evidence. A separately approved destructive rollback must first resolve every
  outstanding cancellation and then drop trigger before function; it is not
  automated here.
- The prior direct-SQL function name remains for compatibility; its scope is
  corrected from NocoDB-only to all direct writers. The only role grant added
  is guard-only `SELECT` on lifecycle evidence; no application or CRM role
  gains a privilege.
- The existing provider keeps its same Calendar-confirmed completion contract.
  Only the order of already-required writes changes inside its existing
  transaction, so a failed contact/loss transition cannot leave committed
  completion evidence behind.

## Deferred

- `379_commercial_billing_candidate_review_decisions` remains its own
  source-unavailable receipt slice under #2476. Until it independently attests,
  the runner intentionally refuses the 386 recovery and all other pending SQL.
- External migration-evidence serialization remains the separately recorded
  H-18 architectural item; this slice preserves its safe interim operation and
  does not invent a second lock protocol.
- Parking predicate: additional historical receipt predicates, external
  administrative-write serialization, broader role-grant redesign, and
  lifecycle/UI changes are parked unless they prevent this exact recovery from
  committing atomically before later pending SQL.

Parked hardening: none.

## Verification

- `python -m py_compile atlas_brain/main_eom.py atlas_brain/services/crm_provider.py atlas_brain/storage/migrations/__init__.py atlas_brain/storage/migrations/reconciliation.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_render_profile.py tests/test_migration_content_integrity_preflight.py tests/test_migrations_runner.py` — passed.
- `python -m pytest -q tests/test_migration_content_integrity_preflight.py` — `157 passed`.
- `python -m pytest -q tests/test_migrations_runner.py` — `90 passed, 1 skipped`; focused `-k '386_forward_recovery or 386_recorded_recovery'` cases: `15 passed, 76 deselected`.
- `python -m pytest -q tests/test_eom_render_profile.py -k 'missed_call_recovery_migration_helper'` — `1 passed, 63 deselected`; `python -m pytest -q tests/test_eom_lead_conversion.py -k 'workflow_enrolls_won_loss_runtime_paths'` — `1 passed, 224 deselected`.
- `ATLAS_MIGRATION_TEST_DATABASE_URL=<disposable PostgreSQL test database> python -m pytest -q tests/test_eom_lead_conversion_integration.py -k 'selected_390_does_not_run_without_the_weak_386_precondition or nocodb_cannot_mutate_won_lead_with_unsettled_cancellation or won_lead_loss_cancels_first_clean_and_revokes_pending_draft'` — `3 passed, 110 deselected`; this proves a selected 390 is inert without the weak precondition, while the recovered case proves non-admin denial, guard ownership/read, former-owner denial, exact catalog ordering, attacker-temp isolation, all four predicate-field fence coverage, rollback of staged completion evidence after an injected later failure, retry to one completion/loss pair, and the normal Calendar-confirmed path.
- `python -m ruff check --ignore E402 atlas_brain/main_eom.py atlas_brain/services/crm_provider.py atlas_brain/storage/migrations/__init__.py atlas_brain/storage/migrations/reconciliation.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_render_profile.py tests/test_migration_content_integrity_preflight.py tests/test_migrations_runner.py` — passed; `main_eom.py` retains the repository's pre-existing intentional E402 import order.
- `git diff --check` — passed.
- Controlled read-only target preflight — expected exit `2`: 386 is `recovery_required` with the trusted guard role ready but with its function ownership and lifecycle read still absent, and 379 remains independently source-unavailable. No target SQL ran. No local Unit Gate ran; broad checks remain on GitHub.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/main_eom.py` | 3 |
| `atlas_brain/services/crm_provider.py` | 37 |
| `atlas_brain/storage/migrations/390_eom_won_loss_direct_sql_fence_recovery.sql` | 147 |
| `atlas_brain/storage/migrations/__init__.py` | 208 |
| `atlas_brain/storage/migrations/reconciliation.py` | 571 |
| `plans/PR-H18-EOM-Won-Loss-Fence-Forward-Recovery.md` | 383 |
| `tests/test_eom_lead_conversion.py` | 2 |
| `tests/test_eom_lead_conversion_integration.py` | 267 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_migration_content_integrity_preflight.py` | 1 |
| `tests/test_migrations_runner.py` | 445 |
| **Total** | **2072** |
