# PR-EOM-Known-Contact-Source-Versions

## Why this slice exists

Tracker issue [#167](https://github.com/canfieldjuan/eom-timetracker/issues/167)
proved a real stale-evidence path: its customer-type refresh receives a type
snapshot from Atlas before the tracker takes its local write lock. A concurrent
tracker type mutation can therefore leave the local mirror with a newer Atlas
answer, then the refresh can overwrite it with its older answer. The tracker
cannot order those answers because the canonical `known-contacts` projection
returns `customerTypes` but no source version.

This is the Atlas provider prerequisite, deliberately first in deployment
order.  It adds the evidence the tracker needs; it does **not** claim to resolve
#167 until the tracker consumes and fences on that evidence in its follow-up PR.

### Problem-derived contract

- Root cause: `GET /eom-funnel/known-contacts` projects a mutable Atlas
  `customer_type` without a database-owned, monotonic version that produced it.
  `contacts.updated_at` cannot substitute: the operator mutation path writes it
  from the application host clock while the backfill writes it from PostgreSQL,
  so a later committed type change can carry an earlier timestamp. The data is
  therefore naked classification evidence: a consumer can tell what Atlas said,
  but cannot safely distinguish a current response from an older one after
  another observed Atlas mutation.
- Correct fix must touch/change: add one atomic Atlas migration that gives every
  `contacts` row a positive `customer_type_revision`, backed by a database
  sequence and a database trigger. The trigger must assign a fresh revision on
  every insert and every actual `customer_type` change, preserving it for
  unrelated contact changes and preventing application or script clocks from
  deciding order. The trigger must retain its database sequence privilege when
  the limited `atlas_nocodb` CRM role performs an allowed ordinary contact
  insert, without widening that role's sequence or protected-column grants.
  Extend the canonical provider query,
  `EOMKnownContactsResponse`, and route projection so every resolved,
  tenant-scoped contact has an integer `customerTypeRevisions` entry paired
  exactly with `customerTypes`. Require the new source column at enabled-funnel
  startup, and add route and real-PostgreSQL proofs for map attribution,
  database ownership, monotonic type writes, and unchanged legacy id/type shape.
- Must not change: `knownContactIds` and `customerTypes` names and meanings;
  EOM funnel authentication, capability names, tenant filtering, request cap,
  identity-data disclosure policy, NocoDB's limited ordinary-CRM write
  capability, operator-contact mutation semantics, tracker code and schema,
  billing-recipient work, UI, payroll, and any unrelated contact projection.
  `updated_at` retains its existing uses but is not used as this ordering token.
  This PR adds no application-side contact mutation and no tracker-side ordering
  decision.

## Scope (this PR)

Ownership lane: eom-crm/customer-type-provenance
Slice phase: Production hardening
Max files: 11

1. Add an atomic migration that establishes database-owned,
   customer-type-specific revisions for existing and future `contacts` rows.
2. Add a parallel `customerTypeRevisions` response map for the current
   `known-contacts` projection only, backed by that database-owned revision.
3. Require the revision column when the enabled EOM funnel starts, so a partial
   deployment fails at readiness rather than issuing a broken type projection.
4. Prove the public EOM-funnel route returns an exact id/type/revision mapping
   from its tenant-scoped provider result and cannot include a caller-unsubmitted
   or unresolved id. Prove real PostgreSQL direct, backfill, and operator writes
   cannot make a later type change look older.
5. Record the migration's deliberate `contacts` backfill in the enforced
   contact-write inventory, keeping the existing write-boundary gate truthful.
6. Close the migration's operational and negative-proof boundary: document the
   safe rollback order and retained schema artifacts, prove the NocoDB role has
   neither direct revision-column nor sequence privileges, prove readiness
   rejects a present non-`BIGINT` revision column, enroll migration 367 in both
   EOM workflow path filters, and bring the operator enablement runbook's
   prerequisite and verification steps through migration 367.

### Review Contract

- Acceptance criteria:
  1. Every normal contact insert receives a positive database-owned revision;
     each actual `customer_type` update receives a strictly greater revision;
     a same-type or unrelated update preserves it; and a revision-only mutation
     is rejected while a supplied revision alongside a real type change is
     overwritten. The database trigger, rather than an application clock or one
     writer, enforces that class for direct,
     backfill, and operator paths; it does not break the permitted unprivileged
     NocoDB insert or grant the NocoDB role sequence/protected-column access.
     Settled by real-PostgreSQL migration tests that assert both permitted
     ordinary writes and denied sequence/revision-column privileges.
  2. `GET /eom-funnel/known-contacts` returns `customerTypeRevisions` for every
     `knownContactIds` member, keyed identically to `customerTypes`, with an
     integer source revision from the same tenant-scoped contact row; settled by
     `tests/test_eom_link_verification.py` route assertions.
  3. Unknown, foreign-tenant, and caller-unsubmitted ids have no type or source
     revision entry; settled by the existing real-entrypoint attribution and
     tenant tests extended for the new map.
  4. `knownContactIds`, `customerTypes`, auth, limit, and response status retain
     their existing behavior; settled by the focused route test file.
  5. The enabled funnel refuses a store with the revision column absent **or
     present with a non-`BIGINT` type. This provider PR does not claim #167
     closed; its tracker consumer remains
     a separately deployable follow-up, settled by the plan scope and cold diff.
  6. The committed contact-write inventory contains the migration's one-time
     revision backfill and otherwise matches the tree; settled by
     `tests/test_contact_write_boundary.py`.
  7. Both EOM workflow trigger filters name migration 367, and the enablement
     runbook requires/verifies its column and trigger; settled by direct
     workflow/runbook inspection.
  8. A normal runtime rollback retains the column, trigger, function, sequence,
     constraint, and revisions after code rollback; any destructive teardown is
     an explicit later DBA operation after every revision-aware deployment is
     gone. Settled by the migration and runbook rollback procedure.
- Reachability proof: the mounted `/eom-funnel/known-contacts` FastAPI route is
  exercised through `httpx.ASGITransport`; its JSON response is the observable
  provider contract consumed by the tracker.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/eom_api/funnel_store.py`, `atlas_brain/services/crm_provider.py`,
  `atlas_brain/storage/migrations/367_contacts_customer_type_revision.sql`, and
  the directly related EOM contact tests, workflow path enrollment, and EOM
  funnel enablement runbook.
- Risk areas: EOM tenant scope, narrow disclosure, response compatibility,
  migration atomicity, database ordering, NocoDB privilege preservation,
  fail-closed readiness typing, rollback safety, CI enrollment, and exact map
  attribution.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  authorization/privacy, R4 data and migration safety, R5 backward
  compatibility, R6 error handling, R8 concurrency/idempotency, R10
  maintainability, R12 deployment/CI, R14 codebase verification.

### Boundary-change enumeration

N/A - no admission, resolver, or capability decision changes. This is an
additive field on an already-authenticated, already-tenant-scoped projection;
its membership derives from the route's existing `known_ids` result. The
database trigger has a deliberately closed write rule: insert or actual type
change advances the revision; any other contact update retains it.

### Deployed-config probing

N/A - no configuration or environment fallback changes.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/367_contacts_customer_type_revision.sql`
- `docs/EOM_FUNNEL_ENABLEMENT_RUNBOOK.md`
- `plans/PR-EOM-Known-Contact-Source-Versions.md`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_backfill_eom_customer_type.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_link_verification.py`

## Mechanism

The atomic migration first creates a sequence and adds/backfills the revision
column under the migration runner's one transaction. A narrowly scoped,
safe-search-path `SECURITY DEFINER` `BEFORE INSERT OR UPDATE` trigger assigns
the next sequence value for every insert and every actual type change, then
preserves the old revision for a non-type update. Its ownership supplies the
sequence privilege for the permitted NocoDB CRM insert without granting NocoDB
sequence access. It therefore covers the operator, backfill, direct SQL, and
limited-NocoDB writers without relying on their clocks or on duplicated
application logic. Sequence gaps caused by a rolled-back transaction are
acceptable: ordering needs strict increase, not contiguity. The tracker will
compare revisions only for the same contact.

A normal runtime rollback rolls application code back first and retains the
additive revision column, positive constraint, trigger, function, and sequence:
old contact writers do not name the column and the trigger continues to fill it.
Physical removal is not an ordinary rollback, because it discards ordering
evidence. It is a later DBA-approved teardown only after every provider and
tracker consumer that knows the revision contract is gone; its safe order is
trigger, function, sequence ownership, column, then sequence.

`DatabaseCRMProvider.list_known_eom_contact_ids` already filters contact rows by
the EOM business context and requested ids. It will select
`customer_type_revision` with the existing id/type columns. The route will
construct a parallel `customerTypeRevisions` map from that same post-filtered
`known_ids` sequence, and the response model will serialize integer revisions.
The enabled-funnel readiness query verifies the source column before serving.

The membership set is **OPEN, DERIVED**: arbitrary contact ids can be submitted
within the existing request cap, while map membership is derived at runtime from
the already-filtered `known_ids`. For any unresolved, foreign-tenant, or
unsubmitted id, the safe/default behavior is no id, type, or revision entry.
There is no copied list to drift.

## Intentional

- A parallel map preserves the legacy `knownContactIds` list and
  `customerTypes` map instead of replacing either with contact objects.
- `customerTypeRevisions` is a narrow ordering token, not an identity
  projection; no name, email, phone, address, notes, tags, or lifecycle data is
  added.
- `updated_at` remains for its existing uses but is deliberately excluded from
  the ordering contract. The new trigger is database-owned and covers normal
  SQL writes, so no application host clock or backfill clock decides a revision.
- The `SECURITY DEFINER` trigger has a fixed `pg_catalog` search path and uses
  its triggering table's schema to resolve the owned sequence. It grants no
  direct sequence privilege to the NocoDB role.
- The revision changes only on insertion or an actual `customer_type` change;
  unrelated contact changes do not manufacture new classification evidence.
- The rollback procedure deliberately keeps migration 367's schema artifacts
  during a code rollback. It avoids a mixed-version `NOT NULL` insert failure
  and preserves source evidence; destructive teardown is explicitly deferred.
- Both workflow filters list migration 367 so a migration-only correction still
  runs the real PostgreSQL EOM suite. The runbook verifies the `BIGINT` column
  and trigger rather than treating an older 353–363 range as sufficient.
- The diff exceeds the normal 400-LOC review budget because P1 review
  remediation replaced an unsafe timestamp design with the migration and direct
  writer proofs required for the same provider contract. Splitting the migration
  from its published response would knowingly expose an ordering token that has
  no database enforcement, so the one root-cause repair remains one review unit.
- The tracker’s stale-write fence is intentionally deferred to its own PR.  A
  provider-only diff cannot truthfully claim the tracker now rejects stale
  responses.

## Deferred

- `canfieldjuan/eom-timetracker#167` tracker consumer: add a local source
  watermark, require valid revision evidence for refresh writes, and fence every
  tracker mirror writer. It may merge only after this provider response is
  deployed or it must fail closed on an absent revision map.

Parking predicate: park any additional contact field, provider refactor,
out-of-band Atlas write protocol that disables database triggers, UI change, or
tracker behavior that is not required to publish the narrow revision evidence
contract.

Parked hardening: none.

## Verification

- Ruff lint across the changed Python files -- passed.
- pytest -q tests/test_eom_link_verification.py
  tests/test_backfill_eom_customer_type.py
  tests/test_eom_lead_conversion_integration.py -- 24 passed, 88 skipped
  because ATLAS_MIGRATION_TEST_DATABASE_URL is not configured, and one
  pre-existing third-party `pynvml` FutureWarning from the mounted-app test.
- pytest -q tests/test_eom_write_boundary_audit.py -- 28 passed, 2 skipped.
- pytest -q tests/test_contact_write_boundary.py::test_repository_is_currently_clean
  tests/test_contact_write_boundary.py::test_baseline_file_matches_the_tree --
  2 passed.
- pytest -q tests/test_migrations_runner.py -- 30 passed, 1 skipped.
- An isolated local PostgreSQL transaction applied migration 367, then proved
  positive backfill, insert, non-type preservation, type-change advancement,
  forged-value override, and revision-only rejection. A second role probe
  proved `atlas_nocodb` can perform its existing permitted insert/update without
  any direct sequence grant.
- git diff --check; python scripts/sync_pr_plan.py --check
  plans/PR-EOM-Known-Contact-Source-Versions.md origin/main; python
  scripts/audit_plan_code_consistency.py --base-ref origin/main
  plans/PR-EOM-Known-Contact-Source-Versions.md; and python
  scripts/check_contact_write_boundary.py --baseline
  tests/contact_write_boundary/baseline.json -- passed.
- ruff format --diff reports only pre-existing whole-file formatter hunks on
  `origin/main`; the remediation adds no formatter hunk, so this slice avoids a
  drive-by formatting sweep.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `atlas_brain/eom_api/funnel.py` | 11 |
| `atlas_brain/eom_api/funnel_store.py` | 12 |
| `atlas_brain/services/crm_provider.py` | 16 |
| `atlas_brain/storage/migrations/367_contacts_customer_type_revision.sql` | 112 |
| `docs/EOM_FUNNEL_ENABLEMENT_RUNBOOK.md` | 16 |
| `plans/PR-EOM-Known-Contact-Source-Versions.md` | 276 |
| `tests/contact_write_boundary/baseline.json` | 2 |
| `tests/test_backfill_eom_customer_type.py` | 20 |
| `tests/test_eom_lead_conversion_integration.py` | 105 |
| `tests/test_eom_link_verification.py` | 157 |
| **Total** | **729** |
