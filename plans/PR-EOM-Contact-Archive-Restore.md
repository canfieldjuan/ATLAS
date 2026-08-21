# PR-EOM-Contact-Archive-Restore

## Why this slice exists

Website #253 (canonical child of the #105 CRM umbrella): operators need to
archive junk/stale CRM contacts out of the Atlas-backed directory and restore
them when archiving was wrong. Phase 0 (documented on #253) proved the only
archive today is `DatabaseCRMProvider.delete_contact` -- a generic
`SET status='archived'` reachable only from the MCP tool, with no tenant
scoping at the provider layer, no lifecycle audit event, no idempotency
receipt, and no transition validation -- and that no contact restore exists at
all. The portal cannot park a contact, and nothing can bring one back.

### Problem-derived contract

- Root cause: the status axis of `contacts` has no canonical, receipted
  transition pair the operator boundary can call -- archive exists only as an
  unaudited generic mutation and restore does not exist, so the directory can
  never offer archive/restore without inventing an unsafe side door.
- Correct fix must touch/change: `atlas_brain/services/crm_provider.py` (two
  transition methods cloned from the lost/reopen receipt machine, plus a
  `lifecycle` admission parameter on the directory read),
  `atlas_brain/services/eom_lead_conversion.py` (command dataclasses +
  delegators), `atlas_brain/eom_api/funnel.py` (two POST routes, the closed
  `lifecycle` query filter with a page-homogeneity check, the widened response
  `status` Literal, three new capability-manifest entries), migration 388
  (widen the disposition operation-key index to the two new event types), and
  the EOM pipeline workflow enrollment for the new files.
- Must not change: the legacy MCP `delete_contact` tool and its provider
  method (its one caller compensates tenancy; unification is a #247 note),
  the lost/reopen transitions and their replay semantics, the won-loss
  teardown protocol, the operator mutation boundary, the pipeline `/leads`
  read (already status-scoped), billing-recipient eligibility, and every
  non-EOM surface.

## Scope (this PR)

Ownership lane: eom-crm/contact-archive-restore
Slice phase: Vertical slice
Max files: 11

1. `POST /eom-funnel/contacts/{contact_id}/archive` and
   `POST /eom-funnel/contacts/{contact_id}/restore`: tenant-pinned,
   actor-attributed, Idempotency-Key-receipted status transitions with
   truthful replays, ABA supersession refusal, cross-contact key-ownership
   rejection, a won-stage-lead archive refusal deferring to the lost flow,
   and the won-loss cancellation fence on both directions; each writes a
   `contact_archived` / `contact_restored` lifecycle event carrying actor,
   operation key, previous and resulting status, and the stage snapshot.
2. The contact directory gains a closed `lifecycle: active|archived` filter
   (default `active`) whose admission predicate, widened response Literal,
   and route-level page-homogeneity check enforce one status per page; the
   capability manifest advertises `contact.archive`, `contact.restore`, and
   `contact.directory.archived` derived from registered routes.
3. Proof: `tests/test_eom_contact_archive.py` (route boundary, capability
   truth, a generated 48-case admission matrix judged by a spec-derived
   oracle, and real-Postgres receipt/replay/fence/directory proofs) plus
   lifecycle-filter closure tests in `tests/test_eom_contact_directory.py`.

### Review Contract

- Acceptance criteria:
  - Foreign-tenant target reads as missing -- settled by
    `tests/test_eom_contact_archive.py::test_transition_admission_holds_across_operations_tenants_and_kinds`
    (tenant axis) and the removed-tenant-check negative control run recorded
    in Verification.
  - Admission over operation x tenant x status x kind/stage is class-closed
    per the 3k.3 evidence-gated mechanism: cases are GENERATED with
    `itertools.product` over four grammar axes and judged by the
    spec-derived `_transition_oracle`, not a sampled fixture list -- settled
    by the same matrix test.
  - A fresh transition writes exactly one sequenced receipt naming the
    actor, the Idempotency-Key, the prior and new status values, and the
    stage snapshot -- settled
    by `test_archive_writes_a_sequenced_receipt_and_replays_truthfully`.
  - A replay is truthful: 200-idempotent only while the row still holds the
    receipt's status and no later disposition owns it; the ABA shape
    (archive, restore, re-archive under a new key) refuses both original
    keys -- settled by `test_restore_round_trip_and_aba_replays_are_refused`.
  - An operation key belongs to exactly one contact per event type --
    settled by `test_an_operation_key_belongs_to_exactly_one_contact`.
  - An unresolved won-loss cancellation blocks both transitions -- settled
    by `test_the_wonloss_cancellation_fence_blocks_both_transitions`.
  - The two lifecycle views are disjoint and restore returns a row exactly
    once -- settled by
    `test_directory_lifecycle_views_are_disjoint_and_restore_returns_once`
    and the route-level homogeneity tests
    (`test_an_archived_row_can_never_be_emitted_by_the_active_view`,
    `test_an_active_row_can_never_be_emitted_by_the_archived_view`).
  - The three new capability names are advertised only from registered
    routes and ride the `/eom-funnel/leads` envelope -- settled by
    `test_the_transitions_and_archived_view_are_advertised_from_real_routes`
    and `test_the_lead_review_envelope_advertises_all_three_names`.
  - Both deployed entrypoints serve both POST paths -- settled by
    `test_every_deployed_entrypoint_serves_both_transition_routes`.
- Reachability proof: `POST /api/v1/eom-funnel/contacts/{id}/archive` on the
  live service flips the row's status, excludes it from the default
  directory page, and admits it to `?lifecycle=archived`; exercised in tests
  against both `atlas_brain.main:app` and `main_eom:app`, and live-probed
  after deployment per the #253 verification plan.
- Affected surfaces: EOM funnel router (two new POST routes, one widened GET
  filter, capability manifest), funnel CRM provider (two new transaction
  methods, directory predicate), `eom_lead_lifecycle_events` ledger (two new
  event types), migration 388 (index widen), EOM pipeline workflow file
  list. Single consumer: the tracker proxy (website #253's next PR).
- Risk areas: replay truthfulness under the ABA shape; tenant scoping of the
  new writes; leakage between the two directory lifecycle views; deploy skew
  (an old build receiving `lifecycle=` or a new capability name); index
  replacement on a ledger table; interaction with the won-loss cancellation
  protocol.
- Reviewer rules triggered: R1 (tenant/auth boundary), R2 (migration/index
  replacement on a ledger table), R3 (replay truthfulness and echo
  contracts), R4 (SQL/schema change), R5 (migration safety), R7 (capability
  truthfulness), R8 (advisory-lock/transaction execution model, declared
  below), R11 (guard class closure per 3k.3).

### Boundary-change enumeration

- Boundary path/seam: `archive_eom_contact` / `restore_eom_contact`
  admission (new); `list_eom_contact_directory` status admission (widened
  from a literal `'active'` to the closed `lifecycle` parameter);
  `_reject_unknown_contact_directory_filters` set (adds `lifecycle`);
  directory response `status` Literal (widened) plus the new route-level
  page-homogeneity check.
- Replaced-path behaviors: none replaced -- the legacy `delete_contact`
  remains untouched with its single MCP caller; the directory's default
  behavior (active-only) is unchanged for existing callers.
- Guard-relevant fields: `business_context_id`, `contact_type`,
  `lead_stage`, `status`, `operation_key`, `lifecycle_sequence`,
  `event_type`, `lifecycle` query parameter.
- Closure declarations (one per decision-driving set; membership /
  source / outside-set behavior):
  - Directory lifecycle values: CLOSED, ENUMERATED as
    `_CONTACT_DIRECTORY_LIFECYCLES = ("active", "archived")` (funnel) and
    `_EOM_CONTACT_DIRECTORY_LIFECYCLES` (provider); outside-set -> 422 at
    the route, ValueError at the provider, and the page-homogeneity check
    500s any row whose status is outside the requested member.
  - Archive-disposition event types: CLOSED, ENUMERATED as
    `_EOM_CONTACT_ARCHIVE_DISPOSITION_EVENTS = ("contact_archived",
    "contact_restored")`; outside-set events are invisible to the
    supersession query by design (the legacy MCP writers that bypass
    receipts are the #247 unification item, recorded in Deferred).
  - Directory query parameters: CLOSED, ENUMERATED as
    `_CONTACT_DIRECTORY_QUERY_PARAMS` (now including `lifecycle`);
    outside-set names -> 422 via
    `_reject_unknown_contact_directory_filters`, which is also the
    deploy-skew defense on older builds.
  - Capability names: CLOSED, ENUMERATED as `_CAPABILITY_ROUTES` keys with
    membership DERIVED from registered routes at serve time; outside-set
    names are simply absent and every consumer treats absence as
    disable-the-control.
  - Admitted contact kinds for the transitions: CLOSED, DERIVED from
    `EOM_OPERATOR_CONTACT_TYPES`; outside-set kinds -> typed 409.
  - Transition status preconditions: CLOSED, ENUMERATED per direction
    (archive admits only `active`; restore admits only `archived`);
    outside-set current status -> typed 409, with the won-stage active
    lead carved out toward the lost flow by a dedicated 409.
- Caller x input shape: tracker proxy (the only caller) x
  {archive|restore} x {active, archived, foreign-tenant, missing, won-lead,
  non-directory-kind rows} x {fresh key, replayed key, foreign-owned key,
  malformed key}; directory caller x {no lifecycle, lifecycle=active,
  lifecycle=archived, unknown lifecycle, unknown parameter}.

### Deployed-config probing

- Deployed/default config values: no new config. The routes inherit
  `EOMFunnelConfig` (api_enabled + service token) and the actor dependency.
- Explicit value probe: valid bearer + actor + well-formed key -> 201/200
  (route tests).
- Absent value probe: missing bearer -> 401; missing actor -> 422; missing
  or malformed Idempotency-Key -> 422 (route tests).
- Default-session/default-context probe: `lifecycle` omitted -> the provider
  receives `active`
  (`test_omitting_lifecycle_defaults_to_the_active_view`), so existing
  callers see the pre-slice directory unchanged.
- Side-effect ordering: locks (contact, operation, won-loss execution) ->
  replay receipt read -> FOR UPDATE row read -> tenant check (404 for
  foreign or missing targets BEFORE any key-ownership disclosure) ->
  key-ownership probe -> replay validation -> admission checks ->
  cancellation fence -> guarded UPDATE -> receipt INSERT, all inside one
  transaction, so no observer sees a status flip without its receipt.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_lead_conversion.py`
- `atlas_brain/storage/migrations/388_eom_contact_archive_disposition_index.sql`
- `plans/PR-EOM-Contact-Archive-Restore.md`
- `tests/contact_write_boundary/baseline.json`
- `tests/test_contact_write_boundary.py`
- `tests/test_eom_contact_archive.py`
- `tests/test_eom_contact_directory.py`
- `tests/test_eom_lead_conversion.py`

## Mechanism

Archive/restore clone the lost/reopen transition machine onto the status
axis. Each transition: takes sorted advisory xact locks for the contact and
the operation key (both directions share the contact lock so they serialize),
plus the won-loss execution lock; reads a replay receipt from
`eom_lead_lifecycle_events` keyed (contact, event type, operation key);
rejects a key owned by another contact; reads the row FOR UPDATE and treats a
foreign tenant as missing; validates a replay truthfully (receipt shape,
current status, and no later archive/restore receipt by
`lifecycle_sequence` -- both event types postdate migration 363, so a NULL
sequence is treated as superseded); refuses invalid fresh transitions with
typed 409s (already archived/active under a different key, non-directory
kind, won-stage lead toward the lost flow); asserts the won-loss cancellation
fence; then performs a state-scoped UPDATE and inserts the receipt event
recording the actor, the source, the Idempotency-Key, the prior and new
status values, and the stage snapshot in one transaction. The directory read parameterizes its
status predicate with the closed `lifecycle` filter; the route validates the
filter against the closed set, forwards it, and fails closed (500) on any
returned row whose status differs from the requested lifecycle -- the second
enforcement the narrowed Literal used to provide alone. Capability names are
derived from registered routes as before; `contact.directory.archived` maps
to the directory GET deliberately, because only a build carrying the map
entry (and therefore the `lifecycle` code) can advertise it, and an older
build 422s the unknown parameter anyway.

### Execution model (3k.4)

Admitted execution model: one PostgreSQL transaction per transition call on
one connection. Seams and the guarantee that closes each:

- Same-contact concurrency (archive vs archive, archive vs restore):
  serialized by `pg_advisory_xact_lock` on the shared
  `eom-contact-archive:contact:{id}` key, taken in sorted order with the
  operation-key lock before any read; the locks release only at
  commit/rollback, so admission checks and the UPDATE+INSERT pair are
  atomic with respect to every other transition on that contact.
- Same-key concurrency across contacts: the operation-key advisory lock
  serializes two calls reusing one key, and the partial unique index
  `(contact_id, event_type, operation_key)` is the durable backstop -- even
  a writer that somehow bypassed the locks cannot commit two receipts for
  one key+type on one contact.
- Lost-update on the row: the `FOR UPDATE` read pins the row version; the
  final UPDATE re-asserts the full observed state (tenant, type, stage via
  IS NOT DISTINCT FROM, status), so any interleaved change that slipped a
  guard turns the UPDATE into zero rows and a loud RuntimeError rollback
  rather than a silent overwrite.
- Replay chronology: `lifecycle_sequence` is a database-owned nextval
  default (migration 363), so receipt ordering is total and monotonic per
  append regardless of application clocks; the supersession probe compares
  only sequences of the two receipt event types (closure declared above).
- Won-loss teardown interaction: the `eom-won-lead-loss:execution:{id}`
  advisory lock plus the requested/completed lifecycle-event fence refuse a
  status flip while a Calendar cancellation is executing or unreconciled,
  in BOTH directions.
- Invariants across every admitted interleaving: (1) at most one receipt
  per (contact, event type, key); (2) a 200 replay implies the row
  currently holds the receipt's resulting status AND no later
  archive-family receipt exists; (3) a fresh 201 implies exactly one
  receipt was appended in the same transaction as its status flip; (4) a
  foreign-tenant or missing target reads 404 before any key-ownership
  disclosure.
- Explicit assumption: writers outside the receipt system (legacy MCP
  `delete_contact` / `update_contact` status writes) do not advance this
  chronology; their unification is the #247 item recorded in Deferred, and
  invariant (2) is stated over the receipted writers this slice ships.

## Intentional

- The legacy MCP `delete_contact` stays untouched; unifying it onto
  `archive_eom_contact` is recorded in #247 rather than widening this slice.
- Archive refuses an ACTIVE won-stage lead (typed 409) instead of running
  won teardown itself: the lost flow owns Calendar cancellation, and a
  second teardown door is the exact duplicate-framework mistake this arc
  refuses. Archived-won rows (creatable only via the legacy tool) remain
  restorable so nothing is stranded.
- Restore re-checks no identity ambiguity: contact info is not identity
  (#105/#107), no uniqueness constraint exists to violate, and the operator
  mutation boundary already 409s ambiguous matches. Merge UX stays in #247.
- The estimate-booking fence is NOT applied to archive (only the won-loss
  cancellation fence is): archive is reversible visibility parking that
  asserts nothing about a booking's disposition, and blocking it would make
  exactly the stuck rows unarchivable. Lost keeps that stricter fence
  because lost is a terminal business claim.
- `contact.directory.archived` shares the directory's registered route on
  purpose; the map-entry-per-build derivation plus the unknown-parameter 422
  give two independent skew defenses.

## Deferred

- Unify MCP `delete_contact` onto the canonical archive transition (#247).
- Tracker relays/proofs and the portal Archived view land as the next PRs of
  website #253, after this deploys.

Parked hardening: none.

## Verification

- `tests/test_eom_contact_archive.py` (15) + `tests/test_eom_contact_directory.py`
  (39) green against real Postgres (`ATLAS_MIGRATION_TEST_DATABASE_URL`, disposable
  container); `tests/test_eom_lead_conversion.py` (224),
  `tests/test_eom_funnel_capability_manifest.py`, `tests/test_migrations_runner.py`
  green.
- Negative controls (each: enforcement removed -> named test failed -> code
  restored): page-homogeneity check removed -> both leak tests failed;
  tenant check removed from both transitions -> admission matrix failed;
  ABA supersession helper stubbed false -> ABA test failed; key-ownership
  probe disabled -> key-ownership test failed.
- Pending before push: none.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| `atlas_brain/eom_api/funnel.py` | 110 |
| `atlas_brain/services/crm_provider.py` | 429 |
| `atlas_brain/services/eom_lead_conversion.py` | 46 |
| `atlas_brain/storage/migrations/388_eom_contact_archive_disposition_index.sql` | 43 |
| `plans/PR-EOM-Contact-Archive-Restore.md` | 320 |
| `tests/contact_write_boundary/baseline.json` | 2 |
| `tests/test_contact_write_boundary.py` | 8 |
| `tests/test_eom_contact_archive.py` | 943 |
| `tests/test_eom_contact_directory.py` | 38 |
| `tests/test_eom_lead_conversion.py` | 1 |
| **Total** | **1945** |
