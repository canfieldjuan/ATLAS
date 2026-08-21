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
  - A fresh transition writes exactly one sequenced receipt with actor,
    operation key, previous/resulting status, and stage snapshot -- settled
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
  replacement on a ledger table), R3 (idempotency and replay truthfulness),
  R4 (SQL/schema change), R5 (migration safety), R7 (capability
  truthfulness), R11 (guard class closure per 3k.3).

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
  replay receipt read -> key-ownership probe -> FOR UPDATE row read ->
  tenant check -> replay validation -> admission checks -> cancellation
  fence -> guarded UPDATE -> receipt INSERT, all inside one transaction, so
  no observer sees a status flip without its receipt.

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
recording actor, source, operation key, previous/resulting status, and the
stage snapshot in one transaction. The directory read parameterizes its
status predicate with the closed `lifecycle` filter; the route validates the
filter against the closed set, forwards it, and fails closed (500) on any
returned row whose status differs from the requested lifecycle -- the second
enforcement the narrowed Literal used to provide alone. Capability names are
derived from registered routes as before; `contact.directory.archived` maps
to the directory GET deliberately, because only a build carrying the map
entry (and therefore the `lifecycle` code) can advertise it, and an older
build 422s the unknown parameter anyway.

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
| `atlas_brain/services/crm_provider.py` | 421 |
| `atlas_brain/services/eom_lead_conversion.py` | 46 |
| `atlas_brain/storage/migrations/388_eom_contact_archive_disposition_index.sql` | 43 |
| `plans/PR-EOM-Contact-Archive-Restore.md` | 246 |
| `tests/contact_write_boundary/baseline.json` | 2 |
| `tests/test_contact_write_boundary.py` | 8 |
| `tests/test_eom_contact_archive.py` | 785 |
| `tests/test_eom_contact_directory.py` | 38 |
| `tests/test_eom_lead_conversion.py` | 1 |
| **Total** | **1705** |
