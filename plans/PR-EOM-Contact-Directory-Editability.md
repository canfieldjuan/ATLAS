# PR-EOM-Contact-Directory-Editability

## Why this slice exists

Website issue #247 item 7 records a live operator failure: an active Lost lead
is intentionally discoverable in the CRM contact directory, but the existing
operator-mutation boundary rejects edits to a lead whose stage is outside its
supported active-stage set. The portal therefore offers an action that the
authoritative service will refuse.

The full diff is expected to exceed the 400 LOC soft cap only because the
required plan and cross-path regression coverage are part of the receipt; the
production mechanism stays three narrow modules with no migration or new route.

### Problem-derived contract

- Root cause: Atlas makes the editability decision only inside
  `DatabaseCRMProvider.mutate_eom_operator_contact_atomic`, while
  `DatabaseCRMProvider.list_eom_contact_directory` deliberately returns all
  active leads, including `lead_stage=lost`. The directory response exposes
  the stage but no authoritative decision or reason, so a consumer can only
  offer Edit globally or copy Atlas stage policy.
- Correct fix must touch/change: the canonical EOM operator-contact domain
  module must own one closed, row-state editability decision over stored
  contact type, lifecycle status, and lead stage. The existing mutation guard
  and the directory projection must consume that one decision. The funnel
  directory schema must serialize `editable` and a closed
  `editBlockedReason`, and advertise a distinct capability tied to the
  existing directory route so downstream services can require this response
  semantic rather than infer it from route reachability. Focused API and
  provider tests must prove active supported leads/customers remain editable,
  Lost leads remain visible but are non-editable with the stage reason, and
  archived rows are non-editable with the lifecycle reason.
- Must not change: lead-stage transitions, the definition of a Lost lead,
  directory membership/search/pagination, archive/restore behavior, existing
  authentication and tenant scoping, edit-field validation, mutation status
  codes, database schema/migrations, or any tracker/Website code. In
  particular, this PR must not reimplement the stage list in a downstream
  client or hide Lost leads to mask the offered-but-fails Edit control.

## Scope (this PR)

Ownership lane: eom/contact-directory-editability
Slice phase: Vertical slice

1. Publish an Atlas-owned, per-directory-contact editability verdict and
   closed reason code from the same lifecycle policy the mutation boundary
   already enforces.
2. Version that additive directory response semantic with a capability name
   and prove it through the real `/api/v1/eom-funnel/contact-directory`
   entrypoint on both shipped Atlas app objects.

### Review Contract

- Acceptance criteria:
  - [ ] An authenticated directory request for an active Lost lead returns the
    same contact row with `editable: false` and
    `editBlockedReason: "not_editable_stage"`; it does not filter the row out.
    Settled by `tests/test_eom_contact_directory.py` route and real-Postgres
    projection cases.
  - [ ] Active customers and active leads in the stages already admitted by
    the mutation boundary return `editable: true` and no block reason, while
    an archived directory row returns `editable: false` with the closed
    lifecycle reason. Settled by the shared-policy and route contract tests.
  - [ ] The mutation boundary uses the same canonical decision rather than a
    second copy of the supported-stage rule, and retains its existing refusal
    behavior for non-editable records. Settled by direct policy tests plus the
    existing mutation integration coverage exercised in the focused test run.
  - [ ] `contact.directory.editability` is advertised only by a build that
    serves the existing `GET /eom-funnel/contact-directory` contract carrying
    these fields. Settled by capability-manifest and deployed-entrypoint tests.
  - [ ] The additive fields preserve the existing directory envelope,
    authentication, tenant filtering, lifecycle filtering, and keyset
    pagination. Settled by `tests/test_eom_contact_directory.py`.
- Reachability proof: invoke the real mounted
  `/api/v1/eom-funnel/contact-directory` route on both `atlas_brain.main:app`
  and `atlas_brain.main_eom:app`, with the service bearer and actor headers,
  and assert the serialized per-row verdict.
- Affected surfaces: Atlas EOM funnel response schema and capability manifest;
  canonical operator-contact domain policy; database CRM directory projection;
  no browser or tracker surface in this PR.
- Risk areas: public response compatibility, mismatch between read and write
  eligibility, lifecycle-status ordering, and stale deployments that expose
  the route without the new fields.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: canonical row-state editability decision used by
  `mutate_eom_operator_contact_atomic` and emitted by the contact-directory
  projection; additive Pydantic serialization boundary for directory items.
- Replaced-path behaviors: the mutation duplicated inline stage predicate is
  replaced with the canonical decision; directory admission stays status-based
  and does not become a stage filter.
- Guard-relevant fields: `contact_type`, `status`, and `lead_stage`; no
  browser-supplied eligibility field is accepted.
- Caller x input shape: the database provider and test CRM yield canonical
  stored-row mappings; the authenticated GET route serializes the derived
  verdict. Existing mutation commands remain unchanged.

### Deployed-config probing

N/A - this slice adds no environment variable, fallback, or configuration
branch. It remains behind the existing EOM funnel enablement and authentication
dependencies; the capability is the deployment-proof mechanism for the new
response semantic.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_crm_mutations.py`
- `plans/PR-EOM-Contact-Directory-Editability.md`
- `tests/test_eom_contact_directory.py`
- `tests/test_eom_lead_conversion_integration.py`

## Mechanism

The domain module returns a small immutable decision with exactly one of the
closed outcomes: editable, blocked by lifecycle status, blocked by unsupported
contact kind, or blocked by lead stage. Its order preserves the existing
mutation behavior: lifecycle status wins where it already won before identity
conflict handling; a Lost lead remains discoverable but is blocked by stage.

`DatabaseCRMProvider` uses that decision when it validates an existing-contact
edit and when it turns each canonical directory row into the closed directory
projection. The route schema requires a boolean and either `null` or one of the
same closed reason codes. `contact.directory.editability` maps to the existing
GET route, so older Atlas deployments cannot advertise the field contract and
downstream relays can fail closed rather than treating route availability as
semantic compatibility.

## Intentional

- A non-editable verdict is a truthful lifecycle precondition, not a promise
  that every future mutation will succeed: identity-collision checks remain
  request-specific and fail closed at the existing write boundary.
- Lost leads stay in the active directory. Removing them would destroy the
  operator ability to find/reopen/inspect them instead of correcting the
  offered-but-fails Edit control.
- The capability has its own name even though it uses the same route, matching
  the existing `contact.directory.archived` and `contact.field_clear` rollout
  pattern for response semantics that older builds do not understand.

## Deferred

- Tracker capability proof/parser and Website rendering/translated explanation
  will consume this Atlas contract in the two downstream slices for Website
  #247 item 7. They must not carry a copied lead-stage list.
- Collision-resolution UX, cross-tab recovery, same-field concurrency, address
  and `customer_type` editing remain deferred exactly as listed in Website
  #247.
- No new endpoint, database table, migration, or generic edit-workflow
  redesign is justified by this read/write contract correction.

Parked hardening: request-specific identity conflict handling, general edit
workflow redesign, and downstream recovery/copy changes are parked unless they
block the per-row verdict or violate the existing mutation safety boundary.

## Verification

- `pytest tests/test_eom_contact_directory.py -q` - 43 passed, 6 skipped;
  the skipped real-Postgres cases require `ATLAS_MIGRATION_TEST_DATABASE_URL`.
- `pytest tests/test_eom_lead_conversion_integration.py -q -k
  "unsupported_lead_stages or rejects_inactive_legacy_lead or
  rejects_unsupported_existing_contact_type"` - 4 skipped, 108 deselected;
  the focused real-Postgres guard cases, including Lost, require the same
  unavailable local migration-test database.
- `ruff check --ignore F401` on the five changed Python files - passed. The
  excluded F401 is pre-existing on the `EOM_BUSINESS_CONTEXT_ID` re-export in
  `atlas_brain/services/eom_crm_mutations.py`, verified against `origin/main`.
- `bash scripts/check_ascii_python.sh` - passed.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` -
  passed with no guard-shaped change missing a property test.
- `python scripts/sync_pr_plan.py
  plans/PR-EOM-Contact-Directory-Editability.md --check` - passed.
- Pending before push: cold diff reconstruction with file-and-line citations
  in the PR body, then `bash scripts/push_pr.sh` as the single local-review
  entrypoint.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 23 |
| `atlas_brain/services/crm_provider.py` | 65 |
| `atlas_brain/services/eom_crm_mutations.py` | 63 |
| `plans/PR-EOM-Contact-Directory-Editability.md` | 191 |
| `tests/test_eom_contact_directory.py` | 98 |
| `tests/test_eom_lead_conversion_integration.py` | 27 |
| **Total** | **467** |
