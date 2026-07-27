# PR-EOM-Complaints-Tracking

## Why this slice exists

Issue #2168, split from the #2151 EOM CRM audit, records that customer-service
complaints are currently only free-text `contact_interactions` notes. That
representation cannot preserve a complaint's lifecycle fields or answer the
operator question "which complaints are open?" This vertical slice adds the
smallest tenant-safe ticket lifecycle reachable through the existing CRM MCP.

### Problem-derived contract

- Root cause: complaint notes have no dedicated row identity or structured
  status, priority, assignee, resolution, and close timestamp. Consequently,
  CRM reads cannot filter open complaints and CRM writes cannot update or
  close one complaint without rewriting undifferentiated interaction text.
- Correct fix must touch/change: add an additive, tenant-stamped,
  contact-linked customer-service ticket table; add canonical provider
  operations that scope every query/mutation in SQL; expose open, list, update,
  and close through the CRM MCP; atomically claim a visible NULL-legacy contact
  when the first ticket is opened; make close retries preserve the first
  resolution while returning success to simultaneous callers; and prove the
  real MCP-to-Postgres lifecycle, including foreign tenant refusal,
  filter-before-limit behavior, and the close race, in the existing EOM CRM CI
  lane.
- Must not change: consumer-review `complaint_*` tables, Content Ops support
  tickets, existing `contact_interactions`, contact read-scoping semantics,
  the HTTP contacts API tracked by #2170, email-store tenancy tracked by #2171,
  complaint intake/UI/notifications, or any public complaint taxonomy beyond
  the issue's required fields.

## Scope (this PR)

Ownership lane: eom-crm/complaints
Slice phase: Vertical slice

1. Add a dedicated EOM customer-service ticket lifecycle with `open` and
   `closed` state, caller-supplied priority and assignee, and immutable-on-retry
   close resolution.
2. Add tenant-safe CRM provider and MCP operations to open, list, update, and
   close tickets, defaulting list queries to open tickets.
3. Add structural migration coverage and a real-Postgres reachability test
   through the exported MCP functions, enrolled in the existing EOM CRM
   workflow.

### Review Contract

- Acceptance criteria:
  - [ ] Migration creates only the dedicated contact-linked,
        tenant-stamped customer-service ticket structure plus open-queue and
        per-contact query indexes; existing CRM data is unchanged.
  - [ ] Opening a ticket through CRM MCP atomically claims a visible
        NULL-context contact for the effective tenant, persists all supplied
        lifecycle fields, and refuses archived, missing, or foreign contacts.
  - [ ] Listing defaults to `open`, applies tenant/contact/status/priority/
        assignee predicates before ordering and pagination, and never exposes a
        foreign tenant's tickets.
  - [ ] Update mutates only an open ticket in the effective tenant and refuses
        missing, foreign, closed, invalid, or empty updates.
  - [ ] Close records the first resolution and timestamp; simultaneous and
        later retry callers succeed without overwriting either value.
  - [ ] Existing CRM tools and unrelated complaint/ticket stores are unchanged.
  - [ ] The CRM MCP documentation/inventory reflects all exported tools.
- Reachability proof: call the real exported CRM MCP open/list/update/close
  functions with the production `DatabaseCRMProvider` against disposable
  Postgres, then assert the persisted tenant claim, filtered result order,
  updated ticket, closed state, and idempotent resolution.
- Affected surfaces: CRM MCP public tools, canonical direct-Postgres CRM
  provider, additive Postgres migration, MCP tool inventory, EOM CRM GitHub
  Actions enrollment, focused migration and integration tests.
- Risk areas: cross-tenant reads/writes, legacy-contact claim races,
  status-transition races, retry overwrites, pagination page starvation,
  backward-compatible MCP inventory, migration deployment order.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `CLAUDE.md`
- `README.md`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/347_customer_service_tickets.sql`
- `plans/INDEX.md`
- `plans/PR-EOM-Complaints-Tracking.md`
- `plans/archive/PR-EOM-Lead-Pipeline.md`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/test_eom_complaints_integration.py`
- `tests/test_migrations_runner.py`
- `tests/test_pre_push_audit.py`

## Mechanism

Migration `347` creates `customer_service_tickets` with a contact foreign key,
required tenant id, bounded structured fields, `open`/`closed` status
constraint, lifecycle timestamps, a partial index for each tenant's open queue,
and a tenant/contact history index.

The provider opens a ticket with one SQL statement: a CTE compare-and-set
claims a non-archived contact only while its tenant is NULL or already matches,
and the `INSERT ... SELECT` runs only from that visible row. List predicates
are assembled before `ORDER BY`/`LIMIT`. Update uses a single
tenant-and-open-state predicate. Close locks the tenant ticket row first, then
updates only the locked open row or returns the locked already-closed row.
Concurrent and later retry callers therefore see the winning resolution
instead of reading through the stale statement snapshot.

The MCP functions require an explicit tenant or the configured deployment
default, validate bounded text and UUIDs, call only provider operations, and
return foreign rows as not found. `list_customer_service_tickets` defaults to
the operator-visible open queue.

## Intentional

- Status is deliberately only `open` or `closed`. The accepted issue requires a
  trackable lifecycle but does not authorize a broader workflow taxonomy;
  progress can be represented by assignee/priority until operator requirements
  justify another state.
- Priority and assignee are bounded caller-supplied strings rather than a new
  priority taxonomy or user-directory foreign key.
- The existing `atlas_eom_lead_pipeline_checks.yml` workflow is extended
  instead of renamed so its established check identity remains stable while it
  becomes the shared EOM CRM database-backed lane.
- Deployment is additive: migration `347` must precede use of the four new
  tools, while every existing CRM operation remains usable before and after
  it. Rollback is to remove/disable the new callers first, then drop only
  `customer_service_tickets` if the stored complaint data is intentionally
  discarded or exported.
- The merged lead-pipeline plan is archived in this branch as required
  teardown housekeeping; no lead-pipeline runtime behavior changes.
- The real-Postgres reachability test uses the provider module's existing
  `get_db_pool` seam to connect production `DatabaseCRMProvider` behavior to
  disposable Postgres. The maturity baseline therefore records the intentional
  `database.py` `INTERNAL_MOCK` increase from 33 to 34 (score 136 to 140);
  adding a production-only pool abstraction solely for this test would widen
  runtime code without improving the proof.

Diff-budget override: The migration, four MCP operations, canonical SQL,
CI enrollment, and real-Postgres proof are one indivisible reachable slice;
splitting them would ship public tools without storage or behavior evidence.

## Deferred

- Complaint intake UI/API, notifications, SLA/escalation policy, attachments,
  categories, and audit-event history need explicit operator/product
  requirements before they change the product shape.
- HTTP contacts API auth and tenant scoping remains #2170.
- Tenant-addressable email storage remains #2171.

Parked hardening: none.

## Verification

- Passed (108 tests):

      ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://... python -m pytest tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py tests/test_eom_lead_pipeline_integration.py tests/test_leads_intake.py tests/test_migrations_runner.py -q

- Passed (11 tests):

      python -m pytest tests/test_migrations_runner.py tests/test_pre_push_audit.py -q

- Passed:

      python -m py_compile atlas_brain/mcp/crm_server.py atlas_brain/services/crm_provider.py tests/test_eom_complaints_integration.py
      python scripts/audit_claude_md_claims.py
      python -m ruff check atlas_brain/mcp/crm_server.py atlas_brain/services/crm_provider.py tests/test_eom_complaints_integration.py tests/test_migrations_runner.py --ignore F841

  `F841` is existing debt in untouched CRM exception handlers.
- Passed:

      python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/*webhook*/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' --sensitive-glob '**/delete*/**' --sensitive-glob 'atlas_brain/security/**' --sensitive-glob 'atlas_brain/storage/**'

- Baseline-only failures outside this slice:
  `python -m pytest tests/test_mcp_servers.py tests/test_pre_push_audit.py -q`
  (79 passed, 6 failures in unchanged email, IMAP, Twilio, and calendar tests).
- Pending before push: bash scripts/local_pr_review.sh (through
  scripts/push_pr.sh).

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| `CLAUDE.md` | 6 |
| `README.md` | 2 |
| `atlas_brain/mcp/crm_server.py` | 275 |
| `atlas_brain/services/crm_provider.py` | 174 |
| `atlas_brain/storage/migrations/347_customer_service_tickets.sql` | 56 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Complaints-Tracking.md` | 196 |
| `plans/archive/PR-EOM-Lead-Pipeline.md` | 0 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/test_eom_complaints_integration.py` | 350 |
| `tests/test_migrations_runner.py` | 21 |
| `tests/test_pre_push_audit.py` | 6 |
| **Total** | **1098** |
