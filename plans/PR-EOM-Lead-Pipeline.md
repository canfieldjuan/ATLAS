# PR-EOM-Lead-Pipeline

## Why this slice exists

Issue #2167 splits Phase 4a from the completed EOM operational-CRM audit
(#2151). EOM leads are durable contacts, but the data model has no durable
pipeline state, owner, or follow-up timestamp. An operator therefore cannot
record lead progress or ask the CRM which EOM leads need attention.

This vertical slice exceeds the 400-LOC target because the production behavior
and its inseparable proof span the additive migration, canonical provider,
existing MCP tools, intake entrypoint, boundary tests, real-Postgres
reachability test, and the CI job that provisions Postgres for that test. No
adjacent product behavior is included.

### Problem-derived contract

- Root cause: lead lifecycle attributes are absent from the canonical contact
  persistence and CRM port. The intake path cannot initialize pipeline state,
  and the operator surface cannot persist or query actionable follow-ups.
- Correct fix must touch/change: extend the existing contact record with
  backward-compatible lead stage, owner, and next-follow-up fields; carry those
  fields through the canonical CRM provider; expose tenant-scoped MCP operations
  that update a visible lead and query visible leads needing follow-up; and have
  the real EOM intake entrypoint assign the initial stage only when it creates a
  new lead. Prove the migration, provider behavior, tenant boundary, idempotent
  re-intake behavior, and real entrypoint-to-persisted-state path.
- Must not change: the existing definition of a lead
  (`contacts.contact_type = 'lead'`), contact visibility/legacy-claim semantics
  from #2157/#2165, existing customer or non-lead records, acknowledgement
  delivery, website copy, complaint tracking, scheduling, pricing, API auth,
  or tenant-addressability of email history.

## Scope (this PR)

Ownership lane: eom-crm/lead-funnel
Slice phase: Vertical slice

1. Add durable, optional pipeline fields to canonical contact records and make
   them available through the CRM provider.
2. Add a narrow MCP write/read pair for lead pipeline work, preserving the
   default EOM-plus-unclaimed tenant visibility boundary.
3. Stamp newly created EOM intake leads with the initial `new` stage without
   resetting pipeline state when the same lead submits again.
4. Add migration, provider, MCP, and real intake-entrypoint behavioral proof.

### Review Contract

- Acceptance criteria:
  - Existing contacts remain valid after migration with nullable pipeline
    fields, and returned contact objects include those fields.
  - A caller can update stage, owner, and next follow-up for a visible lead
    without converting or mutating a non-lead contact.
  - A caller can query due leads by an inclusive follow-up cutoff; the tenant
    predicate is applied in SQL before ordering and limiting.
  - Default-scoped MCP calls see only `effingham_maids` plus claimable legacy
    rows, while explicit context remains supported and foreign rows fail closed.
  - A newly created EOM intake lead is persisted with stage `new`; a repeated
    intake does not reset an existing lead's pipeline fields.
  - Existing CRM callers and contact responses remain backward compatible.
- Reachability proof: call the real lead-intake FastAPI entrypoint and assert
  the canonical provider persists a new lead with `lead_stage = 'new'`; invoke
  the registered MCP tool functions and assert observable provider/SQL results.
- Affected surfaces: Postgres contact migration, CRM provider port and database
  adapter, CRM MCP server, EOM lead-intake API, focused tests, and the focused
  Postgres-backed CI workflow.
- Risk areas: online migration compatibility, tenant isolation, SQL filtering
  before `LIMIT`, mutation of non-leads, repeated-intake state reset, timestamp
  timezone handling, and public/MCP backward compatibility.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R7, R8, R10, R12, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/leads.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/346_contact_lead_pipeline.sql`
- `plans/PR-EOM-Lead-Pipeline.md`
- `tests/test_crm_read_scoping.py`
- `tests/test_eom_lead_pipeline_integration.py`
- `tests/test_leads_intake.py`
- `tests/test_migrations_runner.py`

## Mechanism

Use nullable columns on `contacts`, rather than a second lead table, because a
lead is already represented by the canonical contact row and the three fields
have the same lifecycle. The MCP layer performs transport-level text/timestamp
normalization while the provider enforces lead-only persistence and builds the
tenant-scoped due query. The existing `update_contact` and `list_contacts`
tools are extended instead of adding a parallel surface. Intake supplies the
initial stage on creation and leaves an existing row's pipeline state intact.

## Intentional

- Stage and owner remain caller-defined strings, matching the existing
  unconstrained CRM vocabulary; this slice does not invent a larger stage
  taxonomy or user-management model.
- Existing rows stay nullable instead of receiving a speculative backfill.
- Existing `create_contact`/`update_contact` callers that omit the three new
  fields keep their prior SQL and merge behavior; the CRM MCP compatibility
  tests cover the unchanged generic paths.
- The additive migration uses a normal partial index rather than a concurrent
  index because the live contacts table is small; rollback is the direct
  removal of the optional index/columns if deployment must be reversed.
- A dedicated leads table is rejected because it would duplicate contact
  identity and tenant ownership for three contact-lifecycle attributes.

## Deferred

- Complaints tracking remains #2168.
- Recurring schedules, cleaner assignment, and per-visit pricing remain #2169.
- Contacts API auth/scoping remains #2170.
- Tenant-addressable email history remains #2171.
- Terms acceptance and card-on-file remain epic #2156.

Parked hardening: none.

## Verification

- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@127.0.0.1:55432/atlas_migration_tests /home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_crm_read_scoping.py tests/test_eom_lead_pipeline_integration.py tests/test_leads_intake.py tests/test_migrations_runner.py -q`
  - passed: 107 (disposable Postgres 16 container)
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest tests/test_mcp_servers.py -q -k 'CRM or crm'`
  - passed: 22; deselected: 58
- /home/juan-canfield/Desktop/Atlas/.venv/bin/python scripts/audit_mcp_tool_names_match_docs.py
  - passed: CRM inventory remains 10/10; all MCP inventories match.
- /home/juan-canfield/Desktop/Atlas/.venv/bin/python scripts/audit_claude_md_claims.py
  - passed: CRM count remains 10/10; all MCP count claims match.
- bash scripts/check_ascii_python.sh
  - passed.
- `git diff --check`
  - passed.
- Workflow YAML parse and pinned Postgres-image assertion
  - passed.
- Plan sync check
  - passed.
- ATLAS_CURRENT_PR_BODY_FILE=/tmp/eom-lead-pipeline-pr-body.md ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-eom-lead-pipeline.local.md bash scripts/local_pr_review.sh
  - passed; the pre-existing plans-archive backlog was advisory only.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 74 |
| `atlas_brain/api/leads.py` | 1 |
| `atlas_brain/mcp/crm_server.py` | 166 |
| `atlas_brain/services/crm_provider.py` | 81 |
| `atlas_brain/storage/migrations/346_contact_lead_pipeline.sql` | 21 |
| `plans/PR-EOM-Lead-Pipeline.md` | 155 |
| `tests/test_crm_read_scoping.py` | 154 |
| `tests/test_eom_lead_pipeline_integration.py` | 114 |
| `tests/test_leads_intake.py` | 20 |
| `tests/test_migrations_runner.py` | 17 |
| **Total** | **803** |
