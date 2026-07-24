# PR-EOM-Recurring-Appointments

## Why this slice exists

Issue #2169, split from the #2151 EOM CRM audit, records that appointment rows
are isolated visits with no structured recurrence, assigned cleaner, or
per-visit price. The existing `customer_services.rate` is a standing agreement
rate, not an appointment price snapshot, and the CRM MCP can only read linked
appointments. This vertical slice makes those operating facts writable and
readable on the existing tenant-linked appointment through the canonical CRM
MCP.

### Problem-derived contract

- Root cause: `appointments` has no fields for recurrence, cleaner ownership,
  or the exact price of that visit, and the CRM surface has no tenant-safe
  mutation that can record them. Operators therefore cannot distinguish a
  recurring schedule from a one-off, see who owns the work, or preserve the
  price that applied to a visit.
- Correct fix must touch/change: add additive structured appointment columns
  for an every-N-days/weeks/months recurrence, a bounded cleaner label, and an
  exact non-negative two-decimal price; add one canonical provider mutation
  that scopes the linked appointment and owning contact atomically in SQL and
  claims a visible NULL-legacy contact on write; expose a validated partial
  update through the CRM MCP; return the fields from linked-appointment reads;
  and prove the exported MCP update/read round trip against disposable
  Postgres, including foreign/archived contact refusal, lost-claim refusal,
  invalid recurrence and money boundaries, and preservation of separate
  contact-linked billing accounts.
- Must not change: create future appointment occurrences, booking/conflict
  behavior, Google Calendar sync, `customer_services` rates or links, invoice
  generation, T&C/card-on-file work in #2156, contact identity or billing
  account rows, appointment customer/status fields, or any consumer-facing
  scheduling UI/copy.

## Scope (this PR)

Ownership lane: eom-crm/appointments
Slice phase: Vertical slice

1. Add nullable structured recurrence, assigned-cleaner, and per-visit-price
   fields to existing appointments without changing legacy rows.
2. Add a tenant-safe CRM MCP partial update for those operating fields and
   include them in the existing linked-appointment read.
3. Enroll a real-Postgres MCP update/read proof in the existing EOM CRM
   workflow.

### Review Contract

- Acceptance criteria:
  - [ ] Migration adds only nullable recurrence interval/unit,
        assigned-cleaner, and per-visit-price columns plus constraints that
        require recurrence interval/unit together, allow only day/week/month,
        and reject negative prices; the MCP rejects sub-cent prices before
        SQL, and existing rows remain valid.
  - [ ] The exported CRM MCP update accepts at least one operating-field
        change, validates UUIDs, paired recurrence, bounded cleaner text, and
        exact finite money before SQL, and returns the persisted appointment.
  - [ ] The provider claims a visible NULL-context contact and updates only an
        appointment linked to that contact and stamped with the effective
        tenant in one SQL statement; archived, missing, foreign, unlinked, or
        lost-claim rows read as not found.
  - [ ] Linked-appointment reads return the three new operating facts while
        preserving tenant filtering before `LIMIT` and refusing a concurrently
        foreign-owned contact.
  - [ ] Two contacts that represent separate billing accounts remain distinct;
        no contact, child row, `customer_services`, or invoice is merged or
        repointed.
  - [ ] Existing appointment booking, calendar, invoicing, and complaint/lead
        behavior is unchanged, and the MCP inventory reflects the new tool.
- Reachability proof: call the real exported CRM MCP operating-field update
  and linked-appointment read with the production `DatabaseCRMProvider`
  against disposable Postgres, then assert the tenant claim and exact
  recurrence/cleaner/price persisted on only the intended contact-linked row.
- Affected surfaces: CRM MCP public tools, direct-Postgres CRM provider,
  additive appointment migration, MCP inventory, EOM CRM GitHub Actions
  enrollment, and focused migration/integration tests.
- Risk areas: cross-tenant appointment writes/reads, legacy-contact claim
  races, contact/appointment mismatch, partial-update semantics, recurrence
  coherence, Decimal precision, and accidental billing-account consolidation.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12,
  R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `CLAUDE.md`
- `README.md`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/348_appointment_operating_fields.sql`
- `plans/INDEX.md`
- `plans/PR-EOM-Recurring-Appointments.md`
- `plans/archive/PR-EOM-Complaints-Tracking.md`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/test_eom_recurring_appointments_integration.py`
- `tests/test_migrations_runner.py`
- `tests/test_pre_push_audit.py`

## Mechanism

Migration `348` adds paired `recurrence_interval`/`recurrence_unit` columns,
`assigned_cleaner`, and `per_visit_price`. A recurrence is represented without
a fixed weekly/biweekly taxonomy: every positive integer `day`, `week`, or
`month`; both fields are NULL for a one-off or unknown schedule. Price is a
`NUMERIC(12,2)` snapshot on the appointment, independent of later changes to a
standing service rate.

The MCP partial update validates only caller-supplied fields. When recurrence
changes, interval and unit must be supplied together; cleaner and price may be
changed independently. The provider builds assignments only from that
validated internal field set. A single contact compare-and-set CTE claims a
visible non-archived legacy contact for the effective tenant, then updates only
the matching contact-linked, tenant-stamped appointment. The linked read joins
the contact in SQL so a concurrent foreign claim cannot leak an appointment.

## Intentional

- Recurrence is stored as every-N unit metadata; this slice records the
  operator's schedule but deliberately does not generate future appointment
  rows or change booking conflict behavior.
- Cleaner assignment is a bounded display label because Atlas has no accepted
  cleaner/user directory contract. Adding an employee identity system here
  would widen the product shape.
- Per-visit price is stored directly as an exact appointment snapshot. This
  avoids coupling the visit to nullable-tenant legacy `customer_services` and
  preserves the operator-approved separate-contact billing-account shape.
- The real-Postgres proof uses the provider module's existing `get_db_pool`
  seam to connect production `DatabaseCRMProvider` behavior to its disposable
  schema. The maturity baseline records that one intentional use
  (`INTERNAL_MOCK` 34 to 35, score 140 to 144); adding a production pool
  abstraction solely for this test would widen runtime code without improving
  the proof.
- Partial updates do not clear a field in this first slice: omitted values mean
  unchanged. Explicit clearing needs an operator-approved lifecycle for
  recurrence cancellation, cleaner unassignment, and price removal.
- Deployment is additive and migration-first: apply migration `348` before the
  MCP/provider release because linked reads select the new columns. Existing
  booking writers remain compatible because every new column is nullable.
  Rollback removes the new tool/read fields first, then drops the columns only
  if their stored operating data is intentionally discarded or exported.

Diff-budget override: The additive schema, tenant-safe MCP update/read path,
claim-race protection, CI enrollment, and real-Postgres billing-account and
concurrency proof are one reachable slice; splitting them would ship operating
fields without a safe operator entrypoint or behavioral evidence.

## Deferred

- Future occurrence generation, series-wide edits, calendar recurrence sync,
  cleaner identity/directory management, and field-clearing semantics need
  separate product requirements.
- A customer -> billing-account/site model is deferred; the two Mid Illinois
  Concrete contact rows and their separate services/invoices remain supported
  and untouched.
- T&C/card-on-file work remains in #2156.

Parked hardening: none.

## Verification

- Passed (110 tests):

      ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://... python -m pytest tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py tests/test_eom_lead_pipeline_integration.py tests/test_eom_recurring_appointments_integration.py tests/test_leads_intake.py tests/test_migrations_runner.py -q

- Passed (12 tests):

      python -m pytest tests/test_migrations_runner.py tests/test_pre_push_audit.py -q

- Passed:

      python -m py_compile atlas_brain/mcp/crm_server.py atlas_brain/services/crm_provider.py tests/test_eom_recurring_appointments_integration.py
      python -m ruff check atlas_brain/mcp/crm_server.py atlas_brain/services/crm_provider.py tests/test_eom_recurring_appointments_integration.py tests/test_migrations_runner.py --ignore F841
      python scripts/audit_claude_md_claims.py

  `F841` is existing debt in untouched CRM exception handlers.
- Passed:

      python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob '**/billing/**' --sensitive-glob '**/billing*' --sensitive-glob '**/paid*' --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/*webhook*/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' --sensitive-glob '**/delete*/**' --sensitive-glob 'atlas_brain/security/**' --sensitive-glob 'atlas_brain/storage/**'

- Passed:

      python scripts/sync_pr_plan.py plans/PR-EOM-Recurring-Appointments.md origin/main --check
      git diff --check

- Pending before push: repository local review through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 5 |
| `CLAUDE.md` | 5 |
| `README.md` | 2 |
| `atlas_brain/mcp/crm_server.py` | 136 |
| `atlas_brain/services/crm_provider.py` | 103 |
| `atlas_brain/storage/migrations/348_appointment_operating_fields.sql` | 77 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Recurring-Appointments.md` | 205 |
| `plans/archive/PR-EOM-Complaints-Tracking.md` | 0 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 4 |
| `tests/test_eom_recurring_appointments_integration.py` | 497 |
| `tests/test_migrations_runner.py` | 21 |
| `tests/test_pre_push_audit.py` | 6 |
| **Total** | **1064** |
