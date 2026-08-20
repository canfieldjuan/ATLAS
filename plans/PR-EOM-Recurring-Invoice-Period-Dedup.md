# PR-EOM-Recurring-Invoice-Period-Dedup

## Why this slice exists

Two independent writers auto-create commercial-customer invoices: the legacy
monthly cron (`atlas_brain/autonomous/tasks/monthly_invoice_generation.py`,
`source='monthly_auto'`) and the newer commercial-billing approval writer
(`atlas_brain/services/commercial_billing_approvals.py`,
`source='eom_commercial_billing'`). Each already bundles a contact's services
into one invoice per contact per period and dedups only against its OWN prior
invoices; neither is aware of the other. A customer can be auto-invoiced by
both for the same month, producing two real invoices.

**Root cause.** `invoices` carries no queryable covered-period column at all —
only `issue_date`/`due_date`, and both are creation/approval timestamps, not
the period a service covers. They drift independently between the two
writers: the legacy cron issues on the 1st of the month *after* the covered
period; the approval writer issues whenever an admin clicks approve, which the
code places no bound on (a backlogged April candidate can be approved in
June). So neither `issue_date` nor a date-range query on it can safely stand
in for "same billing period," and each writer's dedup is scoped only to its
own `source`.

This is documented in ATLAS #2363 (our comment, 2026-08-18) and was explicitly
deferred by three follow-up PRs — H-21 (#2439), H-22 (#2441), H-23 (#2445) —
each of which scoped itself away from it. H-21's completion comment states:
*"The legacy monthly-invoice/billing-run safety concern remains deferred to
its own evidence-backed financial slice."* This is that slice.

### Diff budget

Over the ~400-line soft cap (873 total per `git diff --numstat`, including
this plan document). The implementation itself is small —
~168 lines across the migration and three production-file edits, most of
which is the migration's root-cause documentation. The overage is entirely
test coverage: this repo's real-Postgres fixture-per-test convention
(isolated schema, migrations applied, explicit teardown) plus this being
guard-shaped, billing-adjacent code, which requires a negative control for
every acceptance claim (REVIEWER_RULES.md boundary-probe rule; AGENTS.md
§3i). Trimming coverage to hit the cap on a change whose entire job is
"prevent a real duplicate invoice" is the wrong trade — this mirrors the
precedent in `plans/PR-EOM-Billing-Recipients.md`, which went over the same
cap for the same reason (tests + plan doc, not code) and declined to split
rather than review the same disclosure decision twice in halves.

### Problem-derived contract

- Root cause: no persisted, queryable covered-period column on `invoices`;
  each recurring writer's dedup is scoped only to its own `source`.
- Correct fix must touch/change: persist the already-computed in-memory
  period value (`period_label` in the legacy task, `_InvoiceDraft.billing_period`
  in the approval writer) as a real `invoices.billing_period VARCHAR(7)`
  column; add a partial unique index on `(contact_id, billing_period)` scoped
  by `source IN ('monthly_auto','eom_commercial_billing')` and
  `status <> 'void'`; add an app-level pre-check in both writers for a clean
  skip/reject path ahead of that constraint.
- Must not change: the `mcp_tool` ad-hoc invoice path (no `billing_period`,
  outside the source allowlist); the `invoicing_mcp` source (confirmed during
  research to be a *payment*-record source, unrelated to `invoices.source`);
  historical invoice rows (left `NULL`, no backfill); void-status invoices
  (excluded, so void-and-reissue keeps working); each writer's existing
  own-source dedup (`get_by_source_ref`, the `(source, source_ref)` partial
  index from migration 372, idempotency-key replay); the architectural
  boundary asserted by
  `test_approval_service_does_not_import_delivery_or_legacy_monthly_writers`
  (neither writer imports the other — this design doesn't either, both only
  reach the shared `invoices` table).

## Scope (this PR)

Ownership lane: eom/recurring-invoice-period-dedup
Slice phase: Production hardening
Max files: 8

1. Migration 385: `invoices.billing_period` column + format CHECK + the
   cross-source partial unique index.
2. Persist `billing_period` in both writers' existing `INSERT`s, reusing each
   writer's already-computed period value — zero new parameters on either
   `InvoiceRepository.create()` or `_InvoiceDraft`.
3. `InvoiceRepository.get_by_contact_and_period` and a matching
   transaction-scoped lookup in `commercial_billing_approvals.py`; call both
   from their respective writers as a pre-check ahead of the new constraint.
4. Tests proving: the index rejects a raw cross-source duplicate insert
   independent of any app code; an `mcp_tool` invoice doesn't block a
   recurring one; a voided invoice doesn't block reissuance; the approval
   writer rejects when the legacy writer already invoiced the period (and
   does not when the period differs); the legacy writer's `run()` skips a
   contact the new pipeline already invoiced without calling `create()`, and
   still creates normally for an un-invoiced contact in the same pass.

### Review Contract

- Acceptance criteria:
  - A raw cross-source duplicate insert for the same `(contact_id,
    billing_period)` is rejected by the database itself — settled by
    `test_invoice_repository.py::test_real_postgres_billing_period_dedup_scoping_and_void_exclusion`.
    Negative controls in the same test: a different contact succeeds; a
    different period succeeds; an `mcp_tool`-source insert for the same
    contact+period succeeds.
  - A voided recurring invoice does not block reissuance for the same
    contact+period — same test. Negative control: skipping the void step
    (two live recurring rows for one contact+period) is rejected by the same
    index.
  - The approval writer refuses to create a second recurring invoice when the
    legacy writer already invoiced the contact+period — settled by
    `test_commercial_billing_approvals.py::test_real_postgres_approval_rejects_when_legacy_monthly_writer_already_invoiced_the_period`.
    Negative control in the same test: the same contact with a *different*
    billing period is not a conflict and creates invoice #2 normally —
    proves the check discriminates on period, not just contact.
  - The legacy writer's `run()` skips a contact the new pipeline already
    invoiced, without ever calling `create()` for that contact, and still
    creates normally for a second, un-invoiced contact in the same pass —
    settled by
    `test_monthly_invoice_generation_cross_pipeline_dedup.py::test_legacy_writer_skips_contact_the_new_pipeline_already_invoiced`.
  - Migration content is additive and correctly scoped (no `DROP`, references
    both recurring sources and the void exclusion, indexes
    `(contact_id, billing_period)` alone — not `source` — since source
    belongs only in the predicate) — settled by
    `test_invoice_repository.py::test_invoices_billing_period_dedup_migration_is_additive_and_scoped`.
- Reachability proof: exercised through the real entry points — the legacy
  writer's `run()` (autonomous task) and the approval writer's `approve()`
  (`atlas_brain/mcp` / admin-facing approval flow) — not through the repo
  methods in isolation.
- Affected surfaces: `invoices` table schema; both recurring invoice-creation
  call sites; no route signatures, no config, no new environment variables.
- Risk areas: a naive dedup signal wrongly blocking a legitimate ad-hoc
  invoice (mitigated by source-scoping, proven negative control); the
  cross-system check silently failing open and never firing (mitigated by
  the DB-level unique index as the actual backstop, independent of the
  app-level pre-check); a race between the two writers (mitigated by the
  partial unique index rather than a bare check-then-act).
- Reviewer rules triggered: R2 (test evidence, every acceptance criterion has
  a negative control run and shown to fail, not just the happy path), R3
  (invoicing/billing code), **R4 (data and migration safety — migration 385
  adds a nullable column with no backfill requirement, guards every DDL
  statement with `IF NOT EXISTS`/existence checks, and documents its rollback
  order in-file)**, R6 (secondary-write error handling — the approval
  writer's existing broad `UniqueViolationError` handler is the backstop and
  needed no new code), **R8 (concurrency/idempotency — the partial unique
  index is the actual fix, not the app-level pre-check)**, R14 (verified
  against `origin/main` @ `cd899b03c`, not cached line numbers — H-21/H-22/H-23
  had shifted them since this was first scoped).

### Boundary-change enumeration

**Seam 1 — cross-pipeline recurring-invoice admission**
(`InvoiceRepository.get_by_contact_and_period`,
`CommercialBillingApprovalService._find_recurring_period_conflict`,
`idx_invoices_recurring_contact_period_source`)

- Replaced behavior: none existed; this is the new decision. Previously each
  writer's only dedup check was `get_by_source_ref`, scoped to its own
  `source_ref` format (a private, incompatible format per writer).
- Guard-relevant fields: `invoices.contact_id`, `invoices.billing_period`
  (new), `invoices.source`, `invoices.status`.
- Caller x input: {legacy writer x contact already invoiced by the approval
  writer for this period, legacy writer x contact not yet invoiced, approval
  writer x contact already invoiced by the legacy cron for this period,
  approval writer x same contact different period, any writer x an
  `mcp_tool` invoice present for the same contact+month, any writer x a
  voided prior recurring invoice for the same contact+period}.
- Disposition: every class above is asserted across
  `test_real_postgres_billing_period_dedup_scoping_and_void_exclusion`,
  `test_real_postgres_approval_rejects_when_legacy_monthly_writer_already_invoiced_the_period`,
  and `test_legacy_writer_skips_contact_the_new_pipeline_already_invoiced`.

**Seam 2 — persistence of the covered period**
(`InvoiceRepository.create`, `CommercialBillingApprovalService._insert_invoice`)

- Replaced behavior: both writers already computed a `"YYYY-MM"` period value
  and used it only to format the `invoice_number` string (`to_char(...,
  'YYYY-Mon')`); it was never persisted as a queryable value.
- Guard-relevant fields: `invoices.billing_period` (new), format-checked by
  `invoices_billing_period_check` (same regex as the existing
  `commercial_billing_runs.billing_period` check).
- Caller x input: {a caller-supplied `billing_period` date (the two recurring
  writers), no `billing_period` supplied (the MCP `create_invoice` tool)}.
  Both reuse an existing bind parameter — no new parameter added to either
  `create()` or `_InvoiceDraft`.
- Disposition: covered implicitly by every test above that reads
  `billing_period` back via `get_by_contact_and_period`; the `NULL` case
  (MCP tool) is proven by the mcp_tool-does-not-block assertions.

### Files touched

- `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql`
- `atlas_brain/storage/repositories/invoice.py`
- `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `tests/test_invoice_repository.py`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py`
- `plans/PR-EOM-Recurring-Invoice-Period-Dedup.md`

## Mechanism

Both writers already compute the covered billing period in memory before
creating an invoice; neither persisted it. Migration 385 adds
`invoices.billing_period VARCHAR(7)` and a partial unique index on
`(contact_id, billing_period)` restricted to non-void rows whose `source` is
one of the two recurring types. `source` is deliberately **not** part of the
index's column list — only its `WHERE` predicate — so a `monthly_auto` row
and an `eom_commercial_billing` row for the same contact+period collide on
the *same* index key. (Verified directly: an earlier draft of this index
included `source` in the column list, which gives the two sources
independent keys and lets both insert cleanly — the exact bug this migration
exists to close. Caught during planning, before any code shipped, by testing
the actual duplicate scenario against the index definition.)

`InvoiceRepository.create()` and the approval writer's raw `_insert_invoice`
both already receive a `date` representing the first day of the covered
period; each now also writes `to_char(<that date>, 'YYYY-MM')` into the new
column, reusing the existing bind parameter. Each writer adds an app-level
pre-check so a detected conflict is a clean skip (legacy: increments the
existing `invoices_skipped_dedup` counter, logs which source got there
first) or a clean `CommercialBillingApprovalConflictError` (approval writer:
a single admin action, so the equivalent of "skip" is refusing it with a
clear message) rather than reaching the `INSERT` in the common case. The
partial unique index is the actual guarantee against a race between the two
writers; the approval writer's existing broad
`except (asyncpg.UniqueViolationError, asyncpg.ForeignKeyViolationError)`
handler (present before this change, unrelated in origin) already converts a
raw constraint violation into the same clean error with no new code, and the
legacy task's existing fail-open-on-check-error behavior is preserved — a
check failure logs and falls through to `create()`, where the new
constraint is now a real backstop where none existed before.

## Intentional

- No new parameter on `InvoiceRepository.create()` or `_InvoiceDraft`: both
  writers already pass/hold the exact `date` needed; reusing it keeps each
  `INSERT` change to a SQL-only edit.
- The legacy task's cross-pipeline skip reuses the existing
  `invoices_skipped_dedup` counter rather than adding a dedicated one; the
  two cases stay distinguishable via the log message text (`"already exists"`
  vs `"already invoiced ... by source=..."`), not a separate metric.
- The partial unique index intentionally omits `source` from its column list
  — see Mechanism. This is the one design detail in this PR that is easy to
  get backwards and silently ship a no-op guard.
- No backfill of `billing_period` on historical rows — the fix only needs to
  work forward from deploy; a backfill is a separate, riskier historical-data
  slice.
- The approval writer's pre-check queries via the same transaction
  connection (`conn`), not a separate pool, to stay inside the same
  transaction/advisory-lock scope as the rest of `approve()`.
- The legacy-writer proof (`test_monthly_invoice_generation_cross_pipeline_dedup.py`)
  is a new, small, fully monkeypatched module rather than an extension of
  `tests/test_monthly_invoice_generation.py` — that file's `run()`-level
  tests either require a real, pre-seeded dev database (not appropriate for
  a disposable schema) or only exercise the early settings-gate return
  (never reaching this code). It is also not an extension of
  `tests/test_legacy_monthly_autoinvoice_writer_harness.py` — see Deferred.

## Deferred

- Backfilling `billing_period` on historical invoice rows.
- The pre-existing void-filter gap in `InvoiceRepository.get_by_source_ref`
  and `.search()` (neither excludes `status='void'` today) — real, but
  unrelated to this slice's cross-pipeline scope.
- A heavier, fully end-to-end proof of the legacy-skip case via
  `tests/test_legacy_monthly_autoinvoice_writer_harness.py` (the armed,
  real-Postgres, loopback-only harness) — the monkeypatched proof in this PR
  covers the same claim (skip fires, `create()` not called, a second
  un-invoiced contact still creates) without the harness's arming
  requirements; can follow up as an isolated test-only addition.
- Further investigation of `source="invoicing_mcp"` — resolved during this
  slice's research as a *payment*-record source
  (`record_customer_payment`, `atlas_brain/mcp/invoicing_server.py:554`),
  not an invoice-creation source; no further action needed.

Parking predicate: hardening beyond the two recurring sources' cross-
awareness (broader financial-integrity auditing, backfills, the unrelated
void-filter gap) is parked unless necessary to prove this specific
constraint. Nothing here qualifies.

Parked hardening: none.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://atlas:atlas@127.0.0.1:<port>/atlas
  pytest tests/test_invoice_repository.py tests/test_commercial_billing_approvals.py
  tests/test_monthly_invoice_generation_cross_pipeline_dedup.py -q`
  against a throwaway `postgres:16` container (not the shared dev DB): **115
  passed**, 0 failed. This includes every pre-existing test in
  `test_commercial_billing_approvals.py` (111 of the 115) — confirming the
  new pre-check does not regress the existing approval contract tests, which
  required adding migration 385 to that file's shared `_approval_database`
  fixture (its `invoices.billing_period` column did not otherwise exist for
  those schemas, and 8 of them failed with
  `CommercialBillingApprovalUnavailableError` wrapping a real
  `UndefinedColumnError` until this was fixed — caught by running the full
  file, not just the new test).
- The three pre-existing settings-gate tests in `tests/test_monthly_invoice_generation.py`
  (`test_billing_month_override`, `test_billing_month_invalid_format`,
  `test_contact_ids_filter`) re-run clean: **3 passed**.
- `ruff check` on all seven changed files: **all checks passed**, zero
  findings.
- **Negative controls, all run and restored, not merely asserted:**
  - Same shape as an existing `monthly_auto` recurring row but re-inserted
    for the same contact+period raises `asyncpg.UniqueViolationError`
    (proves the index is real, not just the app-level check).
  - Skipping the void step before re-issuing (i.e., two live recurring rows
    for the same contact+period) raises the same violation.
  - A different contact, a different period, and an `mcp_tool`-source insert
    for the same contact+period all succeed where a same-source duplicate
    would not — proving the constraint's scope is exactly the two recurring
    sources, not "any invoice."
  - The approval writer's rejection is period-specific, not merely
    contact-specific: the identical contact with a different billing period
    is approved normally and produces invoice #2.
  - The legacy writer's pre-check is called for every bundle in a run, not
    only the one that hits — asserted directly against both call arguments,
    not inferred from the skip count alone.
- The unique-index design itself was verified against the exact duplicate
  scenario before being implemented: a draft with `source` inside the
  index's column list was checked against a same-contact/period,
  different-source pair and found to admit both rows, which is how the
  final index (source in the predicate only) was arrived at rather than
  shipped incorrectly.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql` | 74 |
| `atlas_brain/storage/repositories/invoice.py` | 45 |
| `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` | 15 |
| `atlas_brain/services/commercial_billing_approvals.py` | 34 |
| `tests/test_invoice_repository.py` | 132 |
| `tests/test_commercial_billing_approvals.py` | 97 |
| `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py` | 147 |
| `plans/PR-EOM-Recurring-Invoice-Period-Dedup.md` | 327 |
| **Total** | **873** |
