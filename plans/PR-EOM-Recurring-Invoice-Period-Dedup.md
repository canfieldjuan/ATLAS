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

**Amendment after initial review.** `chatgpt-codex-connector` left three P1/
BLOCKER threads on the opened PR (#2448) across two review rounds, all
confirmed real by independent investigation (including empirical proof
against real Postgres) before any fix was written:

1. **Historical rows were left `billing_period = NULL`** (the original design
   deferred backfilling them). `NULL = 'x'` is `NULL`, not true, in SQL, so a
   pre-migration legacy invoice for an old period is invisible to both the
   app-level pre-check and the partial unique index — and nothing bounds how
   far in the past an admin can approve a commercial-billing candidate, so
   this transition window is a live risk, not a theoretical one. Fixed by
   backfilling `billing_period` from data each writer already persisted,
   with non-destructive collision quarantine for any historical row that
   would itself collide. See "Amendment: historical backfill" below.
2. **No startup fence existed for migration 385.** The existing
   `atlas_brain/main.py` fence protects only migration 382 under
   `receivables_api_enabled`; it never checked migration 385, and the legacy
   cron reaches `billing_period` through an entirely independent flag
   (`auto_invoice_enabled`) the old fence never looked at. Adversarial
   re-review found the true scope larger than the original fix: two more
   independent OS processes — `atlas_brain/mcp/invoicing_server.py` and
   `atlas_brain/mcp/invoicing_draft_writer_server.py` — also reach
   `InvoiceRepository.create()` and were unfenced (one already had a general
   readiness-check hook that just didn't know about the new column/index yet;
   the other had no schema check of any kind). See "Amendment: startup
   fence" below.
3. **A quarantined collision group had no protection at all, going
   forward.** Fix #1's collision quarantine leaves every row in an ambiguous
   group `billing_period = NULL` permanently, which is inert against both
   the partial index and the app-level pre-checks (`WHERE billing_period =
   $2` never matches `NULL`) — so a THIRD invoice for that same
   contact+period would go unblocked, despite the period being known and
   recorded in the quarantine metadata. Verified empirically: inserting a
   collision pair, then a third row with `billing_period` set explicitly to
   the shared period, succeeded with no constraint violation. Fixed by a
   small `invoices_billing_period_reservations` table — Codex's own
   suggested shape, "one database-visible reservation per quarantined
   period" — populated by the same backfill pass, checked by both writers'
   pre-checks. See "Amendment: quarantine reservation" below.

**Amendment after live CI + a third review round.** After the round-2 fixes
above were pushed, CI surfaced two real gaps (see "CI-caught fixes" below) and
`chatgpt-codex-connector` posted four more findings against the pushed
commit, three confirmed real and fixed, one investigated and waived with
evidence rather than fixed:

4. **The `eom_commercial_billing` backfill regex required exactly 4 digits
   for the invoice-number sequence** (`-\d{4}$`), but the writer that
   produces those numbers (`commercial_billing_approvals.py`'s
   `lpad(nextval(...)::text, 4, '0')`) only guarantees a *minimum* of 4 —
   once the sequence passes 9999, a real invoice number like
   `INV-2026-Oct-10000` silently failed the backfill regex and was excluded
   from both collision detection and the partial unique index, with no
   error, warning, or quarantine marker. Fixed by widening both occurrences
   to `\d{4,}` (the match and the extraction substring, in both backfill
   passes). See "Amendment: round 3 hardening" below.
5. **`atlas_brain/main_eom.py`'s lifespan had no independent readiness fence
   at all**, in either of the two other processes' patterns (`main.py`'s
   ledger-based `_migration_is_recorded`, or the MCP servers'
   `ReceivablesService.is_ready()` hook) — it ran migrations only when
   `eom_profile_settings.run_migrations` is true, which **defaults to
   `False`**, and otherwise proceeded straight to serving requests
   regardless of schema state. This file does mount `receivables_router`,
   so the commercial-billing approval endpoint is genuinely reachable
   through this process once its still-draft Render blueprint
   (`render.eom.yaml`) is connected. Fixed by adding the same
   `ReceivablesService(pool).is_ready()` fence the MCP servers already use,
   gated on `receivables_api_enabled`.
6. **Rolling-deployment code/schema skew was raised as a concern** for both
   writer processes. Investigated fresh against this machine's actual
   running units (not assumed from memory) and disposed — see "Amendment:
   round 3 hardening" for the evidence and the waiver rationale.
7. **`main.py`'s auto-invoice fence over-triggered**: it required migration
   385 whenever `settings.invoicing.auto_invoice_enabled` was true, without
   checking the master `settings.invoicing.enabled` gate the legacy task
   itself checks first (`monthly_invoice_generation.py` returns `"Invoicing
   disabled"` before ever reaching `auto_invoice_enabled` or
   `billing_period` when `enabled=False`) — so a deployment with invoicing
   entirely disabled but a stale `auto_invoice_enabled=True` left over in
   config would be false-positive-blocked from starting at all, for a code
   path that can never actually run. Fixed by scoping the fence's
   `auto_invoice_enabled` input to `settings.invoicing.enabled and
   settings.invoicing.auto_invoice_enabled`, matching the condition the
   legacy task itself already gates on.
8. **The legacy writer's cross-pipeline dedup check failed open on error.**
   The new `get_by_contact_and_period` pre-check (Amendment: startup
   fence's sibling change to `monthly_invoice_generation.py`) shared a
   catch-all `try/except` with the pre-existing `get_by_source_ref` check —
   any exception, including a transient DB read error, was logged and
   fell through to `create()`. For a quarantined historical collision
   (protected only by `invoices_billing_period_reservations`, not the
   partial unique index — both leave `billing_period = NULL`), a check
   failure at exactly the wrong moment would admit the unprotected third
   duplicate this reservation table exists to prevent. Fixed by splitting
   the two checks into separate `try/except` blocks: the pre-existing
   `get_by_source_ref` check keeps its original fail-open behavior (safe,
   real DB constraint backstop either way), while the new
   `get_by_contact_and_period` check now fails closed — skips this contact
   this run rather than proceeding to `create()`. See "Amendment: round 3
   hardening" below.

Well over the ~400-line soft cap (per `git diff --numstat origin/main...HEAD`,
including this plan document — see Estimated diff size below for the
per-file breakdown across all rounds). The core fix (migration + dedup
pre-checks in the two writers) is ~168 lines; the post-review amendments
below are what pushed this over, and all eight are real, reviewer- or
CI-found correctness/safety gaps in the original design, not scope creep —
declining to fold them into this PR would mean shipping a dedup migration
that (a) still admits the exact cross-pipeline duplicate this slice exists
to prevent for any pre-existing period, (b) has no protection if it fails
to apply on three of the four processes that depend on it, (c) silently
excludes a real class of invoice numbers from that protection once a
sequence counter passes 9999, and (d) can be defeated by a single transient
DB read error on a quarantined period. Splitting them into follow-up PRs
was considered and rejected: each is a direct, evidence-backed fix to a
BLOCKER finding on the *same* migration this PR introduces, reviewable only
against the migration itself, and a fix-after-merge would mean the merged
state was briefly, correctly flagged as a P1 blocker.

### Problem-derived contract

- Root cause: no persisted, queryable covered-period column on `invoices`;
  each recurring writer's dedup is scoped only to its own `source`; no
  startup verification that the migration providing that column actually
  applied before code that depends on it starts serving.
- Correct fix must touch/change: persist the already-computed in-memory
  period value (`period_label` in the legacy task, `_InvoiceDraft.billing_period`
  in the approval writer) as a real `invoices.billing_period VARCHAR(7)`
  column; add a partial unique index on `(contact_id, billing_period)` scoped
  by `source IN ('monthly_auto','eom_commercial_billing')` and
  `status <> 'void'`; add an app-level pre-check in both writers for a clean
  skip/reject path ahead of that constraint; backfill historical rows where
  the period is mechanically derivable, quarantining (not guessing at)
  anything that would collide; fence recurring invoice writers against the
  writer-only dedup schema being unavailable, without making ad-hoc invoice,
  draft invoice, payment/receivables, or EOM funnel startup depend on that
  recurring-only schema.
- Must not change: the `mcp_tool` ad-hoc invoice path (no `billing_period`,
  outside the source allowlist); the `invoicing_mcp` source (confirmed during
  research to be a *payment*-record source, unrelated to `invoices.source`);
  void-status invoices (excluded, so void-and-reissue keeps working); each
  writer's existing own-source dedup (`get_by_source_ref`, the `(source,
  source_ref)` partial index from migration 372, idempotency-key replay); the
  architectural boundary asserted by
  `test_approval_service_does_not_import_delivery_or_legacy_monthly_writers`
  (neither writer imports the other — this design doesn't either, both only
  reach the shared `invoices` table); the existing review-recovery fence for
  migration 382 (preserved, not replaced); base `ReceivablesService.is_ready()`
  readiness for check/ACH/Square payment tracking; EOM slim receivables
  startup; and the draft-writer MCP's ability to create an ad-hoc invoice on
  the minimal invoice schema.

## Scope (this PR)

Ownership lane: eom/recurring-invoice-period-dedup
Slice phase: Production hardening
Max files: 19

1. Migration 385: `invoices.billing_period` column + format CHECK + the
   cross-source partial unique index + historical backfill + collision
   quarantine. This migration intentionally does **not** opt into
   `-- atlas: atomic-bookkeeping`; it uses `CREATE/DROP INDEX CONCURRENTLY`
   and `NOT VALID` constraints so production does not combine a backfill and
   long index build inside one ACCESS-EXCLUSIVE transaction.
2. Persist `billing_period` in both writers' existing `INSERT`s, reusing each
   writer's already-computed period value — zero new parameters on either
   `InvoiceRepository.create()` or `_InvoiceDraft`.
3. `InvoiceRepository.get_by_contact_and_period` and a matching
   transaction-scoped lookup in `commercial_billing_approvals.py`; call both
   from their respective writers as a pre-check ahead of the new constraint.
4. `atlas_brain/main.py`: keep the existing migration-382 receivables fence,
   then separately require the recurring-dedup schema only when a recurring
   writer is enabled (`receivables_api_enabled` for approval writes or the
   master-enabled legacy auto-invoice task).
5. Current base receivables readiness is already recurring-free on refreshed
   `origin/main`; this PR keeps that boundary and removes migration 385 from
   `atlas_brain/main_eom.py`'s curated EOM receivables migration set so EOM
   slim startup is not over-gated by a writer it does not run.
6. `atlas_brain/mcp/invoicing_draft_writer_server.py`: run only the
   draft-runtime schema migrations the exposed tools actually need and verify
   that schema, without requiring recurring-dedup migration 385.
7. `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`: make recurring
   dedup readiness failures and per-contact dedup lookup failures fail closed
   and visible in the notification summary.
8. Tests proving: the index rejects a raw cross-source duplicate insert
   independent of any app code; an `mcp_tool` invoice doesn't block a
   recurring one; a voided invoice doesn't block reissuance; the approval
   writer rejects when the legacy writer already invoiced the period (and
   does not when the period differs); the legacy writer's `run()` skips a
   contact the new pipeline already invoiced without calling `create()`, and
   still creates normally for an un-invoiced contact in the same pass;
   historical backfill derives the right period per source and quarantines
   real collisions without falsely flagging unrelated NULL-contact_id rows;
   the recurring-writer fence blocks only the recurring writer surface when
   the dedup schema is unavailable; the standalone monthly task-level fence
   stops before invoice lookup/create when dedup readiness is false or raises;
   the recurring readiness helper verifies actual index/constraint definitions,
   not just object names; EOM/base receivables readiness no longer requires
   migration 385; the draft-writer MCP server migrates only its draft-runtime
   invoice/payment prerequisites and verifies that schema.

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
  - The standalone monthly task-level recurring-dedup readiness fence fails
    closed before service loading, cross-pipeline invoice lookup, or invoice
    creation when readiness returns `False` or raises — settled by
    `test_monthly_invoice_generation_cross_pipeline_dedup.py::test_legacy_writer_fails_closed_when_task_level_dedup_schema_not_ready`
    and
    `::test_legacy_writer_fails_closed_when_task_level_dedup_schema_check_raises`.
  - The recurring-dedup readiness helper verifies actual schema definitions,
    not just same-named objects: a same-named valid unique index with
    `(contact_id, billing_period, source)` is rejected, and a same-named
    required-period CHECK missing the legacy-null exemption is rejected —
    settled by
    `test_invoice_repository.py::test_real_postgres_recurring_dedup_readiness_rejects_drifted_definitions`
    when real Postgres is configured.
  - Migration content is additive and correctly scoped for production rollout:
    it references both recurring sources and the void exclusion, indexes
    `(contact_id, billing_period)` alone — not `source` — since source
    belongs only in the predicate, omits the atomic-bookkeeping marker,
    uses `CREATE INDEX CONCURRENTLY`/`DROP INDEX CONCURRENTLY`, adds `NOT VALID`
    checks, validates producer years, and includes the collision-quarantine
    marker) — settled by
    `test_invoice_repository.py::test_invoices_billing_period_dedup_migration_is_additive_and_scoped`.
  - Historical rows backfill correctly per source, quarantine real
    collisions, leave void/mcp_tool/unparseable rows untouched, and do not
    falsely quarantine unrelated NULL-`contact_id` rows that share a derived
    period — settled by
    `test_invoice_repository.py::test_real_postgres_billing_period_backfill_and_collision_handling`,
    which also proves the migration is idempotent on replay and that the
    backfilled row closes the transition-window gap end to end (findable by
    the pre-check; a second insert for the same contact+period is rejected).
  - The startup fence preserves migration-382 receivables recovery and blocks
    enabled recurring writers when the recurring dedup schema is unavailable,
    using a schema readiness check rather than a broad migration-ledger proxy —
    settled by four tests in
    `test_commercial_billing_runs.py`
    (`test_full_atlas_migration_check_allows_recovered_or_disabled_receivables`,
    `test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_readiness`,
    `test_full_atlas_migration_check_allows_auto_invoice_when_dedup_schema_ready`,
    `test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness`),
    the last exercised through the real `main.lifespan(...)` end to end, not
    only the unit-level check function.
  - The draft-writer MCP server's `_lifespan` now runs only draft-runtime
    prerequisite migrations and verifies that schema instead of requiring
    recurring-dedup readiness; it blocks on an incomplete runtime schema —
    settled by
    `test_invoicing_draft_writer_mcp.py::test_draft_writer_lifespan_now_migrates_and_verifies_readiness`
    and `::test_draft_writer_lifespan_blocks_on_incomplete_schema`.
- Reachability proof: exercised through the real entry points — the legacy
  writer's `run()` (autonomous task), the approval writer's `approve()`
  (`atlas_brain/mcp` / admin-facing approval flow), the real
  `atlas_brain.main.lifespan(...)` startup handler, and the draft-writer MCP
  server's real `_lifespan` — not through the repo methods or check functions
  in isolation, except where a function-level unit test is the more precise
  proof for a specific branch (e.g. the four-way fence matrix).
- Affected surfaces: `invoices` table schema; both recurring invoice-creation
  call sites; the shared migration-recovery startup fence in
  `atlas_brain/main.py`; the recurring dedup readiness helper in
  `atlas_brain/storage/repositories/invoice.py`; the base receivables readiness
  contract in `atlas_brain/services/receivables.py`; the draft-writer MCP server's
  lifespan; the EOM profile's curated readiness-migration set in
  `atlas_brain/main_eom.py`. No route signatures, no config, no new
  environment variables.
- Risk areas: a naive dedup signal wrongly blocking a legitimate ad-hoc
  invoice (mitigated by source-scoping, proven negative control); the
  cross-system check silently failing open and never firing (mitigated by
  the DB-level unique index as the actual backstop, independent of the
  app-level pre-check); a race between the two writers (mitigated by the
  partial unique index rather than a bare check-then-act); a pre-deploy
  invoice for an old period going unprotected (mitigated by the backfill,
  with quarantine rather than guessing for genuine ambiguity); the migration
  causing production lock pressure (mitigated by concurrent index operations
  and NOT VALID constraints); code depending on the new column/index starting
  before the migration applied (mitigated by a recurring-writer schema
  readiness fence instead of a broad receivables/startup gate).
- Reviewer rules triggered: **R1 (requirements match —
  `invoicing_draft_writer_server.py`'s `_lifespan` now runs migrations and
  verifies readiness where it previously did neither; this is an intentional
  fix to the startup-fence gap, not scope creep, and is covered by two new
  tests)**, R2 (test evidence, every acceptance criterion has
  a negative control run and shown to fail, not just the happy path), R3
  (invoicing/billing code), **R4 (data and migration safety — migration 385
  adds a nullable column and a historical backfill; atomicity was verified,
  not assumed, by forcing a real mid-migration failure and confirming zero
  partial state persisted, then confirming the marker closes the narrower
  SQL-succeeds/ledger-write-fails crash window that a bare multi-statement
  `execute()` does not)**, **R5 (backward compatibility — the draft-writer
  MCP server's startup contract changes from "always starts" to "can refuse
  to start on an incomplete schema"; this is the intended, narrow behavior
  change the fence exists to add, not an accidental break, and both
  directions — refuses when incomplete, still starts when ready — are
  tested)**, R6 (secondary-write error handling — the approval
  writer's existing broad `UniqueViolationError` handler is the backstop and
  needed no new code), **R8 (concurrency/idempotency — the partial unique
  index is the actual fix, not the app-level pre-check)**, **R12 (deployment
  safety — a startup gate that can now refuse to boot three independent
  processes rather than only one; proven with both a positive fence test and
  a negative "does not false-positive-block a healthy deploy" control for
  each)**, R14 (verified against `origin/main` at the exact commit Codex
  reviewed, `bd9a1bfdec8066774d94acc879be99bfaefd7f84`, not cached line
  numbers or assumed file contents).

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

**Seam 3 — historical backfill and collision quarantine** (migration 385's
two `WITH candidates ... UPDATE` passes)

- Replaced behavior: none; historical rows were left permanently `NULL` in
  the original design, which Codex correctly identified as leaving the
  transition window (pre-migration invoices vs. post-migration candidate
  approvals for the same old period) unprotected.
- Guard-relevant fields: `source_ref` (monthly_auto derivation),
  `invoice_number` (eom_commercial_billing derivation), `contact_id`
  (collision grouping key — nullable, and Postgres's per-row-distinct NULL
  semantics for unique indexes do **not** match SQL `GROUP BY`'s
  NULLs-group-together semantics, which the collision CTE must and does
  correct for).
- Caller x input: {unambiguous monthly_auto row, unambiguous
  eom_commercial_billing row, a genuine historical collision pair across
  both sources, a void row, an unparseable legacy-format `invoice_number`, an
  `mcp_tool` row, a syntactically-shaped but invalid month abbreviation
  (`INV-2026-Xyz-0008`, which an unconstrained regex would crash Postgres's
  `to_date` on), a NULL-`contact_id` pair sharing a derived period}.
- Disposition: every class above is asserted in
  `test_real_postgres_billing_period_backfill_and_collision_handling`,
  including a real-Postgres proof that a forced mid-migration failure (after
  the backfill already ran, before the index) rolls back the column, the
  backfill, and the ledger record together, and that a clean retry then
  succeeds.

**Seam 4 — recurring-writer schema fence** (`atlas_brain/main.py`
`_run_database_migration_check`; `atlas_brain/storage/repositories/invoice.py`
`recurring_invoice_dedup_schema_ready`; `atlas_brain/mcp/invoicing_draft_writer_server.py`
`_lifespan`)

- Replaced behavior: the existing fence in `main.py` checked only migration
  382 under `receivables_api_enabled` — never the recurring dedup schema, and
  never under `auto_invoice_enabled` (an independent flag the legacy cron alone
  depends on). A later review found the broad migration-385/readiness fence was
  too blunt: it would also block ad-hoc/draft invoice and base receivables
  surfaces that do not create recurring invoices.
- Guard-relevant fields: `settings.invoicing.receivables_api_enabled`,
  `settings.invoicing.auto_invoice_enabled`, the recurring-dedup schema shape
  (`invoices.billing_period`, reservations table columns, both invoice checks,
  and a valid/ready unique index).
- Caller x input: {receivables_api_enabled=true x recurring schema ready/not
  ready, auto_invoice_enabled=true x recurring schema ready/not ready, both
  flags false (no recurring fence should fire), the real `main.lifespan(...)`
  end to end for the auto-invoice-only shape, the draft-writer server's real
  `_lifespan` with a ready/not-ready draft schema}.
- Disposition: every class above is asserted across
  `test_full_atlas_migration_check_allows_recovered_or_disabled_receivables`,
  `test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_readiness`,
  `test_full_atlas_migration_check_allows_auto_invoice_when_dedup_schema_ready`,
  `test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness`,
  the two `test_invoicing_draft_writer_mcp.py` lifespan tests, the new
  invoice repository create tests, and the EOM/receivables readiness tests
  that now prove migration 385 is not part of base receivables readiness.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql`
- `atlas_brain/storage/repositories/invoice.py`
- `plans/PR-EOM-Recurring-Invoice-Period-Dedup.md`
- `tests/maturity_sweep/baseline_atlas_brain_mcp.json`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_commercial_billing_runs.py`
- `tests/test_eom_render_profile.py`
- `tests/test_invoice_repository.py`
- `tests/test_invoicing_draft_writer_mcp.py`
- `tests/test_legacy_monthly_autoinvoice_writer_harness.py`
- `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py`
- `tests/test_receivables.py`

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

### Amendment: historical backfill

The two backfill passes derive `billing_period` for pre-migration rows from
data each writer already persisted: `monthly_auto`'s `source_ref` always ends
`_{YYYY-MM}` (stable since the writer's first commit); `eom_commercial_billing`'s
`invoice_number` always has the shape `INV-{YYYY-Mon}-{seq}` (every row of
this source postdates that format's introduction). The month-abbreviation set
is enumerated literally, not matched by shape alone, because Postgres's
`to_date` raises on an unrecognized abbreviation rather than failing safe — an earlier
draft using a shape-only regex was verified to crash on exactly this input
before the fix. A row whose format predates either convention is left `NULL`
— inert, not guessed at.

Before backfilling, both passes compute a `collisions` CTE: any `(contact_id,
candidate_period)` pair with more than one otherwise-backfillable row (across
or within a source — the unique index doesn't key on `source`, so two
`monthly_auto` rows for the same contact+period would also collide) is
excluded from the backfill and instead stamped with a `metadata` marker
(`billing_period_backfill_collision`, `billing_period_backfill_candidate_period`)
so an operator can find and manually reconcile a genuine historical
duplicate rather than have it silently deleted, guessed at, or left to abort
the whole migration. The collision CTE explicitly excludes `contact_id IS
NULL` rows from grouping against each other: SQL's `GROUP BY` treats multiple
`NULL`s as one group, but Postgres's unique index treats every `NULL`
`contact_id` as distinct from every other — grouping them the naive way would
falsely quarantine unrelated customers' invoices that happen to derive the
same period. Verified directly: a NULL-`contact_id` pair sharing a derived
period backfills independently, with no quarantine metadata, exactly as the
real index would allow.

The migration deliberately does **not** opt into this repo's
`-- atlas: atomic-bookkeeping` marker. Migration 385 is a production-facing
backfill plus uniqueness rollout, so it uses `CREATE INDEX CONCURRENTLY` /
`DROP INDEX CONCURRENTLY` and `NOT VALID` constraints. That shape avoids
packing a backfill and index build into one long table-locking transaction.
The startup guard therefore checks the schema objects that recurring writers
actually need (`billing_period`, reservation columns, checks, and a
valid/ready unique index) rather than treating a migration-ledger row as the
only readiness signal.

### Amendment: startup fence

`atlas_brain/main.py` keeps the migration-382 ledger fence for the receivables
review-recovery table. Separately, when either recurring writer is reachable
(`receivables_api_enabled` for approval writes, or the master-enabled legacy
auto-invoice task), it calls
`recurring_invoice_dedup_schema_ready(pool)` and fails closed if the schema
objects are not ready. This is a recurring-writer fence, not a general
receivables startup requirement.

The broad readiness side effect is removed from
`atlas_brain/services/receivables.py`, so payment tracking and the full
invoicing MCP's base readiness do not require recurring-only columns/indexes.
`InvoiceRepository.create()` also omits the `billing_period` column entirely
when no recurring billing period is supplied, preserving pre-385 ad-hoc/draft
invoice writes on schemas that only need the base invoice fields.

`atlas_brain/mcp/invoicing_draft_writer_server.py` reuses the full MCP
lifespan wrapper but provides a smaller migration/readiness pair: only the
contact/invoice migrations needed to insert a draft invoice, and a
`_draft_invoice_schema_ready()` check over the invoice columns that surface
actually writes. `atlas_brain/main_eom.py` likewise leaves migration 385 out
of `EOM_RECEIVABLES_READINESS_MIGRATIONS`, because that profile's payment
readiness does not by itself create recurring invoices.

### Amendment: quarantine reservation

A new `invoices_billing_period_reservations` table (`(contact_id,
billing_period)` as a real primary key) is created inside migration 385,
before the backfill. Backfill 2/2's collision-quarantine `UPDATE` is
rewritten as a writable CTE chain: the same `UPDATE ... RETURNING` that
stamps `metadata` on each ambiguous row now feeds directly into `INSERT INTO
invoices_billing_period_reservations ... SELECT DISTINCT ... FROM
quarantined ON CONFLICT DO NOTHING`, so the exact set of rows just
quarantined is also exactly the set reserved — the two can never drift apart
from a third scan recomputing `collisions` differently.

Both `InvoiceRepository.get_by_contact_and_period` and
`CommercialBillingApprovalService._find_recurring_period_conflict` are
extended from a plain `SELECT ... FROM invoices` to a `UNION ALL` with a
second branch reading the reservation table, scoped to reservations that
still have at least one matching non-void quarantined invoice. That branch
synthesizes `source = 'quarantined_collision'` and a `invoice_number` string
that names the real diagnostic query — so a hit on a quarantined period
reads distinctly in logs/error messages from a hit on a real invoice,
without either caller needing a second code path (both callers already only
read the `source`/`invoice_number` keys of the returned dict). When every
matching quarantined invoice is voided, the reservation remains as
historical evidence but stops blocking a clean reissue, matching the partial
unique index's `status <> 'void'` contract. `get_by_contact_and_period`'s
`SELECT *` was narrowed to `SELECT source, invoice_number` for this — the
only two fields either caller ever reads — so the two `UNION ALL` branches
have a compatible shape without projecting placeholder values into every
other column.

This design was chosen over two alternatives, both rejected for reintroducing
the exact problem the backfill's collision quarantine exists to avoid: (a)
writing a real `billing_period` onto one specific quarantined row (whichever
is "first" by some tiebreak) — this would make that specific historical
invoice findable by `get_by_contact_and_period` as if it were the
authoritative one, silently picking a winner among genuinely ambiguous
records; (b) inserting a synthetic placeholder row into `invoices` itself to
hold the real unique-index slot — this would pollute real invoice listings,
revenue totals, and any dashboard that scans the `invoices` table, with a
row that isn't a real invoice.

The tradeoff this design accepts, stated directly in the migration's own
comment rather than left implicit: the reservation is enforced by the two
writers' pre-checks, not by `idx_invoices_recurring_contact_period_source`
itself — no DB constraint can enforce a slot without a real row claiming it,
and this design deliberately avoids creating one. A future writer that
creates recurring invoices without going through either existing pre-check
would need to be audited against this table too. This is a narrower
guarantee than the unambiguous-period case gets from the partial unique
index, and it is disclosed as such, not silently absent — which is the
entire point of this fix relative to doing nothing.

### CI-caught fixes (not Codex findings, discovered by live CI on the pushed branch)

Two real gaps surfaced by GitHub Actions on this PR that no local gate run
this session had exercised:

- `tests/maturity_sweep/baseline_atlas_brain_mcp.json` and
  `tests/maturity_sweep/baseline_atlas_brain_storage.json` were stale
  against the PR's intentionally added test seams. Regenerated via the
  CI-documented `--update-baseline` command rather than hardening unrelated
  old files inside this production-fence slice.
- `tests/test_legacy_monthly_autoinvoice_writer_harness.py`'s fixed
  `_HARNESS_MIGRATIONS` tuple never included migration 385, so its
  real-Postgres proof of the *actual* legacy writer code — not a mock —
  hit `column "billing_period" does not exist` the moment that code tried
  to persist an invoice. This is different from the harness-level dedup
  proof this plan's Deferred section already declined to add (a *new*
  capability this PR chose not to extend that armed harness with); this is
  the *existing* harness breaking because the legacy writer it exercises now
  unconditionally references a column that harness's fixture didn't have.
  Not optional to fix. Verified against a harness-compliant container
  (loopback, port 5432, database `atlas_receivables_test`, `ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_HARNESS=1`):
  9 passed, matching CI's own count.

### Amendment: round 3 hardening

**Finding #4 — sequence-width fix.** Both backfill passes' `eom_commercial_billing`
branch matched `^INV-\d{4}-(Jan|...|Dec)-\d{4}$` and extracted with the
matching `\d{4}` group. `commercial_billing_approvals.py`'s number generator
(`lpad(nextval('invoice_number_seq')::text, 4, '0')`) only guarantees a
*minimum* width of 4 — `lpad` does not truncate a longer string, so a
sequence value of 10000+ produces `INV-2026-Oct-10000`, five digits, which
the old regex's exact `\d{4}$` anchor rejected outright (no partial match,
no error — the row simply fell through to `ELSE NULL` like a genuinely
unparseable legacy row, indistinguishable from one). Fixed by widening both
occurrences (the match and the extraction substring, in both backfill
passes) from `\d{4}` to `\d{4,}`. Verified against real Postgres: a
`>9999`-sequence invoice number now backfills its `billing_period`
correctly and is protected by the same collision/reservation machinery as
every other row —
`test_invoice_repository.py::test_real_postgres_billing_period_backfill_and_collision_handling`,
extended with a fifth contact whose sole invoice uses a five-digit sequence.
Negative-control proof that this is a real regression test, not a
tautology: the fix was reverted locally, the same test re-run, and it failed
exactly as expected (`billing_period` stayed `NULL`) before being restored.

**Finding #5 — `main_eom.py` readiness scope corrected after later review.**
An earlier round added migration 385 to EOM receivables readiness because the
EOM profile can mount receivables routes. The class-level repair narrows that:
EOM/base receivables readiness may require the payment/receipt schema, but it
must not require the recurring-invoice dedup schema unless that process is
actually running a recurring invoice writer. This PR therefore leaves
`385_invoices_billing_period_dedup` out of
`EOM_RECEIVABLES_READINESS_MIGRATIONS` and removes the recurring
`billing_period` column/index from the base `ReceivablesService.is_ready()`
contract.

**Finding #6 — rolling-deployment skew: investigated and waived, not fixed.**
Checked fresh against this machine's actual running deployment, not
recalled from memory. `atlas-api.service` (`main.py`'s real production
unit) is a systemd user service, `Type=simple`, a single `MainPID`,
`ExecStart=... uvicorn atlas_brain.main:app --host 127.0.0.1 --port 8012`.
Its own deployment history (recorded as dated comments in the unit file
itself, one per past "Provider cutover") shows every prior deploy repoints
`WorkingDirectory` to a new worktree, then `daemon-reload && restart` — a
hard stop-then-start cutover, never two instances of this process running
concurrently against the same port. There is no rolling-deployment
mechanism in this process's actual production topology for old-code/
new-schema or new-code/old-schema overlap to occur in. `main_eom.py`'s
Render `pserv` blueprint (which theoretically supports Render's rolling
deploys) is, per the finding-#5 investigation above, still an unconnected
draft — confirmed fresh via `render.eom.yaml`'s header comment and the
absence of any local systemd unit or other process currently running
`main_eom:app`. Disposition: **waived as speculative for `main.py`** (its
real deployment mechanism categorically cannot rolling-deploy) and
**deferred for `main_eom.py`** (not yet connected; when it is, the
finding-#5 fence already closes the new-code/old-schema half of the skew
window, since a freshly-started instance now fails closed on an unready
schema instead of serving against it — the old-code/new-schema half, if
Render's rolling deploy keeps a prior instance alive during a migration
window, would need its own follow-up once this profile is actually
connected and using real rolling deploys). See Deferred.

**Finding #7 — auto-invoice fence scoped to the master gate.** The fence's
call site in `main.py`'s `lifespan()` passed
`auto_invoice_enabled=settings.invoicing.auto_invoice_enabled` directly,
without the master `settings.invoicing.enabled` gate the legacy task itself
checks first — `monthly_invoice_generation.py` (lines 82-86) returns
`{"_skip_synthesis": "Invoicing disabled"}` before ever reading
`auto_invoice_enabled` or touching `billing_period` when `enabled=False`.
(By contrast, `receivables_api_enabled` alone is already a correct,
complete reachability condition — `require_receivables_api()`
(`atlas_brain/api/invoicing/auth.py`) depends only on that flag, not on
`invoicing.enabled`, so no equivalent gap exists on that side; verified by
reading both call sites, not assumed from symmetry.) So a deployment with
invoicing entirely disabled but a stale `auto_invoice_enabled=True` left in
config — a state that costs nothing to reach, since the two flags are
independent settings — was false-positive-blocked from starting at all, for
a code path proven unreachable by the task's own first line. Fixed by
scoping the call site to `settings.invoicing.enabled and
settings.invoicing.auto_invoice_enabled`. Verified with a new test driving
the real `main.lifespan(...)` with a mocked `_run_database_migration_check`
that captures its kwargs then raises immediately (avoiding the need to mock
this file's unrelated LLM-loading/evidence-engine startup surface) —
`test_commercial_billing_runs.py::test_full_atlas_lifespan_scopes_auto_invoice_fence_to_master_invoicing_gate`
asserts the captured `auto_invoice_enabled` is `False` despite the stale
flag being `True`, given `invoicing.enabled=False`. Negative control: the
pre-existing
`test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness`
(now updated to set `invoicing.enabled=True` explicitly, matching the
healthy-deployment shape it was already meant to model) proves the fence
still fires when the master gate is genuinely on.

**Finding #8 — cross-pipeline dedup check now fails closed.** Discovered
on a follow-up sweep of the review threads after findings #4/#5/#7 were
already pushed. `monthly_invoice_generation.py`'s per-bundle dedup logic
originally wrapped both the pre-existing `get_by_source_ref` check and the
new `get_by_contact_and_period` check in one `try/except Exception: logger.warning(...)`
— any exception from either call was logged and the code fell through to
`create()`. That fail-open behavior was already correct and intentional
for `get_by_source_ref` (its own gap is closed by the pre-existing
`(source, source_ref)` partial unique index from migration 372 regardless
of whether the app-level check ran), but is not correct for
`get_by_contact_and_period`: a quarantined historical collision (Amendment:
quarantine reservation, above) is protected only by
`invoices_billing_period_reservations`, a table with no DB-level constraint
enforcement of its own — the whole point of that table's design, disclosed
in the migration's own header comment, is that the two writers' pre-checks
*are* the enforcement for that narrower case. A transient read error on
`get_by_contact_and_period` at exactly the wrong moment would silently
admit the unprotected third duplicate the reservation table exists to
prevent. Fixed by splitting the shared `try/except` into two: the
`get_by_source_ref` block keeps its original fail-open behavior unchanged;
`get_by_contact_and_period` now fails closed on any exception — increments
a new `invoices_skipped_dedup_check_failed` counter, logs a distinct
warning, and `continue`s past `create()` for that contact this run, rather
than proceeding. The run-level "no invoices generated" early-return gate is
extended to also check this new counter, so a run where every contact hits
this failure mode still logs its summary line rather than returning early
with no trace of what happened. Verified with a new test
(`tests/test_monthly_invoice_generation_cross_pipeline_dedup.py::test_legacy_writer_fails_closed_when_cross_pipeline_check_errors`):
a contact whose `get_by_contact_and_period` call raises is skipped without
a `create()` call, while a second, healthy contact in the same run still
creates normally — proven as a genuine regression, not a tautology, by
temporarily stashing the fix and confirming the test fails
(`assert 2 == 1`, i.e. the flaky contact WAS created) before restoring it.

**Finding #9 — migration replay must preserve a valid recurring-dedup index.**
Review round 4 found a replay hole: if migration 385 had already built a
valid `idx_invoices_recurring_contact_period_source` but failed before the
ledger row was recorded, a later replay could drop the valid index while a
recurring writer was already allowed to run by schema readiness. Fixed in
`c19168bfcacd0f8ef3ee8997edce1b87d03b7982`: migration 385 now drops only a
renamed invalid remnant and creates the index with `CREATE UNIQUE INDEX
CONCURRENTLY IF NOT EXISTS`, so a valid live fence stays in place during
replay. Verified by
`tests/test_invoice_repository.py::test_invoices_billing_period_dedup_migration_is_additive_and_scoped`.

**Finding #10 — rollback order must remove the fresh-write fence before old
writers resume.** Review round 4 correctly contradicted the earlier rollback
comment: old recurring writers omit both `billing_period` and
`billing_period_legacy_null`, so leaving
`invoices_recurring_billing_period_required_check` in place would reject their
fresh inserts. Fixed in `c19168bfcacd0f8ef3ee8997edce1b87d03b7982`: the
migration rollback note now explicitly drops that constraint before old
writers resume, while leaving inert period columns/indexes/reservations in
place. Verified by the same migration contract test.

**Finding #11 — draft-writer readiness must check the full draft-runtime
schema.**
Review round 4 found the minimal draft schema probe checked invoice columns
too narrowly: `InvoiceRepository.create()` needs invoice number, customer,
amount, source, metadata, and timestamp columns; `update_invoice()` needs
amount and metadata columns; `get_invoice()` reaches `get_payments()`, which
joins `invoice_payments` to `customer_payments` and expects the post-344
allocation columns. Fixed in
`c19168bfcacd0f8ef3ee8997edce1b87d03b7982` and completed in this repair
commit:
`_draft_invoice_schema_ready()` now requires the complete draft-runtime
column/relation set, and the draft writer's curated migration set includes
`344_receivables_payments` because the tool already exposes `get_invoice`.
Verified by
`tests/test_invoicing_draft_writer_mcp.py::test_draft_invoice_schema_ready_requires_runtime_dependencies`
and the real-Postgres curated-migration smoke when
`ATLAS_RECEIVABLES_TEST_DATABASE_URL` is available.

**Finding #12 — quarantine reservations must follow the non-void invariant.**
Review round 4 found that reservation rows were permanent blockers even after
an operator voided every invoice in the historical collision group. Fixed in
`c19168bfcacd0f8ef3ee8997edce1b87d03b7982`:
`InvoiceRepository.get_by_contact_and_period()` and the approval service's
transaction-scoped conflict query now treat a reservation as a block only
while a matching non-void quarantined invoice still exists. Verified
by `tests/test_invoice_repository.py::test_real_postgres_billing_period_backfill_and_collision_handling`
when real Postgres is configured; in the local no-URL environment, that test
is collected but skipped.

**Finding #13 — voided quarantine rows must not be resurrectable through the
shared status updater.** Review round 5 found a caller-independent hole:
after every quarantined collision invoice was voided, a clean reissue could
be created, but the old voided NULL-period invoice could still be marked
`sent` through the shared `InvoiceRepository.update_status()` path. Because
the old row still has `billing_period=NULL`, the partial unique index cannot
collide it with the clean replacement. Fixed in this repair commit:
`update_status()` now activates a migration-aware guard when
`billing_period`/`billing_period_legacy_null` exist and rejects
`void -> non-void` transitions for legacy recurring NULL-period rows. Verified
by the real Postgres backfill/collision test and
`tests/test_invoice_repository.py::test_invoice_status_update_blocks_quarantined_void_resurrection_in_repository`.

**Finding #14 — the duplicate-index regression test must target the right
failure layer.** CI showed the test still inserted a fresh recurring duplicate
without `billing_period`, so migration 385's new required-period CHECK rejected
it before the unique index was reached. That is correct behavior for a raw old
writer, but it no longer proves the index. Fixed in this repair commit: the unique
index proof now inserts a same-contact/same-period duplicate with
`billing_period='2026-04'`, while the separate missing-period CHECK proof
remains in place for raw recurring rows without the period.

**Finding #15 — monthly cross-pipeline regression tests must be enrolled in
PR CI.** Review round 6 found the new
`tests/test_monthly_invoice_generation_cross_pipeline_dedup.py` file was
referenced as Review Contract evidence but was not included in
`.github/workflows/atlas_invoicing_checks.yml` path filters or its explicit
pytest command. Fixed in this repair commit: the workflow now triggers on
that file and runs it in the receivables/repository job, so the required PR
checks exercise the monthly writer hit and fail-closed branches instead of
leaving them to the scheduled repo-wide backstop.

**Finding #16 — recurring dedup readiness must verify definitions, not just
names.** Review round 7 found that an externally managed schema could contain
same-named, valid, ready objects with weaker definitions — for example a
unique index on `(contact_id, billing_period, source)`, which would allow the
two recurring sources to coexist for the same contact+period. Fixed in this
repair commit: `recurring_invoice_dedup_schema_ready()` now reads
Postgres catalog definition text through the `pg_get_constraintdef` and
`pg_get_indexdef` built-ins and verifies the actual required CHECK/index
shapes, including the two-column index key and recurring source/void
predicate. Verified by
`tests/test_invoice_repository.py::test_real_postgres_recurring_dedup_readiness_rejects_drifted_definitions`
when real Postgres is configured.

**Finding #17 — the standalone monthly task-level readiness fence needed
direct run evidence.** Review round 7 found that the new early-return and
exception branches were not exercised by the monthly writer tests because the
fake invoice repository had no `recurring_dedup_ready()` method. Fixed in this
repair commit: the fake repo now implements that method, and two direct
`run()` tests assert that readiness `False` or readiness exceptions stop
before service loading, cross-pipeline lookup, or invoice creation.

**Finding #18 — recurring dedup readiness must compare complete index
semantics.** Review round 8 found the previous definition-readiness repair
still used token fragments, so a same-named valid index on the correct columns
but with an impossible extra predicate clause could pass readiness while
indexing no recurring rows. Fixed in this repair commit:
`recurring_invoice_dedup_schema_ready()` now reads catalog key columns and
predicate expression separately, requires exactly `(contact_id,
billing_period)`, and compares the normalized predicate clause set exactly.
Verified by
`tests/test_invoice_repository.py::test_recurring_dedup_predicate_readiness_rejects_extra_clauses`
and the real-Postgres drift test when configured.

**Finding #19 — migration 385 must trigger the invoicing workflow.** Review
round 8 found that the workflow ran the new regression tests but did not list
`atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql` in
either the pull-request or push path filters. Fixed in this repair commit:
both filter lists now include migration 385, so migration-only follow-ups run
the invoice repository, approval, monthly-writer, and harness proofs.

**Finding #20 — verification must not claim atomic rollback for a non-atomic
migration.** Review round 8 found an obsolete verification sentence still
claimed a forced mid-migration failure rolled back the column/table/ledger
together. That contradicts migration 385's intentional non-atomic concurrent
index shape. Fixed in this repair commit: the verification text now describes
the actual recovery proof — idempotent retry and no duplicate/changed state
after re-running the already-applied migration — rather than atomic rollback.

**Finding #21 — readiness must include migration 385's full NULL predicate.**
Review round 9 found the exact-predicate repair omitted the
`billing_period IS NOT NULL` clause that migration 385 installs on
`idx_invoices_recurring_contact_period_source`, so startup readiness would
reject the freshly migrated schema. Fixed in this repair commit: the expected
predicate clause set now includes the migration's NULL guard, the in-memory
approval fake returns that full predicate, and both pure and real-Postgres
readiness tests use the migration-equivalent predicate as the accepted shape
while still rejecting impossible extra clauses.

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
- Historical `billing_period` IS backfilled, revised from an earlier draft of
  this plan that deferred it — see Amendment: historical backfill for the
  full reasoning and Codex's original finding.
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
- The startup fence is generalized in-place (a parameterized check function,
  a shared exception base) rather than adopting the migration-content-hash
  infrastructure a concurrently-open, unrelated PR (#2447) was building —
  confirmed via `git merge-base --is-ancestor` that PR had not merged to
  `origin/main` as of this work; depending on unmerged code was rejected.
- `invoicing_draft_writer_server.py`'s fix now also runs `run_migrations` on
  that process, not only a readiness check — this is a real behavior
  addition (that process previously ran zero migration logic), but it is
  exactly what `invoicing_server.py`'s own lifespan already does today for
  the identical concern, and `run_migrations` is idempotent and
  advisory-lock-serialized against concurrent callers by design, so a second
  process calling it is the established pattern, not a new risk.
- The quarantine reservation is a dedicated small table, not a `billing_period`
  value written onto one historical row and not a synthetic invoice row —
  see Amendment: quarantine reservation for why both alternatives were
  rejected as reintroducing the exact "guess a winner" problem the backfill's
  collision quarantine exists to avoid.

## Deferred

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
- Adopting PR #2447's migration-content-hash verification infrastructure for
  this fence — that PR had not merged as of this work; the generalized
  one-off pattern here is a smaller, immediately-available fix and does not
  block adopting the more general infrastructure later if #2447 lands.
- Rolling-deployment old-code/new-schema skew protection for
  `main_eom.py`'s Render `pserv` blueprint (round-3 finding #6) — that
  blueprint is still an unconnected draft (`render.eom.yaml`), so there is
  no live rolling-deploy window today; the new-code/old-schema half is
  already closed by this PR's finding-#5 fence. Revisit once that profile
  is actually connected and Render's rolling-deploy behavior for it is
  configured, not before — fixing it now would be speculative hardening
  against a deployment shape that does not yet exist.

Parking predicate: hardening beyond the two recurring sources' cross-
awareness (broader financial-integrity auditing, the unrelated void-filter
gap) is parked unless necessary to prove this specific constraint or one of
the two amendments above. Nothing here qualifies.

Parked hardening: none.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://atlas:atlas@127.0.0.1:<port>/atlas
  pytest tests/test_invoice_repository.py tests/test_commercial_billing_approvals.py
  tests/test_monthly_invoice_generation_cross_pipeline_dedup.py
  tests/test_receivables.py tests/test_commercial_billing_runs.py
  tests/test_invoicing_draft_writer_mcp.py tests/test_invoicing_draft_writer_oauth.py
  tests/test_check_invoicing_draft_writer_funnel_routes.py
  tests/test_check_invoicing_draft_writer_live_write.py
  tests/test_check_invoicing_draft_writer_mcp_connector.py
  tests/test_check_invoicing_draft_writer_oauth_discovery.py
  tests/test_check_invoicing_draft_writer_oauth_e2e.py
  tests/test_start_invoicing_draft_writer_oauth_server.py
  tests/test_invoicing_readonly_mcp.py tests/test_invoicing_readonly_oauth.py
  tests/test_eom_render_profile.py -q`
  against a throwaway `postgres:16` container (not the shared dev DB): **464
  passed**, 0 failed (re-run after finding #8's fix; was 463 after findings
  #4/#5/#7, and 460 after the two round-2 amendments). This is every test
  file touched by any of the original slice, the round-2 amendments, or
  round 3's fixes, run together, including all pre-existing tests in each —
  confirming no regression anywhere the changes could plausibly reach.
  `test_eom_render_profile.py` was added to this list after the repo's full
  local unit-gate mirror caught a real regression this list didn't
  originally include: a second, independent static test pinning
  `EOM_RECEIVABLES_READINESS_MIGRATIONS`'s exact contents, in a different
  file than the real-schema readiness test already covered here; it now
  also carries round 3's two new `main_eom.py` readiness-fence lifespan
  tests. `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py`
  now also carries finding #8's fail-closed regression test.
- `ruff check` on all seventeen changed source/test files: zero new
  findings. `atlas_brain/main.py` and `atlas_brain/main_eom.py` each carry
  pre-existing E402 findings (intentional `load_dotenv`-before-imports
  pattern) — confirmed identical counts (26 and 15 respectively, 41 total)
  against each file's `origin/main` baseline before concluding these are
  not new, re-checked again after every round 3 edit, including finding
  #8's.
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
  - A NULL-`contact_id` pair sharing a derived period backfills
    independently, uncounted as a collision — proving the collision CTE's
    NULL-safety fix, not merely that it compiles.
  - Migration 385 is explicitly non-atomic and production-shaped: the static
    migration test asserts no `-- atlas: atomic-bookkeeping` marker, asserts
    `CREATE INDEX CONCURRENTLY`/`DROP INDEX CONCURRENTLY`, and asserts
    `NOT VALID` constraints plus a non-zero-year period recognizer.
  - The startup fence: `auto_invoice_enabled=True` with recurring dedup schema
    not ready raises and closes the pool; the identical shape with schema ready
    returns normally; the real `main.lifespan(...)` end to end raises for the
    auto-invoice-only deployment shape with recurring schema not ready.
  - The draft-writer MCP server's lifespan: a ready draft-runtime schema
    serves normally (init → curated migrations → ready-check → serving →
    close); an incomplete runtime schema raises and closes before serving.
  - The quarantine reservation: a raw third insert with `billing_period` set
    explicitly to a quarantined period's value succeeds cleanly with no
    constraint violation before this fix (proving the gap), then the
    app-level pre-check in both writers correctly returns/raises a
    `quarantined_collision` hit after it; the same contact with a different
    (unreserved) period is not a conflict and approves normally. Migration
    385 is non-atomic by design, so recovery evidence is idempotent retry
    rather than rollback: the migration uses `IF NOT EXISTS`/drop-recreate
    guards around persistent objects, freezes collision membership inside
    the active migration session, and the real-Postgres backfill test reruns
    the already-applied migration and proves it produces no duplicate or
    changed invoice/reservation state.
  - Round 3, finding #4: a five-digit (`>9999`) `eom_commercial_billing`
    invoice-number sequence now backfills `billing_period` correctly and is
    protected by the same collision/reservation machinery — proven by
    reverting the regex-width fix locally, re-running the same test, and
    watching it fail exactly as expected (`billing_period` stayed `NULL`),
    then restoring the fix and confirming it passes again.
  - Round 3, finding #5: `main_eom.py`'s new readiness fence raises and
    closes the pool on an unready schema, and serves normally once ready —
    both directions exercised through the real `main_eom.lifespan(...)`,
    mirroring the file's existing lifespan-test pattern.
  - Round 3, finding #7: the real `main.lifespan(...)` computes
    `auto_invoice_enabled=False` for the migration-385 fence when
    `invoicing.enabled=False`, even with the stale flag `auto_invoice_enabled=True`
    still set — proving the false-positive block is gone. Negative control:
    the sibling healthy-deployment test (`invoicing.enabled=True`) still
    fires the fence, proving the master gate didn't just disable the fence
    outright.
  - Round 3, finding #8: a contact whose `get_by_contact_and_period` call
    raises is skipped without a `create()` call, incrementing the new
    `invoices_skipped_dedup_check_failed` counter; a second, healthy
    contact in the same run still creates normally, proving the failure is
    per-contact, not a run-wide abort. Proven as a genuine regression, not
    a tautology, by temporarily stashing the fix and confirming the test
    fails (`assert 2 == 1`, i.e. the flaky contact was created) before
    restoring it.
- The unique-index design itself was verified against the exact duplicate
  scenario before being implemented: a draft with `source` inside the
  index's column list was checked against a same-contact/period,
  different-source pair and found to admit both rows, which is how the
  final index (source in the predicate only) was arrived at rather than
  shipped incorrectly.
- Focused round-5 repair verification: `python -m pytest
  tests/test_invoicing_draft_writer_mcp.py::test_draft_invoice_schema_ready_requires_runtime_dependencies
  tests/test_invoicing_draft_writer_mcp.py::test_draft_writer_curated_migrations_apply_to_empty_schema
  tests/test_invoice_repository.py::test_real_postgres_billing_period_backfill_and_collision_handling
  tests/test_invoice_repository.py::test_invoices_billing_period_dedup_migration_is_additive_and_scoped
  tests/test_invoice_repository.py::test_invoice_status_update_blocks_quarantined_void_resurrection_in_repository -q`
  passed locally: **3 passed, 2 skipped**.
- Current repair verification: `python -m pytest tests/test_invoice_repository.py
  tests/test_monthly_invoice_generation_cross_pipeline_dedup.py
  tests/test_invoicing_draft_writer_mcp.py
  tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_receivables_readiness
  tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_receivables_without_ready_schema
  tests/test_eom_render_profile.py::test_eom_lifespan_accepts_enabled_receivables_with_ready_schema
  tests/test_receivables.py::test_eom_receivables_readiness_migration_set_builds_ready_schema
  tests/test_receivables.py::test_eom_readiness_migration_set_is_closed_over_receivables_dependencies
  tests/test_commercial_billing_runs.py::test_full_atlas_migration_check_allows_recovered_or_disabled_receivables
  tests/test_commercial_billing_runs.py::test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_readiness
  tests/test_commercial_billing_runs.py::test_full_atlas_migration_check_blocks_when_dedup_readiness_query_errors
  tests/test_commercial_billing_runs.py::test_full_atlas_migration_check_allows_auto_invoice_when_dedup_schema_ready
  tests/test_commercial_billing_runs.py::test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_readiness
  tests/test_commercial_billing_approvals.py
  tests/test_legacy_monthly_autoinvoice_writer_harness.py -q`
  passed locally: **124 passed, 38 skipped, 1 warning**.
- Current repair lint: `python -m ruff check` over the touched modules/tests
  except baseline-noisy entrypoints `atlas_brain/main.py` and
  `atlas_brain/main_eom.py` passed with **All checks passed!**. Running ruff
  with those two entrypoints included reports the pre-existing 41 E402
  findings already documented above, and no recurrence of the unrelated
  `invoicing_server.py` F841 baseline noise after removing that file from this
  repair surface.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 3 |
| `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` | 86 |
| `atlas_brain/main.py` | 132 |
| `atlas_brain/main_eom.py` | 30 |
| `atlas_brain/mcp/invoicing_draft_writer_server.py` | 121 |
| `atlas_brain/services/commercial_billing_approvals.py` | 64 |
| `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql` | 311 |
| `atlas_brain/storage/repositories/invoice.py` | 303 |
| `plans/PR-EOM-Recurring-Invoice-Period-Dedup.md` | 1061 |
| `tests/maturity_sweep/baseline_atlas_brain_mcp.json` | 4 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 27 |
| `tests/test_commercial_billing_approvals.py` | 292 |
| `tests/test_commercial_billing_runs.py` | 323 |
| `tests/test_eom_render_profile.py` | 90 |
| `tests/test_invoice_repository.py` | 540 |
| `tests/test_invoicing_draft_writer_mcp.py` | 173 |
| `tests/test_legacy_monthly_autoinvoice_writer_harness.py` | 44 |
| `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py` | 239 |
| `tests/test_receivables.py` | 11 |
| **Total** | **3854** |
