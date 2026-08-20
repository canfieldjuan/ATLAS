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
  anything that would collide; fence every process that can reach
  `InvoiceRepository.create()`/the raw `invoices` INSERT against migration
  385 not being recorded.
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
  migration 382 (extended, not replaced); `invoicing_server.py`'s own code
  (its existing readiness-check hook picks up the new requirement without any
  edit there).

## Scope (this PR)

Ownership lane: eom/recurring-invoice-period-dedup
Slice phase: Production hardening
Max files: 18

1. Migration 385: `invoices.billing_period` column + format CHECK + the
   cross-source partial unique index + historical backfill + collision
   quarantine, wrapped atomically (see Amendment: historical backfill).
2. Persist `billing_period` in both writers' existing `INSERT`s, reusing each
   writer's already-computed period value — zero new parameters on either
   `InvoiceRepository.create()` or `_InvoiceDraft`.
3. `InvoiceRepository.get_by_contact_and_period` and a matching
   transaction-scoped lookup in `commercial_billing_approvals.py`; call both
   from their respective writers as a pre-check ahead of the new constraint.
4. `atlas_brain/main.py`: generalize the existing migration-recovery fence to
   also require migration 385 whenever `receivables_api_enabled` or
   `auto_invoice_enabled` is true (see Amendment: startup fence).
5. `atlas_brain/services/receivables.py`: extend the existing
   `_RECEIVABLES_REQUIRED_COLUMNS`/`_RECEIVABLES_REQUIRED_INDEXES` readiness
   contract with `invoices.billing_period` and the new index, which fences
   `invoicing_server.py` for free (it already calls this check).
6. `atlas_brain/mcp/invoicing_draft_writer_server.py`: replace its bare,
   unverified `_lifespan` (no migration run, no schema check at all) with a
   reuse of `invoicing_server.py`'s own `_database_lifespan`, matching this
   file's existing convention of reusing that module's internals.
7. `atlas_brain/main_eom.py`: add migration 385 to the curated
   `EOM_RECEIVABLES_READINESS_MIGRATIONS` tuple — that file's own comment
   requires this for any new receivables readiness requirement.
8. Tests proving: the index rejects a raw cross-source duplicate insert
   independent of any app code; an `mcp_tool` invoice doesn't block a
   recurring one; a voided invoice doesn't block reissuance; the approval
   writer rejects when the legacy writer already invoiced the period (and
   does not when the period differs); the legacy writer's `run()` skips a
   contact the new pipeline already invoiced without calling `create()`, and
   still creates normally for an un-invoiced contact in the same pass;
   historical backfill derives the right period per source and quarantines
   real collisions without falsely flagging unrelated NULL-contact_id rows;
   the startup fence blocks both the receivables path and the
   auto-invoice-only path when migration 385 is missing, and does not
   false-positive-block either when it's present; the draft-writer MCP
   server's lifespan now migrates and verifies readiness instead of silently
   starting unchecked.

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
    belongs only in the predicate, opts into atomic-bookkeeping, and includes
    the collision-quarantine marker) — settled by
    `test_invoice_repository.py::test_invoices_billing_period_dedup_migration_is_additive_and_scoped`.
  - Historical rows backfill correctly per source, quarantine real
    collisions, leave void/mcp_tool/unparseable rows untouched, and do not
    falsely quarantine unrelated NULL-`contact_id` rows that share a derived
    period — settled by
    `test_invoice_repository.py::test_real_postgres_billing_period_backfill_and_collision_handling`,
    which also proves the migration is idempotent on replay and that the
    backfilled row closes the transition-window gap end to end (findable by
    the pre-check; a second insert for the same contact+period is rejected).
  - The startup fence blocks `receivables_api_enabled` and, independently,
    `auto_invoice_enabled` when migration 385 is unrecorded, and does not
    block either when it is recorded — settled by four tests in
    `test_commercial_billing_runs.py`
    (`test_full_atlas_migration_check_allows_recovered_or_disabled_receivables`,
    `test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_migration`,
    `test_full_atlas_migration_check_allows_auto_invoice_when_dedup_migration_recorded`,
    `test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_migration`),
    the last exercised through the real `main.lifespan(...)` end to end, not
    only the unit-level check function.
  - The draft-writer MCP server's `_lifespan` now runs migrations and
    verifies readiness instead of silently starting unchecked, and blocks on
    an incomplete schema — settled by
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
  `atlas_brain/main.py`; the receivables readiness contract in
  `atlas_brain/services/receivables.py`; the draft-writer MCP server's
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
  failing partway and leaving inconsistent state (mitigated by
  atomic-bookkeeping, verified by forcing a real mid-migration failure and
  confirming full rollback); code depending on the new column/index starting
  before the migration applied, across three independent processes
  (mitigated by extending the existing fence pattern to all three, reusing
  house convention rather than inventing new mechanisms per process).
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

**Seam 4 — startup fence for migration 385** (`atlas_brain/main.py`
`_run_database_migration_check`; `atlas_brain/services/receivables.py`
`_RECEIVABLES_REQUIRED_COLUMNS`/`_RECEIVABLES_REQUIRED_INDEXES`;
`atlas_brain/mcp/invoicing_draft_writer_server.py` `_lifespan`)

- Replaced behavior: the existing fence in `main.py` checked only migration
  382, only under `receivables_api_enabled` — never migration 385, and never
  under `auto_invoice_enabled` (an independent flag the legacy cron alone
  depends on). `invoicing_server.py` already called a general readiness
  check but that check didn't know about `billing_period`/the new index
  yet. `invoicing_draft_writer_server.py` had no migration run or schema
  check of any kind.
- Guard-relevant fields: `settings.invoicing.receivables_api_enabled`,
  `settings.invoicing.auto_invoice_enabled`, `schema_migrations` (ledger
  record for `385_invoices_billing_period_dedup`).
- Caller x input: {receivables_api_enabled=true x 385 recorded/unrecorded,
  auto_invoice_enabled=true x 385 recorded/unrecorded, both flags false (no
  fence should fire, matching current no-op behavior), the real
  `main.lifespan(...)` end to end for the auto-invoice-only shape, the
  draft-writer server's real `_lifespan` with a ready/not-ready schema}.
- Disposition: every class above is asserted across
  `test_full_atlas_migration_check_allows_recovered_or_disabled_receivables`,
  `test_full_atlas_migration_check_blocks_enabled_auto_invoice_without_dedup_migration`,
  `test_full_atlas_migration_check_allows_auto_invoice_when_dedup_migration_recorded`,
  `test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_migration`,
  and the two new `test_invoicing_draft_writer_mcp.py` lifespan tests.
  `invoicing_server.py` needed no test changes — it already calls the
  general readiness check, which `test_receivables.py`'s existing real-schema
  test (`test_eom_receivables_readiness_migration_set_builds_ready_schema`,
  `assert await service.is_ready() is True`) now proves is `True` only once
  migration 385 has actually applied.

### Files touched

- `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql`
- `atlas_brain/storage/repositories/invoice.py`
- `atlas_brain/autonomous/tasks/monthly_invoice_generation.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/mcp/invoicing_draft_writer_server.py`
- `tests/test_invoice_repository.py`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py`
- `tests/test_receivables.py`
- `tests/test_commercial_billing_runs.py`
- `tests/test_invoicing_draft_writer_mcp.py`
- `tests/test_eom_render_profile.py`
- `tests/maturity_sweep/baseline_atlas_brain_storage.json`
- `tests/test_legacy_monthly_autoinvoice_writer_harness.py`
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

The migration opts into this repo's `-- atlas: atomic-bookkeeping` marker
(literal first line; nine other migrations already use it, including 384,
the one immediately preceding this one) so the column add, CHECK constraint,
both backfill passes, and the index creation commit or roll back together
with the `schema_migrations` ledger write, in one real transaction. This was
verified empirically, not assumed: forcing a division-by-zero failure between
the backfill and the index creation, on a schema that already had a
pre-migration row, confirmed the column, the backfilled data, and the ledger
record all roll back completely, then a clean retry applies correctly. A
second run of the same test *without* the marker showed the SQL itself was
already atomic even then — asyncpg's simple-query protocol wraps a
multi-statement string in an implicit transaction regardless. What the
marker actually, distinctly adds is narrower: it closes the crash window
between "SQL committed" and "ledger row written," where a process crash
between the two would leave the migration correctly applied but reported as
still pending — safe, since every statement is idempotent, but it would
break the "recorded implies applied" invariant the startup fence below
depends on. The migration's own header comment states this precisely rather
than overclaiming a stronger guarantee than what was actually verified.

### Amendment: startup fence

`atlas_brain/main.py`'s existing `_commercial_billing_review_recovery_is_recorded`/
`_run_database_migration_check` pair is generalized: the specific
single-migration check becomes `_migration_is_recorded(pool, migration_name)`,
and `_run_database_migration_check` now builds an ordered list of required
migrations — migration 382 when `receivables_api_enabled`, migration 385 when
`receivables_api_enabled` **or** `auto_invoice_enabled` — checking each in
turn and closing the pool before raising a specific
`_DatabaseMigrationFenceError` subclass on the first one missing. Both
existing and new exception classes share that common base so the two
`except` sites in `lifespan()` catch either without listing two types.

Adversarial re-review of the first draft of this fix (confined to
`atlas_brain/main.py`) found it structurally could not protect two other,
independent OS processes that also reach `InvoiceRepository.create()`:
`atlas_brain/mcp/invoicing_server.py` (its own lifespan, own migration run,
`settings.mcp.invoicing_enabled` defaults `True`) and
`atlas_brain/mcp/invoicing_draft_writer_server.py` (had no migration run or
schema check at all). Rather than duplicate `main.py`'s fence into each
process, the fix reuses the house convention each already had a hook for:
`invoicing_server.py`'s `_lifespan` already calls
`ReceivablesService(pool).is_ready()`, which checks
`_RECEIVABLES_REQUIRED_COLUMNS`/`_RECEIVABLES_REQUIRED_INDEXES` — extending
that dict/tuple with `invoices.billing_period` and the new index fences that
process **for free**, no code change needed there.
`invoicing_draft_writer_server.py` had no such hook, but already imports the
full `invoicing_server` module (`from . import invoicing_server as _full`)
and freely reuses its other internals (`_full._is_uuid`, `_full._uuid`,
`_full.update_invoice`, ...) — so its `_lifespan` now reuses
`_full._database_lifespan` directly, the exact same contract already proven
in `tests/test_receivables.py`'s `test_standalone_mcp_*` tests, rather than
hand-rolling a third implementation.

`atlas_brain/main_eom.py` maintains its own separately curated, explicitly
"CLOSED and ENUMERATED" tuple of migrations (`EOM_RECEIVABLES_READINESS_MIGRATIONS`)
for the EOM deployment profile — its own header comment requires updating it
for any new receivables readiness requirement, so migration 385 is added
there too; this was discovered by running the existing real-schema readiness
test for that profile and watching it fail, not predicted in advance. A
second, independent static test in `tests/test_eom_render_profile.py`
(`test_eom_startup_migrations_are_curated_for_receivables_readiness`) pins
the tuple's exact contents and needed the same one-line update — caught by
the full local unit-gate mirror, not anticipated from reading either file.

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
second branch reading the reservation table, synthesizing `source =
'quarantined_collision'` and a `invoice_number` string that names the real
diagnostic query — so a hit on a quarantined period reads distinctly in
logs/error messages from a hit on a real invoice, without either caller
needing a second code path (both callers already only read the `source`/
`invoice_number` keys of the returned dict). `get_by_contact_and_period`'s
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

- `tests/maturity_sweep/baseline_atlas_brain_storage.json` was stale —
  6 of the 7 files it flagged as "new brittleness" are untouched by this PR
  (pre-existing drift the ratchet had simply never been run against since
  its last update, long before this branch existed); the 7th
  (`atlas_brain/storage/repositories/invoice.py`) genuinely did increase in
  score from this PR's new method and mock. Regenerated via the CI-documented
  `--update-baseline` command rather than fixing the 6 unrelated files' own
  brittleness, which is out of this slice's scope.
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

**Finding #5 — `main_eom.py` independent readiness fence.** This profile's
`eom_profile_settings.run_migrations` defaults to `False`
(`atlas_brain/eom_api/config.py`), and unlike `main.py` (ledger-based
`_migration_is_recorded`) or the two MCP servers
(`ReceivablesService.is_ready()`), this lifespan had *no* readiness check of
any kind independent of that flag — an enabled receivables API would start
serving requests against whatever schema state happened to exist. Confirmed
this is a live-reachable gap, not theoretical: this file does
`app.include_router(receivables_router, prefix="/api/v1")` unconditionally,
so the commercial-billing approval endpoint is mounted whenever this
process runs (`receivables_api_enabled` gates individual request handling
via `require_receivables_api()`, not whether the router is mounted).
`render.eom.yaml`'s own header comment confirms this profile's Render
deployment is still a draft ("deliberately not named render.yaml yet ...
should be connected manually ... after the branch is reviewed") — not live
today, but the fence protects the moment it is. Fixed by adding
`_require_receivables_schema_ready()`, called from `lifespan()` when
`db_settings.enabled and invoicing_settings.receivables_api_enabled`,
reusing the exact `ReceivablesService(pool).is_ready()` hook the MCP
servers already use rather than inventing a fourth mechanism. Verified with
two new tests mirroring this file's existing lifespan-test pattern
(`tests/test_eom_render_profile.py`): a not-ready schema raises
`ReceivablesSchemaUnavailableError` and the pool still closes (positive
fence); a ready schema serves normally (negative control).

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
`test_full_atlas_lifespan_fences_auto_invoice_only_deployment_without_dedup_migration`
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
  - A forced mid-migration failure (division by zero, injected after the
    backfill and before the index creation) rolls back the column, the
    backfill data, and the ledger record completely; a clean retry then
    applies correctly — run against real Postgres via the actual Python
    migration runner, not raw `psql`.
  - The startup fence: `auto_invoice_enabled=True` with migration 385
    unrecorded raises and closes the pool; the identical shape with 385
    recorded returns normally; the real `main.lifespan(...)` end to end
    raises for the auto-invoice-only deployment shape with 385 unrecorded.
  - The draft-writer MCP server's lifespan: a ready schema serves normally
    (init → migrate → ready-check → serving → close, in order); an
    incomplete schema raises and closes before serving.
  - The quarantine reservation: a raw third insert with `billing_period` set
    explicitly to a quarantined period's value succeeds cleanly with no
    constraint violation before this fix (proving the gap), then the
    app-level pre-check in both writers correctly returns/raises a
    `quarantined_collision` hit after it; the same contact with a different
    (unreserved) period is not a conflict and approves normally; a forced
    mid-migration failure with the reservation table already added to the
    file still rolls back the column, the new table, and the ledger record
    together, then a clean retry produces the expected single reservation
    row for the collision pair.
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
- The atomic-bookkeeping marker's actual, narrower benefit (closing the
  ledger-write crash window, not preventing partial SQL application — see
  Amendment: historical backfill) was established by comparing forced-failure
  behavior with and without the marker, not assumed from the marker's name.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/storage/migrations/385_invoices_billing_period_dedup.sql` | 270 |
| `atlas_brain/storage/repositories/invoice.py` | 60 |
| `atlas_brain/autonomous/tasks/monthly_invoice_generation.py` | 43 |
| `atlas_brain/services/commercial_billing_approvals.py` | 45 |
| `atlas_brain/main.py` | 103 |
| `atlas_brain/main_eom.py` | 31 |
| `atlas_brain/services/receivables.py` | 13 |
| `atlas_brain/mcp/invoicing_draft_writer_server.py` | 29 |
| `tests/test_invoice_repository.py` | 342 |
| `tests/test_commercial_billing_approvals.py` | 215 |
| `tests/test_monthly_invoice_generation_cross_pipeline_dedup.py` | 225 |
| `tests/test_receivables.py` | 31 |
| `tests/test_commercial_billing_runs.py` | 269 |
| `tests/test_invoicing_draft_writer_mcp.py` | 106 |
| `tests/test_eom_render_profile.py` | 91 |
| `tests/maturity_sweep/baseline_atlas_brain_storage.json` | 27 |
| `tests/test_legacy_monthly_autoinvoice_writer_harness.py` | 1 |
| `plans/PR-EOM-Recurring-Invoice-Period-Dedup.md` | 997 |
| **Total** | **2898** |

Round 3 (four Codex findings across two pushes: #4/#5/#7 fixed together,
then #8 found on a follow-up thread sweep and fixed separately) added
~557 LOC on top of round 2's 2341-line total — proportionate to four
confirmed real findings across a migration, two other startup fences, and
two legacy-task call sites, each with its own real-Postgres or
lifespan-level proof, plus a fifth finding (#6, rolling-deployment skew)
investigated and waived rather than fixed.
