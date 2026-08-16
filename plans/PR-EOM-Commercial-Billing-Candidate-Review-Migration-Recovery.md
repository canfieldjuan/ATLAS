# PR-EOM-Commercial-Billing-Candidate-Review-Migration-Recovery

## Why this slice exists

The merged commercial billing-candidate review-decision provider slice ([#2391](https://github.com/canfieldjuan/ATLAS/pull/2391)) cannot be safely deployed. Read-only production catalog evidence found that 380_commercial_billing_candidate_review_decisions was recorded in schema_migrations from an earlier partial copy. Its table and old per-run revision constraint exist, but the current migration's global revision identity, trigger-identity index, append-only triggers, and old-worker invoice fence do not. The normal migration runner correctly skips the recorded filename, so a restart cannot converge that state.

The attempted #2391 runtime cutover was rolled back to the known-good 58d6bc54 provider runtime before any tracker or Website consumer existed or any financial/Gmail action was made. This slice restores only the proven schema gap with a new forward-only migration, then permits a fresh provider deployment checkpoint.

The expected diff exceeds the 400-line soft cap because the indivisible proof must build the real migrations, mark 380 recorded, regress it to the observed legacy catalog shape, invoke the production migration runner, and prove both successful convergence and fail-closed preservation of ambiguous historical revision identities.

### Problem-derived contract

- Root cause: the runner tracks an applied migration by filename in schema_migrations and treats that filename as non-pending. Migration 380 was ledger-recorded before later edits added the global revision constraint, index, functions, and triggers. Current code correctly does not replay a recorded file, leaving the provider schema structurally partial.
- Correct fix must touch/change: add one atomic, forward-only migration that recognizes the recorded table, changes its legacy revision uniqueness only when existing historical rows can satisfy the global identity, installs the missing identity index/functions/triggers, and records itself only with that complete DDL. Enroll it in the existing invoicing workflow and prove the real runner's recorded-380 recovery and idempotent rerun in isolated PostgreSQL.
- Must not change: do not edit 380 or its ledger row; do not delete/rewrite review history; do not change payments, allocations, invoices, checks, Gmail, PDFs, Square, customer records, API routes, authorization, tracker, Website, or the generic migration runner. Do not deploy a consumer until the repaired provider catalog is verified.

## Scope (this PR)

Ownership lane: eom/commercial-billing-candidate-exclusions
Slice phase: Production hardening
Max files: 6

1. Add migration 381 to converge only the observed legacy recorded-380 schema into the current durable review-decision enforcement contract.
2. Enroll migration 381 in both existing invoicing-workflow path filters and add isolated PostgreSQL recovery tests that use the production migration runner and ledger semantics.
3. Archive #2391's merged plan and refresh the plan index as required post-merge housekeeping; it does not change product behavior.

### Review Contract

- Acceptance criteria:
  1. Given a schema ledger that records 380 and has its observed legacy UNIQUE (billing_run_id, candidate_key, source_fingerprint, revision) constraint but lacks later safety objects, the real runner applies 381 once, leaves 380 and every non-conflicting pre-existing decision row unchanged, and installs the global revision constraint, idx_commercial_billing_run_candidates_identity, both append-only triggers, and the invoice-writer trigger. Settled by the isolated-schema recovery proof in tests/test_commercial_billing_runs.py.
  2. After successful recovery, the existing service can append one decision and direct update/delete/truncate attempts are rejected; no invoice is created by that proof. Settled by the same isolated PostgreSQL test and the existing service path it invokes.
  3. If an observed legacy table contains duplicate global (candidate_key, source_fingerprint, revision) histories that its old per-run constraint permits, 381 raises before recording itself or replacing the old constraint. Settled by the explicit duplicate-history isolated PostgreSQL case; no silent historical rewrite is allowed.
  4. A fresh/current 380 schema and the recovered legacy schema both tolerate rerunning the real migration runner without another ledger record or schema/fact change. Settled by the same recovery test's second invocation and catalog/row assertions.
  5. A pull request or push changing migration 381 reaches the existing invoicing test workflow. Settled by .github/workflows/atlas_invoicing_checks.yml and its static enrollment assertion in tests/test_commercial_billing_runs.py.
- Reachability proof: atlas_brain.main:app invokes the production migration runner at lifespan startup. The test calls that same run_migrations boundary and observes catalog objects plus service/trigger behavior. After merge, deployment verification inspects only the local authenticated service OpenAPI and database catalog; it does not call a financial write route.
- Affected surfaces: migration execution/ledger bookkeeping, the commercial billing review-decision schema contract, the existing service's database guard, invoicing workflow path enrollment, and merged-plan archive metadata.
- Risk areas: historical review evidence, idempotent migration replay, global identity uniqueness, old-worker invoice fencing, schema migration ordering, provider rollback, and accidental financial/delivery side effects.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R7, R8, R10, R12, R14.

### Boundary-change enumeration

N/A - no HTTP input/admission boundary changes. The closed database writer guard is restored as existing contract enforcement; migration 381 does not change its vocabulary, input shape, or route behavior.

### Deployed-config probing

N/A - no configuration or environment-default change. The deployment check uses the existing ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true service and read-only catalog probes; it adds no fallback or credential.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/storage/migrations/381_commercial_billing_candidate_review_decisions_recovery.sql`
- `plans/INDEX.md`
- `plans/PR-EOM-Commercial-Billing-Candidate-Review-Migration-Recovery.md`
- `plans/archive/PR-EOM-Commercial-Billing-Candidate-Exclusions.md`
- `tests/test_commercial_billing_runs.py`

## Mechanism

Migration 381 starts with the runner's atomic-bookkeeping marker, so its DDL and ledger record commit together. It reads the named revision-key constraint from PostgreSQL's catalog. If it is already the current global identity, no uniqueness DDL changes. If it is legacy or missing, a catalog-safe preflight rejects duplicate global histories before removing the old named constraint and adding UNIQUE (candidate_key, source_fingerprint, revision).

The migration then creates the retained-candidate identity index, restores the two review-history triggers and their function, and restores the old-worker commercial-invoice trigger/function. Each DDL statement is idempotent for a fresh/current 380 schema, while the migration itself is exactly once according to the existing filename ledger. It contains no DML against payments, invoices, decisions, or delivery records.

The recovery test builds current 370/372/380 state through the real runner, records 380, removes only the later objects and reintroduces the production legacy constraint, then calls the real runner for 381. It exercises an actual decision service write after recovery, direct append-only violations, rerun behavior, and the duplicate-history safe failure. It does not need a real customer or provider endpoint.

## Intentional

- Migration 381 is forward-only rather than a rewrite/replay of 380. The filename ledger is behaving as designed; rewriting a recorded migration would not repair production and would make histories harder to audit.
- A duplicate global revision history fails closed. The repair does not choose which audited event to renumber, delete, or retain; that is a separate financial-correction decision if such data ever exists.
- The generic runner is unchanged. Its filename-only compatibility model has broad cross-product reach and is tracked as H-18 instead of being mixed into this evidence-specific provider repair.
- #2391's plan archive is included only because it is mandatory post-merge housekeeping; the archive preserves the exact merged plan and has no runtime effect.

## Deferred

Parking predicate: park generic migration-framework hardening, unobserved partial-schema shapes, and any review-decision product behavior unless it is required to converge the proven production catalog safely.

- [#2363 H-18](https://github.com/canfieldjuan/ATLAS/issues/2363) owns the generic immutable-applied-migration content/checksum policy. This slice adds no generic runner behavior.
- A policy for reconciling any future duplicate global review-revision history is deferred until a real catalog probe finds one. This migration safely stops instead of editing evidence.

Parked hardening: none.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:55432/atlas_receivables_test' pytest -q tests/test_commercial_billing_runs.py -k 'recorded_380_recovery or review_decision_recovery_migration_is_atomic'` — 3 passed after seeding and preserving a non-conflicting legacy decision row.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:55432/atlas_receivables_test' pytest -q tests/test_commercial_billing_runs.py` — 44 passed after review remediation (one existing torch/pynvml warning).
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL='postgresql://postgres:postgres@127.0.0.1:55432/atlas_receivables_test' python -m pytest tests/test_eom_render_profile.py tests/test_receivables.py tests/test_eom_billing_recipients.py tests/test_eom_payment_receipts.py tests/test_residential_payment_receipt_delivery.py tests/test_commercial_billing_candidates.py tests/test_commercial_billing_runs.py tests/test_commercial_billing_approvals.py tests/test_commercial_billing_gmail_drafts.py tests/test_commercial_billing_manual_square_invoices.py tests/test_invoice_repository.py tests/test_invoice_pdf.py -q` — 673 passed after review remediation (one existing torch/pynvml warning), using only the disposable local PostgreSQL database.
- `python -m pytest tests/test_monthly_invoice_generation.py -k 'update_invoice_clears_needs_hours_when_line_items_are_billable or line_items_are_billable_requires_all_positive_quantities' -q` — 2 passed after review remediation; `python -m pytest tests/test_invoicing_readonly_mcp.py tests/test_invoicing_readonly_oauth.py tests/test_invoicing_draft_writer_mcp.py tests/test_invoicing_draft_writer_oauth.py -q` — 43 passed.
- `python -m ruff check tests/test_commercial_billing_runs.py` and `python -m py_compile tests/test_commercial_billing_runs.py` — passed after review remediation. Plan synchronization, committed-diff plan audits, and the managed `scripts/push_pr.sh` full local gate are rerun before the updated head is pushed.
- After merge/deploy: verify that migration 381 is ledger-recorded, the four required catalog objects exist, the mounted route remains authorization-gated, and the atlas-api runtime points at the merged SHA. No financial/Gmail operation will be used as a deployment probe.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/storage/migrations/381_commercial_billing_candidate_review_decisions_recovery.sql` | 158 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Commercial-Billing-Candidate-Review-Migration-Recovery.md` | 100 |
| `plans/archive/PR-EOM-Commercial-Billing-Candidate-Exclusions.md` | 0 |
| `tests/test_commercial_billing_runs.py` | 427 |
| **Total** | **690** |
