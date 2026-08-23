# PR-H18-Company-Signal-Promotion-Source-Attestation

## Why this slice exists

The canonical target-confirmed H-18 preflight for [#2476](https://github.com/canfieldjuan/ATLAS/issues/2476) continues to refuse pending migration 389 because `297_b2b_company_signal_canonical_promotion_type` is a NULL-digest ledger record with no packaged source. This slice is the next independently blocking receipt after #2481: it does not make the generic integrity report green, but it can let the shared runner distinguish this exact, structurally proven historical record from unknown source gaps.

Read-only target evidence at `host=localhost, port=5433, db=atlas` establishes exactly one version-297, NULL-digest ledger row at `2026-04-12T19:28:13.742305Z`; its `b2b_company_signals` relation is a permanent ordinary table with a nullable plain-text `canonical_promotion_type` column and the ready named partial index. Retained history proves a numeric-prefix collision rather than a rename: the current `297_b2b_review_vendor_mentions.sql` was introduced after that target receipt, and neither reachable history nor the 5,559 scanned unreachable blobs contains the missing source path or its schema token outside a planning artifact.

Diff-budget override: the immutable record, closed catalog predicate, false-evidence matrix, real PostgreSQL runner reachability proof, and CI enrollment form one admission-safety claim. Splitting them would publish an unproven exception or omit the before-SQL failure proof.

### Problem-derived contract

- Root cause: an historical numeric-prefix collision left the canonical target with a separately named, version-297 NULL-digest ledger receipt whose original source is unavailable. The generic content-integrity admission gate correctly blocks pending SQL rather than guessing that the later, unrelated packaged `297_b2b_review_vendor_mentions.sql` is the source.
- Correct fix must touch/change: add one source-controlled, immutable reconciliation record for that exact name/version/timestamp; read only its final catalog metadata; dispatch only that name through the existing candidate-derived reconciliation seam; and prove success, failure, no-write, and retry behavior with focused fake and disposable PostgreSQL tests.
- Must not change: packaged migration SQL, `schema_migrations`, target schema/data, the generic content-integrity classifier, other H-18 records (especially 386 and 379), customer/lead/payment/email behavior, configuration, deployment, and active commercial-billing/booking lanes.

## Scope (this PR)

Ownership lane: eom/migration-content-integrity
Slice phase: Production hardening
Max files: 7

1. Add a closed, named source-unavailable receipt for only `297_b2b_company_signal_canonical_promotion_type`.
2. Admit pending test SQL only when the exact ledger receipt and the complete narrow catalog predicate attest; keep the generic report unchanged and every other discrepancy blocking.
3. Add PII-free fake and real PostgreSQL evidence, enroll the real test in the existing migration workflow, and archive only the merged #2481 plan as required teardown.

### Review Contract

1. The closed registry contains the exact `297_b2b_company_signal_canonical_promotion_type` version-297 NULL-digest record at the observed aware-UTC timestamp, and no generic name mapping; settled by `tests/test_migration_content_integrity_preflight.py` record assertions.
2. The candidate's read-only catalog evidence requires a permanent ordinary `b2b_company_signals` relation, a nullable non-generated/non-identity/default-collation `TEXT` column with no default or constraint, and the exact ready nonunique partial index; settled by focused fake preflight cases and the disposable PostgreSQL test.
3. Missing/duplicate/wrong-version/non-NULL/wrong-timestamp ledger evidence, a nonordinary/nonpermanent relation, an altered column, a column constraint, or an absent/altered/unready index returns `not_attested` before pending SQL; settled by the false-evidence matrix and real runner failure proof.
4. The generic preflight continues to emit `missing_source` / exit 2 while exposing boolean-only receipt evidence; no catalog query reads company-signal rows and the fake connection observes zero writes.
5. The shared `run_migrations` choke point applies one disposable pending migration and records it once only after the exact 297 receipt attests; an incomplete receipt leaves its table and ledger row absent, and retry after repair applies it exactly once; settled by `tests/test_b2b_company_signal_promotion_migration_repair.py`.
6. Existing mismatched/missing records remain independently blocking, including 386 and 379; settled by the existing generic runner behavior plus the named candidate-only tests.
7. Only the merged #2481 plan is moved to `plans/archive/` and the plan index is regenerated; settled by `scripts/archive_plans.py index` and the plan checks.
- Reachability proof: `run_migration_content_integrity_preflight` remains the read-only CLI-backed entrypoint; `run_migrations` is the shared pending-SQL admission choke point. The disposable PostgreSQL test observes no pending table/ledger row on failed attestation and exactly one of each after an eligible retry.
- Affected surfaces: source-controlled reconciliation registry, read-only PostgreSQL catalog metadata, shared migration admission, migration-runner workflow enrollment, and H-18 plan archival. No public API or product surface changes.
- Risk areas: false historical admission, source/target identity confusion, schema drift, pre-SQL failure ordering, PII exposure, test enrollment, and conflict with active billing work.
- Reviewer rules triggered: R1, R2, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: `_unresolved_pending_migration_content_evidence` already derives candidates from the closed reconciliation registry; this slice adds one named 297 attestation branch without another allowlist.
- Replaced-path behaviors: only an attested 297 source absence can move from refusal to pending-SQL admission. Unknown names, incomplete catalog evidence, transport failure, or coexisting discrepancies retain refusal before SQL.
- Guard-relevant fields: record name, version, NULL digest, exact UTC timestamp, relation kind/partition/persistence, column type/nullability/default/generated/identity/collation/constraint state, and index name/key/uniqueness/readiness/predicate.
- Caller x input shape: the CLI and `run_migrations` pass report-derived candidate names to the same dispatcher; operator input cannot select this receipt directly.

### Deployed-config probing

- Deployed/default config values: no configuration is added or changed; the existing typed `ATLAS_DB_*` settings supply the read-only target only.
- Explicit value probe: `python scripts/check_migration_content_integrity.py --show-target` reported `host=localhost, port=5433, db=atlas`; the target-confirmed read-only preflight observed the exact 297 receipt and catalog metadata.
- Absent value probe: disabled persistence, a target-label mismatch, or transport failure remains `could_not_determine` / admission refusal before SQL; no fallback target is introduced.
- Default-session/default-context probe: focused fake preflight proof asserts `readonly=True`, no `execute` calls, and no table-row query; the real test uses a disposable URL only.
- Side-effect ordering: integrity report and named attestation happen before pending migration selection; failure leaves pending SQL and its `schema_migrations` row absent.

### Files touched

- `.github/workflows/atlas_migrations_runner_checks.yml`
- `atlas_brain/storage/migrations/reconciliation.py`
- `plans/INDEX.md`
- `plans/PR-H18-Company-Signal-Promotion-Source-Attestation.md`
- `plans/archive/PR-H18-Watchlist-Alert-Events-Source-Attestation.md`
- `tests/test_b2b_company_signal_promotion_migration_repair.py`
- `tests/test_migration_content_integrity_preflight.py`

## Mechanism

`HistoricalVersionedMissingSourceReconciliation` retains the immutable identity for 297. A single read-only catalog query derives only booleans for the exact target relation, column, constraints, and named partial index; it never reads `b2b_company_signals` rows. The dispatcher returns this attestation only when the generic report actually contains that name, so no caller can turn it into a broad historical exception.

The same existing runner subtracts only an attested candidate from its original `missing_source` set. The generic diagnostic report stays honest and returns exit 2. A dedicated PostgreSQL 16 test owns its schema, uses a synthetic pending migration, proves both failed no-write admission and repaired retry success, and is enrolled in the existing migration workflow.

## Intentional

- The later packaged `297_b2b_review_vendor_mentions.sql` is not mapped to the old 297 receipt: it was committed after the target row and creates a different table.
- The receipt remains `historical_source_unavailable`; target evidence does not reconstruct or verify the original source bytes.
- The predicate does not freeze unrelated later `b2b_company_signals` columns or indexes. It closes only the named column/index contract evidenced by this historical receipt.
- No historical SQL is replayed, no `schema_migrations` row is rewritten, and no target relation is repaired by this PR.
- No migration-runner locking redesign is folded into this receipt; #2476 retains the external-admin serialization follow-up.

## Deferred

- #2476 retains the real migration-386 recovery design, the independently missing 379 billing-review receipt, and the global external migration-evidence serialization design.
- Migration 389 and the missed-call recovery provider deployment remain blocked until all remaining H-18 records independently attest and the normal deployment checkpoint succeeds.
- Any new B2B company-signal schema hardening, future writer behavior, or evidence for another historical migration is a separate receipt slice.

Parking predicate: every other historical record, global runner execution-model change, target schema/data mutation, deployment/configuration change, and active commercial-billing or booking lane is parked unless it is necessary to prove this exact 297 receipt.

Parked hardening: external migration-evidence serialization, tracked in #2476.

## Verification

- Pending before push: focused fake preflight matrix; dedicated disposable PostgreSQL proof when configured; syntax/lint/diff/plan gates; target-confirmed read-only preflight; and GitHub migration/required checks. The full Unit Gate remains GitHub-only.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_migrations_runner_checks.yml` | 3 |
| `atlas_brain/storage/migrations/reconciliation.py` | 320 |
| `plans/INDEX.md` | 3 |
| `plans/PR-H18-Company-Signal-Promotion-Source-Attestation.md` | 105 |
| `plans/archive/PR-H18-Watchlist-Alert-Events-Source-Attestation.md` | 0 |
| `tests/test_b2b_company_signal_promotion_migration_repair.py` | 297 |
| `tests/test_migration_content_integrity_preflight.py` | 292 |
| **Total** | **1020** |
