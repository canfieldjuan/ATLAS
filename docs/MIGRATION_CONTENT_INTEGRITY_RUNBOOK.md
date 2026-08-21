# Migration Content Integrity Admission Runbook

## Purpose

This runbook governs Atlas releases that may apply SQL migrations. The normal
runner records source identity for newly applied files and reports historical
drift. Its admission policy refuses to execute a pending migration when a
recorded source mismatch or missing packaged source remains unexplained.

The policy is deliberately narrower than source verification. A reviewed
historical reconciliation can make its exact mismatch admissible only while its
current evidence is attested. It does not make unavailable historical source
bytes verified, and the read-only preflight continues to report that mismatch.

## Before a migration-bearing rollout

1. Start from the exact Atlas release-candidate worktree and identify the
   configured, log-safe database target without opening a connection:

   ```bash
   python scripts/check_migration_content_integrity.py --show-target
   ```

2. Compare that label with the target you intend to inspect. Then run the
   target-confirmed, read-only receipt against that same target:

   ```bash
   python scripts/check_migration_content_integrity.py \
     --expected-target '<exact target label from step 1>' \
     --attest-known-reconciliations
   ```

   The command never runs migrations or writes financial data. Preserve its
   JSON output with the release record. Do not pass a DSN, credential, or
   database URL on the command line.

3. Interpret the result conservatively:

   - `verified` and `legacy_unverified` can proceed under the normal runner
     policy. Legacy null hashes remain visible; they are not silently repaired.
   - `unresolved_drift` with a matching known record may be admissible only when
     every entry in `known_reconciliation_evidence` for that reported name is
     `attested`, there is no `missing_source`, and there are no additional
     mismatched names. The preflight still exits 2 to preserve forensic truth;
     do not treat exit 2 alone as approval.
   - Any unknown mismatch, missing source, a known record that is currently
     reported as mismatched but has absent/non-attested evidence, target
     mismatch, or `could_not_determine` result is a stop condition. Do not
     probe a known reconciliation that is not an active mismatch as a reason to
     stop an otherwise clean target.

4. Deploy the admission-policy release to every Atlas process that can invoke
   `run_migrations()` before adding a later migration-bearing release. This
   policy release contains no SQL migration itself. Do not use a standalone
   MCP process or an older checkout as an alternate migration path.

5. For the later migration-bearing release, use the ordinary service deployment
   entrypoint. The runner holds its existing advisory lock, evaluates the
   evidence under that lock, and either applies the selected pending files or
   fails before their SQL and `schema_migrations` records are written. Capture
   the startup/migration log and repeat the read-only receipt after the rollout.

## When admission refuses a pending migration

1. Stop the migration-bearing rollout. Do not edit `schema_migrations`, replace
   packaged historical source bytes, replay migration 387, or downgrade solely
   to make the runner apply SQL.
2. Preserve the target-confirmed preflight JSON, the release SHA, the blocked
   migration names, and the runner error. These are the recovery evidence.
3. Classify the evidence:

   - A known reconciliation that is `not_attested` means its current ledger,
     package, timestamp, or catalog predicate no longer matches the reviewed
     record. Repair the real schema/data condition through a separately
     reviewed, forward-only recovery; then rerun the preflight and retry the
     unchanged pending migration.
   - An unknown mismatch or missing source is not an allowlist request. Open a
     dedicated H-18 follow-up with immutable source evidence and a named
     catalog predicate. If historical source bytes cannot be proven, keep the
     migration blocked rather than fabricating a digest.

4. Retry only after the read-only receipt is current for the intended target.
   A failed admission executes no pending SQL, so the existing runner's
   advisory-lock and ledger behavior keeps the retry idempotent.

## Rollback

This policy adds no database migration and changes no financial records. A
runtime rollback can return a no-pending service to its prior application
revision, but it is never an approved way to bypass a refused pending migration.
Keep the policy-capable revision deployed while a migration-bearing release is
pending or under investigation.

If another release already applied SQL before an application rollback, retain
the recorded schema and ledger evidence. Use a separate, forward-compatible
repair or an explicitly authorized database rollback plan for that other
release; do not delete migration history as part of this policy rollback.

## Non-negotiable constraints

- ATLAS remains the financial source of truth; this runbook never authorizes
  invoice, payment, allocation, Gmail, or Square mutations for verification.
- The preflight is read-only evidence, not a migration executor.
- A known reconciliation is admission evidence for one exact migration name,
  not a global exception and not historical-source verification.
- Every new historical discrepancy requires its own reviewed evidence record;
  do not extend a generic runner allowlist.
