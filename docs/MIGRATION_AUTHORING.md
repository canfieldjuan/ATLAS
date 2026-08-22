# Migration SQL Authoring

## Purpose

Atlas normally records each newly applied migration in `schema_migrations` after
its SQL succeeds. Some historic and recovery migrations also write their own
ledger row. That pattern is safe only when the migration SQL, ledger row, and
source digest are committed as one transaction.

The migration runner selects that atomic-bookkeeping path automatically for a
direct, executable `INSERT INTO schema_migrations`. It also has one explicit
authoring opt-in for an equivalent write that is constructed dynamically and
cannot be recognized safely from static SQL source.

This guide is for new, forward-only migration authoring. Do not edit an
already-applied migration or its ledger row merely to add the marker.

## Choose the correct form

### Direct static ledger insert

When the migration contains a direct executable statement such as:

```sql
INSERT INTO schema_migrations (version, name)
VALUES (900, '900_example');
```

the runner recognizes it and uses atomic bookkeeping automatically. Do not add
the marker just for that direct form. The recognizer deliberately ignores
comments and quoted or dollar-quoted literals, so examples and rollback notes
do not change execution mode.

### Dynamic self-recording SQL

When a migration owns the equivalent ledger write through `EXECUTE`, `format`,
a procedure body, or another dynamic construct, place this exact line as the
**first non-empty line** of the file:

```sql
-- atlas: atomic-bookkeeping
```

It must precede every other non-empty comment and every SQL statement. A marker
below a header comment is not an opt-in. For example:

```sql
-- atlas: atomic-bookkeeping
DO $$
BEGIN
    EXECUTE 'INSERT INTO schema_migrations (version, name) VALUES (...)';
END $$;
```

The static recognizer intentionally masks quoted literals and dollar-quoted
bodies. It therefore cannot reliably infer this dynamic write, and the marker
is the safe, explicit declaration that the whole operation must use the
atomic-bookkeeping path.

## Transaction constraint

An atomic-bookkeeping migration cannot contain executable `CONCURRENTLY` DDL,
including `CREATE INDEX CONCURRENTLY`. PostgreSQL requires those statements to
run outside a transaction, while atomic bookkeeping requires one transaction.
Split the work into separately reviewed, forward-only migrations instead. A
comment or string literal that merely mentions `CONCURRENTLY` is not DDL.

## Required proof before review

For a new dynamic self-recording migration:

1. Add a focused test that writes the exact source into a temporary migration
   catalog and calls `run_migrations()` against a disposable PostgreSQL schema.
2. Prove the dynamic source is outside the direct static recognizer, then prove
   the exact marker-first source persists its ledger row and source digest.
3. Cover the boundary that a marker after another non-empty line does not opt
   in, and prove a failed dynamic ledger/digest update rolls back both the
   migration effects and its ledger row before an unchanged retry succeeds.
4. Let GitHub run the full migration-runner and database checks before merge.

Do not broaden the recognizer by scanning arbitrary literals or by adding a
regex for dynamic SQL in a migration PR. A new automatic classifier needs a
separate, evidence-backed design with explicit false-positive and
false-negative semantics.

## Recovery boundary

If a deployment fails, preserve the SQL source, runner error, target-confirmed
evidence, and recorded ledger state. Do not delete or rewrite migration history
to retry. Follow the [Migration Content Integrity Admission
Runbook](MIGRATION_CONTENT_INTEGRITY_RUNBOOK.md) for content-evidence admission
and use a separately reviewed, forward-only repair when needed.
