# EOM first-clean completion schema runbook

## Scope

Migration `394_eom_first_clean_completion_receipts` creates the immutable,
guard-owned ATLAS evidence for an operator-confirmed first residential clean.
It does not send an email, create a public token, contact Stripe, or confirm a
customer appointment.

The ordinary Atlas runtime must not apply this migration. It cannot safely
create a foreign key to the guard-owned customer-handoff table, and it must not
own the append-only receipt tables. Use the controlled DBA entrypoint below.

## Prerequisites

1. Deploy the Atlas code containing the completion route and schema readiness
   fence. Before migration 394 is applied, the route must remain safely
   unavailable rather than accepting a completion report.
2. Confirm the canonical EOM funnel database already contains `contacts`,
   `eom_lead_lifecycle_events`, and `eom_customer_handoffs`. Migration 394
   verifies these prerequisites before creating any receipt evidence.
3. Inject a short-lived, protected PostgreSQL superuser DSN into
   `ATLAS_EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL`. Do not put that DSN in
   a command line, browser configuration, source file, or application runtime
   environment.

## Apply

Run the read-only preflight first:

```bash
python scripts/apply_eom_first_clean_completion_schema.py --json
```

The result redacts credentials and reports only the target host/database label,
executor status, named migration, and whether it is already recorded. Confirm
the executor is a superuser and the target is the intended canonical EOM
database.

Then apply only migration 394:

```bash
python scripts/apply_eom_first_clean_completion_schema.py --apply --json
```

The migration atomically records its ledger row, transfers the two receipt
tables and their trigger functions to `atlas_eom_handoff_owner`, revokes direct
runtime/NocoDB guard membership, rejects any inherited guard path, and grants
the Atlas runtime only `SELECT`, `INSERT`, and `UPDATE` needed for row locking
and receipt creation. It does not grant `DELETE`, `TRUNCATE`, `REFERENCES`,
`TRIGGER`, ownership, or customer delivery authority.

Remove the temporary DBA DSN injection after the result reports
`"migration_recorded": true`.

## Verify safely

1. Restart or deploy the normal EOM profile without the DBA DSN. It must not
   attempt to re-run migration 394 during startup.
2. Confirm the service-authenticated completion capability is available only
   when the schema readiness check succeeds. Do not post a test completion for
   a real customer or use a production customer/service identity as a probe.
3. Use the isolated PostgreSQL CI evidence for role ownership, minimal runtime
   ACLs, immutability, idempotency, and concurrency. No test sends customer
   communication or creates an appointment.

## Recovery

If the DBA preflight or apply fails, leave the route unavailable and correct
the database prerequisites or guard-role configuration before retrying. The
normal runtime must not be granted temporary guard membership or `REFERENCES`
as a workaround.

If application code must be rolled back after a successful apply, remove or
disable the completion consumer/route first and retain the append-only receipt
and migration evidence. There is intentionally no destructive down migration.
