# EOM first-clean completion schema runbook

## Scope

Migration `394_eom_first_clean_completion_receipts` creates the immutable,
guard-owned ATLAS evidence for an operator-confirmed first residential clean.
It does not send an email, create a public token, contact Stripe, or confirm a
customer appointment.

The ordinary Atlas runtime must not apply this migration. It cannot safely
create a foreign key to the guard-owned customer-handoff table, and it must not
own the append-only receipt tables. Use the controlled DBA entrypoint below.
Generic Atlas startup and MCP migration runs intentionally skip migration 394;
only the dedicated runner's explicit selection may apply it.

## Prerequisites

1. Deploy the Atlas code containing the completion route and schema readiness
   fence. Before migration 394 is applied, the route must remain safely
   unavailable rather than accepting a completion report.
2. Confirm migration 354 has already moved `eom_customer_handoffs` plus
   `require_eom_customer_handoff_finalization()` and
   `prevent_eom_customer_handoff_mutation()` to
   `atlas_eom_handoff_owner`. Migration 363 must also have created the
   `eom_lead_lifecycle_events_sequence_seq` default owned by
   `eom_lead_lifecycle_events.lifecycle_sequence`. The canonical EOM funnel
   database must contain those objects plus `contacts`. Migration 394 verifies
   these catalog prerequisites before creating any receipt evidence.
3. Inject a short-lived, protected PostgreSQL superuser DSN into
   `ATLAS_EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL`. Do not put that DSN in
   a command line, browser configuration, source file, or application runtime
   environment. The controlled runner reads only this typed deploy-time setting;
   it does not accept a caller-selected environment-variable name.
4. Set `ATLAS_EOM_FIRST_CLEAN_COMPLETION_DBA_SCHEMA` to one ASCII PostgreSQL
   identifier. It must equal `current_schema()` for the configured
   `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`; the runner reads that runtime schema
   before it opens the DBA pool, pins every DBA pool connection to the declared
   schema, and fails before migration when either value is absent or differs.

## Apply

Run the read-only preflight first:

```bash
python scripts/apply_eom_first_clean_completion_schema.py --json
```

The result redacts credentials and reports only the target host/database/schema
label, executor status, named migration, and whether it is already recorded.
Confirm the executor is a superuser and the target is the intended canonical
EOM database and runtime schema.

Then apply only migration 394:

```bash
python scripts/apply_eom_first_clean_completion_schema.py --apply --json
```

The migration atomically records its ledger row, requires the pre-existing
handoff table and its protected functions to be guard-owned, transfers the
schema plus the two receipt tables, lifecycle table, lifecycle ordering
sequence, and their trigger functions to `atlas_eom_handoff_owner`, rejects any
direct or inherited guard path held by a non-superuser login, and grants the
Atlas runtime only schema `USAGE, CREATE`, table `SELECT, INSERT, UPDATE` for
row locking and receipt creation, plus sequence `USAGE` needed by the lifecycle
default. It does not grant `DELETE`, `TRUNCATE`, `REFERENCES`, `TRIGGER`,
sequence `SELECT` or `UPDATE`, schema ownership, or customer delivery authority.

Remove the temporary DBA DSN injection after the result reports
`"migration_recorded": true`.

## Verify safely

1. Restart or deploy the normal EOM profile without the DBA DSN. It must not
   attempt to re-run migration 394 during startup.
2. Confirm the service-authenticated completion capability is available only
   when the schema readiness check succeeds. Do not post a test completion for
   a real customer or use a production customer/service identity as a probe.
   The readiness fence also requires guard ownership of the canonical schema,
   canonical-handoff table and its protected functions, existing
   canonical-handoff finalization/append-only triggers, lifecycle append-only
   triggers, and the
   guard-owned lifecycle ordering sequence with its exact runtime `USAGE` ACL.
   It also refuses to serve if a non-superuser login can directly or indirectly
   assume the guard role; a missing, disabled, runtime-owned, or broadened
   prerequisite leaves the route unavailable.
3. Use the isolated PostgreSQL CI evidence for role ownership, minimal runtime
   ACLs, immutability, idempotency, and concurrency. No test sends customer
   communication or creates an appointment.

## Recovery

If the DBA preflight or apply fails, leave the route unavailable and correct
the database prerequisites or guard-role configuration before retrying. The
normal runtime must not be granted temporary guard membership or `REFERENCES`
as a workaround. If the configured and observed runtime schemas differ, correct
the typed schema configuration or the funnel DSN's deployment setting; do not
override `search_path` ad hoc on a command line.

If application code must be rolled back after a successful apply, remove or
disable the completion consumer/route first and retain the append-only receipt
and migration evidence. There is intentionally no destructive down migration.
