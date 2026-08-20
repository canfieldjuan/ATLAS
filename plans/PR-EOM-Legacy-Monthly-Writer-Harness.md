# PR-EOM-Legacy-Monthly-Writer-Harness

## Why this slice exists

H-23 in [ATLAS #2363](https://github.com/canfieldjuan/ATLAS/issues/2363)
was discovered while H-21 made the legacy monthly writer explicit opt-in. The
older `tests/test_monthly_invoice_generation.py` calls `init_database()` and
can reach the inherited `ATLAS_DB_*` target, while its admitted writer paths
can call calendar, CRM, PDF, email, and notification collaborators. That makes
those historical tests unsuitable as a safe proof of the enabled legacy writer.

This production-hardening slice unblocks a truthful regression proof for the
legacy writer without enabling it in production or changing its behavior. It is
justified by a money-safety risk: a regression in invoice creation, dedup, or
review-mode delivery could otherwise only be checked against ambient data or
not checked at all.

### Problem-derived contract

- Root cause: there is no explicitly armed, fail-closed integration harness
  that invokes `monthly_invoice_generation.run` with real service/invoice
  repositories against a disposable database namespace while blocking all
  delivery boundaries. The legacy test module's ambient database initialization
  cannot establish that property.
- Correct fix must touch/change: add one isolated-schema writer-harness test
  module and enroll it in `atlas_invoicing_checks.yml`. The harness must reject
  an unarmed or non-loopback/non-test-database target before `asyncpg.connect`,
  create/drop a UUID schema, install only the invoice/service DDL it needs,
  invoke the real task and real repositories, and replace only calendar, CRM,
  PDF rendering, email, and notification outer seams.
- Must not change: `monthly_invoice_generation` production logic, scheduler
  registration, settings defaults, migrations, invoices outside the disposable
  schema, Gmail/email/ntfy delivery, PDFs outside `tmp_path`, customer data,
  product copy, and the new commercial billing-run workflow.

## Scope (this PR)

Ownership lane: eom/billing-payments-security-hardening
Slice phase: Production hardening

1. Add an opt-in integration harness that makes a deterministic service and
   calendar fixture flow through the real legacy monthly task, customer-service
   repository, and invoice repository in a UUID-named schema.
2. Prove one review-mode run writes one draft invoice, its PDF test artifact,
   and the service's invoiced markers; prove a second identical run deduplicates
   without creating another invoice or delivery action.
3. Prove unsafe/unarmed targets cannot open a database connection and the
   disposable schema is dropped both after a successful writer run and after a
   synthetic failure inside the harness context.
4. Enroll the test under a dedicated, explicitly armed local-Postgres workflow
   job; include the test path in both pull-request and main-push triggers.

### Review Contract

- Acceptance criteria:
  1. `tests/test_legacy_monthly_autoinvoice_writer_harness.py` validates its
     explicit opt-in and accepts only the exact loopback
     `atlas_receivables_test` PostgreSQL target before any `asyncpg.connect`;
     its pure rejection cases settle malformed, remote, non-test, and
     non-PostgreSQL targets.
  2. The same test module's real `monthly_invoice_generation.run` fixture uses
     `CustomerServiceRepository` and `InvoiceRepository` over its UUID schema,
     and assertions settle one draft invoice, exact source-ref deduplication,
     the expected service lifecycle markers, and no sent status.
  3. The delivery capture in
     `tests/test_legacy_monthly_autoinvoice_writer_harness.py` proves review
     mode never invokes the email-provider factory while calendar, CRM, PDF,
     and notification test seams supply deterministic local behavior.
  4. The test's observer-connection assertions prove its UUID schema is absent
     after both normal completion and a synthetic context-body failure.
  5. `.github/workflows/atlas_invoicing_checks.yml` provisions a local
     PostgreSQL service for the dedicated harness job, explicitly sets the two
     harness-only environment variables, and runs this test file.
- Reachability proof: the real async entrypoint is
  `atlas_brain.autonomous.tasks.monthly_invoice_generation.run`; observable
  effects are persisted invoice/service rows in the disposable schema, a
  `tmp_path` PDF, captured local notification/CRM calls, the email-factory
  sentinel remaining untouched, and post-teardown catalog absence.
- Affected surfaces: a new test-only isolated database harness and the
  invoicing workflow's test enrollment. No runtime, API, customer-facing, or
  migration surface is affected.
- Risk areas: accidental ambient/production database connection; unscoped DDL
  cleanup; legacy writer duplication; draft-versus-sent state; accidental
  external delivery; CI test enrollment; fixture non-determinism.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  security/authorization, R4 data safety, R5 compatibility, R6 error/cleanup,
  R8 idempotency, R10 maintainability, R11 configuration/dependencies, R12
  deployment/CI, and R14 codebase verification.

### Boundary-change enumeration

- Boundary path/seam: test-only `_require_writer_harness_database_url()` is the
  sole database-admission choke point; it runs before the harness imports or
  calls `asyncpg.connect`.
- Replaced-path behaviors: none. The production task and production database
  initialization path are not modified or called.
- Guard-relevant fields: the exact opt-in marker and the harness-only database
  URL's scheme, host, port, database name, query, and fragment.
- Caller x input shape: the dedicated workflow job supplies the armed marker
  and loopback test URL; an ordinary local pytest invocation is skipped; direct
  pure-validator tests cover unarmed/unsafe strings without a connection. A
  grammar-derived cross-product test covers the URL component combinations,
  rather than a review-targeted URL fixture list.

### Guard class-closure declaration

- Input class: OPEN. The harness-only URL environment value is arbitrary text,
  but the database-admission verdict is a strict allowlist-shaped grammar.
- Closed dependency: `_LOOPBACK_HOSTS` is CLOSED and canonical in the new test
  module: `127.0.0.1` and `::1` are the only accepted host forms for this
  deliberately local CI harness. The exact PostgreSQL scheme, port `5432`,
  database path `/atlas_receivables_test`, and empty query/fragment are likewise
  fixed grammar terminals, not a product inventory.
- Out-of-set behavior: every malformed, remote, non-PostgreSQL, non-test,
  non-default-port, query-bearing, or fragment-bearing value raises before any
  `asyncpg` import or connection; a missing/nonexact opt-in marker skips before
  the URL is consulted. Failing closed is cheaper than even an attempted
  connection to an ambient financial database.
- Evidence gate: the new grammar-derived cross-product test generates terminal
  combinations and asserts only the exact grammar admits. It is paired with an
  async probe that substitutes a forbidden `pytest.importorskip` and proves an
  unsafe URL cannot reach the connection-import step.

### Deployed-config probing

N/A - this PR changes no deployed configuration, default, resolver, or runtime
admission boundary. The two environment variables are test-only and exist only
inside the dedicated CI job or an intentionally armed local command.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `plans/PR-EOM-Legacy-Monthly-Writer-Harness.md`
- `tests/test_legacy_monthly_autoinvoice_writer_harness.py`

## Mechanism

The harness is inactive unless its dedicated opt-in marker is exactly enabled.
Before opening a connection, it parses the harness URL and permits only a
plain loopback PostgreSQL URL for the explicitly named receivables test
database. It then opens one `asyncpg` connection, creates a randomly named
schema, sets that schema first in the search path, creates the minimal
`contacts` anchor, and executes migrations 045, 047, and 048 inside that
schema. `finally` restores the search path, drops only the generated schema,
and closes the connection.

The task receives real repository instances backed by a schema-scoped pool.
Calendar and CRM are deterministic in-memory adapters; PDF rendering returns
test bytes into `tmp_path`; the notification function records locally; and the
email-provider factory is a sentinel that fails if review mode ever tries to
use it. A single active per-visit service and two same-day events create a
deterministic draft. Running the unchanged task again proves its existing
`source_ref` lookup reuses that invoice rather than writing another one.

The workflow gives this test its own PostgreSQL service and harness-only
environment variables, so it cannot inherit `ATLAS_DB_*` or run merely because
someone executes a broad local test command.

## Intentional

- The harness writes real invoice/service rows only inside its UUID schema;
  faking those repositories would not prove the money-writing path.
- Review mode remains enabled even though the send flag is true, specifically
  to prove the legacy task's draft/no-email behavior rather than exercise real
  email delivery.
- The PDF renderer is an outer seam and returns a deterministic byte string;
  invoice-PDF rendering quality remains covered by its existing focused tests.
- This does not rehabilitate or expand the historical ambient-database test
  module. It creates a separate, explicitly safe writer proof.
- A dedicated workflow job is intentionally duplicated rather than sharing an
  ambient test database contract; that makes the writer proof's opt-in and
  target visible at the CI boundary.

## Deferred

- Legacy task financial behavior (float normalization, billing-run cross-dedup,
  actual production enablement, scheduler retirement, and delivery semantics)
  remains outside this test-only harness and stays tracked by #2363 / #2362.
- A reusable repository-wide disposable-schema test library is deferred: it
  would be new test infrastructure beyond this one writer's safety proof.

Parking predicate: harness abstractions, coverage for unrelated legacy task
branches, and cross-product test infrastructure are parked unless they are
necessary to prove this harness cannot use an ambient database or delivery
system.

Parked hardening: none.

## Verification

- `env -u ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_HARNESS -u
  ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_TEST_DATABASE_URL
  /home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest
  tests/test_legacy_monthly_autoinvoice_writer_harness.py -q` — `3 passed, 2
  skipped`; ordinary local invocation cannot open the harness database.
- `ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_HARNESS=1
  ATLAS_LEGACY_MONTHLY_AUTOINVOICE_WRITER_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/atlas_receivables_test
  /home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest
  tests/test_legacy_monthly_autoinvoice_writer_harness.py -q` — `5 passed` on
  a disposable local PostgreSQL 16 container. The post-run catalog query found
  `0` `legacy_monthly_writer_*` schemas and the owned container was removed.
- Ruff check and Ruff format --check on
  `tests/test_legacy_monthly_autoinvoice_writer_harness.py` — passed.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -m pytest
  tests/test_legacy_monthly_autoinvoice_opt_in.py
  tests/test_monthly_invoice_generation.py -k
  'legacy_monthly_autoinvoice_opt_in or
  update_invoice_clears_needs_hours_when_line_items_are_billable or
  line_items_are_billable_requires_all_positive_quantities' -q` — `13 passed,
  41 deselected`.
- `/home/juan-canfield/Desktop/Atlas/.venv/bin/python -c 'from pathlib import
  Path; import yaml; yaml.safe_load(Path(".github/workflows/atlas_invoicing_checks.yml").read_text(encoding="utf-8"));
  print("workflow_yaml=valid")'` — `workflow_yaml=valid`.
- `python scripts/sync_pr_plan.py`, plan-doc/files-touched/diff-size audits,
  plan/code-consistency audit, strict guard-closure lint, and `git diff --check
  origin/main...HEAD` — passed.
- Pending before push: the managed local PR review bundle.
- Hosted CI is supplemental evidence; the required implementation proof is
  runnable locally against the exact loopback test target and cannot mutate a
  production namespace.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 44 |
| `plans/PR-EOM-Legacy-Monthly-Writer-Harness.md` | 229 |
| `tests/test_legacy_monthly_autoinvoice_writer_harness.py` | 458 |
| **Total** | **731** |
