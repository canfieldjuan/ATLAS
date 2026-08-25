# PR-First-Clean-Completion-Receipt

## Why this slice exists

Issue #2156 requires the card-on-file/onboarding path to begin only after a
first residential cleaning actually completes. The current `first_clean_booked`
lifecycle event is created while a Calendar appointment is reconciled, making
it booking evidence rather than service-delivery evidence. The tracker has no
private ATLAS endpoint for an operator-confirmed first-clean completion.

Diff-budget override rationale: the additive database backstop, one atomic
provider transaction, its authenticated route, and concurrency/recovery proof
are one deploy-safe behavior. Splitting them would publish either schema that
cannot be safely exercised or a route without the durable invariants it claims.

### Problem-derived contract

- Root cause: the only current first-clean fact is booking reconciliation, so
  treating it as a post-clean trigger would create customer-facing work from a
  calendar record rather than completed service evidence.
- Correct fix must touch/change: add an immutable, tenant-scoped completion
  receipt and globally bound idempotency receipt; expose one authenticated EOM
  funnel route that validates the canonical customer/handoff and persists that
  fact atomically; bind that route to the slim profile's canonical funnel pool;
  apply its guard-owned schema only through a controlled DBA path; advertise it
  only when registered; and prove retry, conflict, migration, concurrent
  delivery, and normal-runtime non-escalation with isolated tests.
- Must not change: existing first-clean booking, onboarding welcome drafts,
  public onboarding tokens, customer-handoff semantics, calendar behavior,
  email delivery, Stripe/card collection, or any tracker/Website UI. This slice
  records no customer-facing side effect.

## Scope (this PR)

Ownership lane: eom/onboarding-first-clean-completion
Slice phase: vertical slice
Max files: 10

1. Add a durable ATLAS receipt for one authenticated completion of the first
   residential service for an already canonicalized EOM customer.
2. Add the private EOM funnel contract and capability manifest entry for a
   future tracker operator action, with replay/conflict/schema proof.

### Review Contract

- Acceptance criteria:
  - `POST /eom-funnel/customer-handoffs/{contact_id}/first-clean-completions`
    accepts only an authenticated, actor-attributed, RFC-3339 completion
    payload with a closed tracker service identity and a serialized lifecycle
    actor that fits the authoritative column; API tests prove malformed, naive,
    extra-field, future-time, and actor-overflow inputs do not create receipts.
  - `EOMFirstCleanCompletionService.record_completion` takes transaction-scoped
    locks, binds `Idempotency-Key` to a complete request fingerprint before the
    receipt write, and returns the original receipt for an unchanged retry;
    service tests prove sequential, same-key concurrent retries, and distinct
    concurrent operations for one customer leave one receipt.
  - The receipt schema requires active `effingham_maids` residential customer
    scope and a matching immutable `eom_customer_handoffs` customer/site link;
    integration tests prove cross-contact, commercial, inactive, missing, and
    mismatched-handoff requests fail without records.
  - A different source service or completion timestamp for an already completed
    customer fails closed instead of rewriting the receipt; tests prove the
    original receipt remains unchanged.
  - Migration `394_eom_first_clean_completion_receipts` requires a PostgreSQL
    superuser, creates the foreign-keyed receipt tables as a trusted no-login
    guard owner, revokes direct runtime/NocoDB guard membership, rejects an
    inherited guard path, and grants the Atlas runtime only `SELECT`, `INSERT`,
    and `UPDATE`; database tests prove that owner/ACL state and that a
    non-superuser executor is rejected before DDL.
  - The service requires the guarded receipt schema and prerequisite
    handoff/lifecycle integrity triggers before serving; route tests prove a
    missing, owner-mismatched, disabled, or append-only trigger becomes a safe
    `503` with no write. The slim EOM profile never applies migration 394 from
    its normal runtime connection.
  - The capability name is mechanically derived from the registered route;
    capability-manifest tests prove Atlas never advertises an absent route.
  - The existing PostgreSQL-backed EOM lead-pipeline workflow explicitly runs
    the persistence and DBA-runner tests, rather than treating local skipped
    database cases as proof.
- Reachability proof: the next tracker slice will call the authenticated funnel
  route after a manager explicitly confirms a completed first service. This PR
  proves the route reaches the receipt table, lifecycle ledger, and JSON
  response using an ASGI app and isolated PostgreSQL schema.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`, the slim EOM pool binding,
  a new completion service, a DBA-only migration/runner/runbook, the existing
  EOM PostgreSQL workflow, the capability manifest, and focused tests only.
- Risk areas: source identity collision, cross-customer handoff mismatch,
  stale/reused idempotency keys, runtime ownership/ACL escalation,
  migration-before-code deployment ordering, concurrent reporting, future
  timestamps, and accidental email/Stripe side effects.
- Reviewer rules triggered: R1, R2, R3, R4, R7, R8, R10, R11, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: private tracker-to-ATLAS completion report at
  `/eom-funnel/customer-handoffs/{contact_id}/first-clean-completions`.
- Replaced-path behaviors: none; `first-clean-bookings` remains booking-only.
- Guard-relevant fields: path contact UUID, Idempotency-Key, authenticated
  actor, tracker customer/site IDs, closed service kind, positive service ID,
  and timezone-aware completion timestamp.
- Caller x input shape: tracker service bearer plus actor headers may submit
  only `tracker_customer_id`, `tracker_site_id`, `tracker_service_kind`,
  `tracker_service_id`, and `completed_at`; browser credentials and arbitrary
  service-kind strings are not admitted.

### Deployed-config probing

- Deployed/default config values: existing EOM Funnel service authentication is
  required. The DBA-only apply command reads its protected DSN only from
  `ATLAS_EOM_FIRST_CLEAN_COMPLETION_DBA_DATABASE_URL`; it is not runtime or
  browser configuration.
- Explicit value probe: authenticated ASGI tests use a generated service token
  and actor headers.
- Absent value probe: disabled/missing service authentication retains existing
  fail-closed route behavior; a missing receipt schema returns `503`; a missing
  DBA DSN makes the controlled runner fail before a connection or migration.
- Default-session/default-context probe: no default customer or tenant is
  accepted; the contact and immutable handoff must resolve to
  `effingham_maids` in the transaction.
- Side-effect ordering: schema readiness and input admission occur before a
  transaction; operation-key binding occurs before lifecycle/receipt insertion;
  no external provider call exists in this slice.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_first_clean_completion.py`
- `atlas_brain/storage/migrations/394_eom_first_clean_completion_receipts.sql`
- `docs/EOM_FIRST_CLEAN_COMPLETION_RUNBOOK.md`
- `plans/PR-First-Clean-Completion-Receipt.md`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `tests/test_eom_first_clean_completion.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`

## Mechanism

The route is private and requires the existing EOM service bearer plus
authenticated operator headers. Its payload uses a closed pair of tracker
service kinds (`job` or `planned_visit`) and a positive stable ID so a future
tracker action can report either canonical service record without inventing a
parallel identity type.

`EOMFirstCleanCompletionService` normalizes the timestamp and authenticated
actor identity before fingerprinting every immutable request fact. It takes
sorted advisory locks for the contact, operation key, and tracker source
identity, and locks the canonical contact and handoff rows. It binds each
operation key globally before mutation. The service then requires an active EOM
`customer` with customer type `residential`, and verifies that the supplied
tracker customer/site pair equals the immutable handoff. It writes the
append-only lifecycle event and receipt in the same transaction, binding UUID
metadata as text at the SQL boundary. A repeat with identical facts returns the
original receipt; any conflicting key/source/contact/timestamp/actor fails with
`409`.

Migration 394 is a controlled DBA-only operation because it creates foreign
keys to the guarded handoff table and transfers its two receipt tables plus
trigger functions to `atlas_eom_handoff_owner`. It first requires migration
354's guarded handoff table and protected functions, revokes direct
runtime/NocoDB guard membership, rejects inherited membership, and grants Atlas
only `SELECT`, `INSERT`, and `UPDATE` for the row locks and writes the service
needs. The route refuses to serve if receipt ownership/ACLs/triggers or the
prerequisite handoff ownership/functions/triggers and lifecycle integrity
triggers are not exactly ready, so deploying code before the DBA apply is safe.
Actor validation derives the serialized lifecycle value before the transaction
and rejects every value that would overflow the lifecycle ledger column. The
normal slim EOM profile does not run migration 394. The runbook uses one
explicit named migration and a redacted protected-DSN preflight; rollback stops
the route/consumer while preserving audit evidence.

## Intentional

- Treat `first_clean_booked` as booking evidence only. It is intentionally not
  accepted as completion proof.
- Require an existing canonical customer handoff. A tracker record without a
  canonical contact/customer/site bridge cannot trigger customer onboarding.
- Do not infer completion from calendar state, actual-hours projections, or
  generic job status. The next tracker slice will supply an explicit,
  actor-attributed completion action anchored to a durable service identity.
- Do not create a post-clean email draft, public onboarding token, Stripe
  SetupIntent, or payment authorization in this PR.
- Do not grant the normal runtime `REFERENCES`, guard-role membership, table
  ownership, delete/truncate authority, or a startup path to apply migration
  394.

## Deferred

- Tracker manager action and Atlas proxy client that produces this receipt from
  a real completed service: #2156, next consumer slice after this provider
  deploys.
- Post-clean approval candidate, copy, public card-on-file flow, Stripe
  SetupIntent, cancellation policy, and email delivery/retry evidence: #2156.
- Customer-visible completion/status display: #2156 after the tracker contract
  is deployed and capability-gated.
- Migration catalog coordination: #2492 is a separate active DBA-only
  missed-call privilege repair that reserves migration `393`. This slice uses
  migration `394` to avoid a duplicate catalog name; neither workflow depends
  on the other.

Parked hardening: none. Existing calendar/booking data cannot safely serve as
an automatic completion source and is intentionally left outside this slice.

## Verification

- Local fast checks passed:
  - `ruff format` and `ruff check` passed for
    `atlas_brain/services/eom_first_clean_completion.py` and
    `tests/test_eom_first_clean_completion.py`.
  - `python -m py_compile atlas_brain/services/eom_first_clean_completion.py tests/test_eom_first_clean_completion.py`
  - `pytest -q tests/test_eom_first_clean_completion.py tests/test_eom_first_clean_completion_dba_runner.py`
    (`15 passed, 24 skipped`; the skipped cases require the deliberately absent
    `ATLAS_MIGRATION_TEST_DATABASE_URL`).
  - `git diff --check`
- The standalone Ruff target for `atlas_brain/main_eom.py` reports existing
  `E402` findings because that entrypoint intentionally loads local environment
  files before module imports; this slice leaves that bootstrap ordering intact.
- GitHub owns the full/unit and database-backed migration runs. No local test
  contacted Resend, Stripe, Google Calendar, or a real customer.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 12 |
| `atlas_brain/eom_api/funnel.py` | 123 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/eom_first_clean_completion.py` | 766 |
| `atlas_brain/storage/migrations/394_eom_first_clean_completion_receipts.sql` | 431 |
| `docs/EOM_FIRST_CLEAN_COMPLETION_RUNBOOK.md` | 85 |
| `plans/PR-First-Clean-Completion-Receipt.md` | 235 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 173 |
| `tests/test_eom_first_clean_completion.py` | 1441 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 154 |
| **Total** | **3421** |
