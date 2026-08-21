# PR-EOM-Billing-Run-Review

## Why this slice exists

The deployed commercial candidate route is intentionally a pure read: it
returns current source evidence and a per-candidate SHA-256 fingerprint, but it
does not preserve the preview Juan reviewed.  A later approval operation would
therefore have no durable source snapshot to compare against, and could not
prove that a service, rate, canonical customer, or calendar event changed after
review.  Coordinating issue #2362 names stored candidates and source-fingerprint
reconciliation as the next provider-first dependency after the deployed
Website preview (#195).

The current legacy monthly task is not a substitute.  Its review mode still
creates an invoice, writes a PDF, marks services invoiced, logs CRM activity,
and can notify or send (`atlas_brain/autonomous/tasks/monthly_invoice_generation.py:322-453`).
The active full provider instead exposes the pure candidate seam at
`atlas_brain/api/invoicing/receivables.py:364-389`, backed by
`CommercialBillingCandidateService.preview`
(`atlas_brain/services/commercial_billing_candidates.py:401-568`).

### Problem-derived contract

- Root cause: a transient candidate response cannot form an auditable review
  boundary or later detect source drift; the only older monthly path is writeful
  before an operator can review it.
- Correct fix must touch/change: add additive ATLAS-owned draft-run and
  immutable candidate-snapshot tables; add a focused provider service that
  snapshots the already-pure generator result inside one database transaction;
  add authenticated create/list/detail/reconciliation routes; enroll focused
  unit, real-PostgreSQL, migration, route, retry, failure, and no-delivery tests
  in the existing invoicing workflow.
- Must not change: the pure candidate GET, the legacy monthly scheduler,
  invoices, invoice balances, payment/receipt/deposit lifecycle, MCP tools,
  Gmail, email transport, PDFs, service invoiced markers, tracker state, and
  Website behavior.  This slice has no candidate editing, exclusion, selection,
  approval, invoice, PDF, Gmail-draft, Square, or sent-mail writer.

## Scope (this PR)

Ownership lane: eom/billing-run-review
Slice phase: Vertical slice

1. Add an authenticated ATLAS-only `POST /receivables/commercial-billing-runs`
   that takes an explicit billing period plus `Idempotency-Key`, invokes the
   existing pure candidate service, and atomically persists one `draft` run and
   immutable candidate snapshots with the authenticated actor, source evidence,
   candidate fingerprint, contract version, and an aggregate snapshot
   fingerprint.
2. Add bounded authenticated list/detail reads and a read-only reconciliation
   route.  Reconciliation regenerates the current pure preview and reports each
   stored candidate as unchanged, changed, missing, or new, plus an overall
   stale verdict; it writes nothing.
3. Prove that an unchanged same-key retry returns the original run, a reused key
   with another period conflicts, concurrent same-key creation has exactly one
   run/snapshot set, source-reader failure leaves no draft rows, a later source
   fingerprint/roster change is visible as stale, and neither route nor service
   imports or calls invoice/PDF/Gmail/email/scheduler writers.

### Review Contract

- Acceptance criteria:
  - `CommercialBillingRunService.create_run` records a server-generated,
    deterministic snapshot only after the candidate generator returns a valid
    preview; `tests/test_commercial_billing_runs.py` verifies the persisted
    candidate JSON, per-candidate source fingerprint, aggregate fingerprint,
    actor, exact cents, and source evidence through real PostgreSQL.
  - `POST /receivables/commercial-billing-runs` is protected by the existing
    service token and derives the actor with `require_actor`; its ASGI contract
    test proves unauthenticated and malformed-period calls reach neither source
    reader nor database writer, and a valid request produces the durable run.
  - Equal `(source, Idempotency-Key)` calls serialize on a PostgreSQL advisory
    transaction lock and a unique index, compare the canonical
    `billing_period` request fingerprint, and return the original run.  The
    execution invariant is: for every admitted scheduling interleaving, one key
    can commit at most one parent run and one immutable snapshot per candidate;
    a failed transaction commits neither.  Real-PostgreSQL retry/concurrency and
    failure/recovery tests settle this R8 claim.
  - `GET /receivables/commercial-billing-runs/{id}/reconciliation` consumes the
    current pure generator result and compares the closed stored/current
    candidate-key relation.  The focused tests prove changed source fingerprints,
    removed snapshots, and new source candidates mark the run stale, while an
    identical retry remains current and does not update any run row.
  - The migration is additive, contains no backfill or destructive DDL, creates
    its indexes before code relies on the tables, and documents rollback as
    application rollback first while retaining audit evidence.  Migration and
    workflow-enrollment tests prove the packaged SQL and local workflow cover it.
  - AST/source guards prove the new service does not import the writeful monthly
    scheduler, invoice repository, PDF renderer, Gmail, email provider,
    notification sender, or CRM mutation surface; recording or reconciling a
    run cannot create an invoice, PDF, draft, email, service marker, or sent
    state.
- Reachability proof: the active full application mounts the authenticated
  `atlas_brain.api.invoicing.receivables` router under `/api/v1`; an ASGI test
  includes that real router, uses the established hashed service-token
  dependency, and observes create/detail/reconciliation output.  After merge,
  the deployed route will be probed unauthenticated only (expected 401); no
  production billing run or financial record will be created as a probe.
- Affected surfaces: `atlas_brain/services/commercial_billing_runs.py`,
  `atlas_brain/api/invoicing/receivables.py`, migration 370, the existing
  invoicing workflow, its focused test file, and this plan.  The dormant slim
  `eom_api.receivables` router remains unchanged because deployed evidence says
  the full router is active and H-06 already tracks the duplicate-model decision.
- Risk areas: protected-write authorization, source snapshot validity,
  idempotent retry/cancellation, schema rollout, PostgreSQL isolation, stale
  source evidence, bounded reads, nested snapshot JSON, and accidental financial
  or delivery side effects.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12,
  R13, and R14.  R9 is N/A: no browser surface changes.

### Boundary-change enumeration

- Boundary path/seam: authenticated `POST /receivables/commercial-billing-runs`
  -> existing pure `CommercialBillingCandidateService.preview` -> transactional
  run/snapshot persistence; authenticated list/detail/reconciliation reads use
  the same token boundary.  The existing candidate GET and every legacy writer
  remain separate.
- Replaced-path behaviors: none.  This is additive; existing GET preview,
  payment, invoice, MCP, scheduler, and Gmail callers keep their contracts.
- Guard-relevant fields: JSON `billing_period`, `Idempotency-Key`, service-token
  Authorization, server-derived `X-EOM-Actor`, route `run_id`, list `limit` and
  `offset`, generated `candidateKey`, `sourceFingerprint`, and contract version.
- Caller x input shape: tracker service caller with valid/invalid/missing bearer;
  valid `YYYY-MM` and malformed period strings; nonempty/blank/overlong/reused
  idempotency keys; known/unknown UUID run IDs; bounded/out-of-range list
  pagination; generator outputs with empty, valid, duplicate-key, or invalid
  fingerprint candidates.  Unrecognized or malformed input fails before a
  financial/delivery write and before the candidate source is called where the
  route/parser can decide it.

### Deployed-config probing

- Deployed/default config values: N/A - no new setting, flag, credential,
  fallback, or environment variable is introduced.  The route reuses the
  already-deployed receivables token/authentication and primary database pool.
- Explicit value probe: post-deploy unauthenticated route probe must return 401;
  it proves route registration and auth without creating a billing run.
- Absent value probe: N/A - route configuration is the existing fail-closed
  receivables configuration, not a new fallback.
- Default-session/default-context probe: the create route records only the
  authenticated actor supplied by `require_actor`; it never derives a customer,
  delivery preference, or mailbox default.
- Side-effect ordering: candidate generation completes before the one database
  transaction.  The parent row and every candidate row commit together; source
  failure or database rollback produces no run.  Reconciliation is read-only.

### Closure declaration

- Persisted run state is **CLOSED**, with the sole `draft` member authored in
  migration 370 for this pre-approval slice.  The database check rejects any
  other value, which is the safe side because approval/sending semantics have
  not been implemented.
- Reconciliation membership is **CLOSED** and **DERIVED** at comparison time
  from the stored candidate-key map and the current pure-preview candidate-key
  map.  The exhaustive relation emits only `unchanged`, `changed`, `missing`, or
  `new`; a malformed generated candidate or duplicate key fails the read/run
  instead of being silently treated as current.  This is the cheap/safe side:
  refusing to claim freshness may require a retry, while accepting unknown
  evidence could approve a stale invoice later.
- `billing_period` is an **OPEN** string input whose valid grammar is
  **DERIVED** from the existing `parse_billing_period` choke point.  Every
  unmatched shape reaches its existing `invalid_billing_period` 422 before any
  source reader or database mutation.  This plan does not duplicate a new
  period regex/allowlist.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_runs.py`
- `atlas_brain/storage/migrations/370_commercial_billing_runs.sql`
- `plans/PR-EOM-Billing-Run-Review.md`
- `tests/test_commercial_billing_runs.py`

## Mechanism

`CommercialBillingRunService` owns the durable review boundary and accepts a
pool plus an injected pure candidate service for tests.  It validates the period
through the already-canonical parser before any source access, normalizes only
the generator's `candidateKey`/SHA-256 fingerprint/JSON-safe snapshot contract,
then computes a canonical aggregate fingerprint from the period, calendar,
candidate contract version, and sorted candidate key/fingerprint pairs.

The migration adds `commercial_billing_runs` and
`commercial_billing_run_candidates`.  The parent records `draft`, period,
calendar ID, generator contract version, aggregate fingerprint, request/idempotency
fingerprints, actor, and timestamps.  Each child uses a `(billing_run_id,
candidate_key)` uniqueness backstop and holds the complete immutable candidate
JSON plus its original source fingerprint and display order.  This retains line
items, cents, recipient, blockers, services, and calendar evidence without
recomputing history.

Creation uses PostgreSQL's existing transaction/advisory-lock component rather
than a hand-rolled lease.  It serializes the operation key, locks/rechecks an
existing parent row, and compares canonical request fingerprints.  A matching
replay returns that row; a mismatched replay raises a named conflict.  Only an
unseen key calls the pure generator, then starts one transaction that inserts the
parent and every child together.  A unique index remains the storage backstop;
the service re-reads a conflicting same-key row before returning or raising.
The execution model is one database transaction per create, with PostgreSQL
atomic commit/rollback and transaction-scoped advisory lock semantics; it makes
the invariant in the Review Contract hold across all interleavings the database
admits.  No external side effect occurs inside or after that transaction.

List/detail are bounded repository reads.  Reconciliation loads the immutable
run, invokes the same pure generator for its saved period, and performs a
deterministic two-map comparison.  It never updates stored evidence or declares
a missing current source candidate sent/approved.  A source outage returns a
stable 503 so an unknown comparison cannot be reported as fresh.

The router maps only the new domain validation/not-found/conflict/unavailable
errors to stable 4xx/503 responses.  It reuses existing receivables token and
actor dependencies, leaves the pure GET contract untouched, and imports no
financial/delivery writer.

## Intentional

- A draft run is durable review evidence, not an invoice or financial lifecycle
  mutation.  It is the smallest state allowed before explicit approval.
- Multiple distinct operator operations may create separate draft runs for the
  same billing period.  This slice guarantees same-key idempotency but does not
  invent a global one-run-per-month policy; later explicit selection/approval
  must own reconciliation and invoice-level idempotency.
- Reconciliation reports that evidence changed but does not attempt an automatic
  merge, candidate edit, exclusion, regeneration overwrite, or approval.  A
  later operator review slice owns those business decisions.
- The current missing delivery-preference blocker remains stored exactly as
  projected.  Customer type does not infer Gmail or Square.
- Full-provider deployment is the only active route evidence.  The slim EOM
  profile and its migration closure stay untouched rather than creating another
  unproven provider surface.

## Deferred

- #2362: tracker proxy and Website durable-review UI; permitted candidate edits,
  explicit inclusion/exclusion, stale reconciliation/regeneration action, and
  selected-candidate approval remain later slices.
- #2362: idempotent invoice/PDF/Gmail-draft creation/recovery, sent-mail
  reconciliation, manual Square queue, and operating documentation remain later
  slices.
- #2363 H-13: canonical persisted delivery preference, service/site identity,
  and manual Square reference capture are required before an approval writer.
- #2363 H-06: duplicate full/slim receivables model ownership remains parked;
  this slice touches only the confirmed active full router.

Parked hardening: unrelated legacy invoice float arithmetic, scheduler changes,
generic billing preference schema, e-mail transport, migration-runner redesign,
and full historical customer-ledger improvements remain out of scope.

## Verification

- Local workflow-equivalent evidence (2026-08-13): a disposable CPython 3.11
  environment against an isolated local PostgreSQL schema ran the exact enrolled
  approval-blocker command (2 passed), receivables/repository command (242
  passed), and MCP/OAuth command (43 passed).  The focused billing-run plus
  candidate command passed 22 tests.  `webrtcvad` alone could not compile in
  this host's disposable environment because the host lacks Python 3.11 C
  headers; it is unrelated to these test imports, no host package was changed,
  and all workflow test groups themselves passed locally.
- `python -m ruff check` and `python -m compileall -q` passed for every changed
  Python/test file; `python scripts/sync_pr_plan.py
  plans/PR-EOM-Billing-Run-Review.md --check`, `git diff --check`, migration,
  route, retry/concurrency/recovery, workflow-enrollment, and static
  no-writer-import assertions all passed.  The legacy scheduler, MCP surface,
  and invoice repository have no diff from the inspected base.
- Before push, `bash scripts/local_pr_review.sh --current-pr-body-file
  tmp/PR_BODY_EOM_Billing_Run_Review.md origin/main` passed.  Its full local
  unit-gate mirror reported 160 known failing/errored baseline nodes, zero
  regressions, and zero newly passing nodes; every policy, plan, cross-layer,
  reconciliation, closure, and baseline check passed.
- Hosted Actions will be inspected but are not acceptance evidence under Juan's
  explicit local-check instruction.  The known account-spending pre-step
  failure remains H-11; no hosted failure will be reclassified as a source pass.
- Before merge, perform thread-aware GraphQL inspection and address every
  actionable thread on the published head.  A quota-exhaustion conversation note
  with no requested change is recorded but not artificially resolved.
- After merge, deploy ATLAS before any tracker consumer.  Verify the protected
  route with an unauthenticated 401 only; do not create a live draft run, invoice,
  PDF, Gmail draft, or email to prove deployment.

This exceeds the 400-LOC soft target because durable source snapshots,
transactional idempotency, storage rollback proof, source-fingerprint
reconciliation, and the required real-entrypoint/real-PostgreSQL execution
tests are one inseparable financial-review boundary.  Splitting the schema or
route from its concurrency/recovery proof would publish an unsafe writer; the
next tracker and Website consumers remain separate deployable slices.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 109 |
| `atlas_brain/services/commercial_billing_runs.py` | 655 |
| `atlas_brain/storage/migrations/370_commercial_billing_runs.sql` | 79 |
| `plans/PR-EOM-Billing-Run-Review.md` | 298 |
| `tests/test_commercial_billing_runs.py` | 727 |
| **Total** | **1875** |
