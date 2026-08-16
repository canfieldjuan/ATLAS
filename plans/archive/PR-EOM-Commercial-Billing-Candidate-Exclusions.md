# PR-EOM-Commercial-Billing-Candidate-Exclusions

## Why this slice exists

The active Commercial Billing workspace can save and reconcile a durable review,
then approve an individual candidate, but it has no durable decision boundary
for an operator to exclude a reviewed candidate. The Website renders candidate
cards without an exclusion action, the tracker exposes no decision proxy, and
the active ATLAS router exposes only run creation, reads, reconciliation, and
approval. The original workspace contract in [#2362](https://github.com/canfieldjuan/ATLAS/issues/2362)
requires exclusion before approval; the earlier billing-run plan deliberately
deferred it.

This is the provider-first half of the next vertical behavior. It creates the
auditable ATLAS-owned decision state that later tracker and Website slices will
use. It is safe to deploy before those consumers because no existing caller
sends a decision and the default remains included.

Review recovery root cause: the first implementation stated a closed request
boundary, append-only history, and per-candidate serialization, but its route
model silently ignored undeclared fields and bounded raw rather than normalized
reasons; its migration did not block direct database mutation or an old worker's
invoice insert; and its decision scope was narrower than the global
candidate/fingerprint approval identity. This repair fixes those upstream
enforcement points, rather than masking their downstream effects in a caller or
invoice writer. The follow-up review reconstruction also established two
remaining upstream gaps in that same boundary: the new route bounded a raw
`Idempotency-Key` before the shared service could trim it, and the final invoice
trigger's global retained-candidate identity lookup had no matching leading
index as retained runs grow.

### Problem-derived contract

- Root cause: immutable billing-run snapshots have no append-only review-decision
  record, so an excluded candidate is indistinguishable from an included one and
  the approval writer cannot reject it before creating an invoice.
- Correct fix must touch/change: add additive ATLAS persistence for an
  actor/reason/idempotency-audited `included` or `excluded` decision recorded
  against an exact retained snapshot but globally scoped to the existing
  candidate/fingerprint approval identity; add one authenticated run-owned
  route that normalizes reason and delegates idempotency-key length admission to
  the shared normalizing service; project the current global decision in run
  reads; and make both the current approval transaction and a database invoice
  trigger consult that global state before the first invoice write. The trigger
  must have an additive nonunique index beginning with its global retained
  candidate identity `(candidate_key, source_fingerprint)`.
- Must not change: pure preview generation; immutable candidate snapshots and
  source fingerprints; legacy monthly invoicing/MCP paths; invoice/PDF/Gmail/
  Square/receipt writers; existing no-decision approval behavior; customer or
  service source records; and any customer-facing wording or email/PDF shape.

## Scope (this PR)

Ownership lane: eom/commercial-billing-candidate-exclusions
Slice phase: Vertical slice
Max files: 9

Diff-budget override rationale: the additive schema, authenticated route,
existing-invoice writer fence, immutable-read projection, grammar-derived
zero-write boundary proof, and real PostgreSQL retry/concurrency proof are one
transaction boundary. Splitting any one out would publish an exclusion state
that an approval can ignore or a lock that has no durable, actor-audited
decision to protect. The ninth file is the application validation-error
fallback required to return a safe 422 when FastAPI rejects malformed Unicode
before this route can reach the shared service.

1. Add an additive migration and `CommercialBillingRunService` method that
   appends one actor-audited candidate review decision (`included` or
   `excluded`) for an exact retained candidate fingerprint, with a global
   candidate/fingerprint revision identity matching existing approval
   deduplication. A reason and idempotency key are required; a matching retry
   returns the original decision.
2. Add the active full-provider `PUT /receivables/commercial-billing-runs/{id}/candidates/{candidate_key}/review-decision`
   route, guarded by the existing receivables token and authenticated actor,
   that trims reason before its length constraints apply and lets the shared
   service apply the normalized `Idempotency-Key` length bound.
3. Project the latest global decision as derived read state in every matching
   run without rewriting immutable snapshots. Reject a current exclusion in the
   approval service and in a database `BEFORE INSERT` trigger for commercial
   billing invoices, including an old rolling-deployment worker.
4. Prove normal decision/reinclude, malformed/unknown/stale/approved rejection,
   same-key replay and mismatch conflict, current/old-worker invoice fencing,
   duplicate-run identity fencing, concurrent decision/approval ordering, route
   reachability/auth, normalized idempotency-header admission, the retained
   identity index, and no financial/delivery side effect from the decision route.

### Review Contract

- Acceptance criteria:
  - `CommercialBillingRunService.set_candidate_review_decision` appends exactly
    one event with actor, reason, request fingerprint, and ordered revision;
    `tests/test_commercial_billing_runs.py` proves default inclusion,
    exclusion, re-inclusion, and exact replay against real PostgreSQL.
  - The decision route derives actor through `require_actor`, admits only the
    closed `included`/`excluded` state set, and proves missing/invalid auth or
    malformed request data reaches neither the decision table nor an invoice
    writer in its ASGI test.
  - The route rejects every undeclared top-level JSON field before it calls the
    decision writer; its ASGI contract test exercises seven unsupported billing
    field shapes and observes zero service calls.
  - The route trims a scalar reason before Pydantic evaluates its one-to-1,000
    character grammar. Its ASGI test exercises seven whitespace/length forms,
    proving accepted input reaches the service normalized and rejected input
    reaches no writer.
  - The review-decision route does not apply its 128-character idempotency cap
    to the raw HTTP header. It delegates that bound to
    `CommercialBillingRunService._validate_idempotency_key`, which trims before
    validating. The ASGI route proof admits diverse padded at-limit header
    forms, while the existing grammar-derived service test rejects blank or
    normalized-over-limit forms before a transaction; the mounted full-app
    proof persists the canonical trimmed key and replays it without a second row.
  - The shared service text admission rejects embedded database-unsafe NUL
    characters and UTF-8-unencodable Unicode text in candidate key, reason,
    idempotency key, and actor before a transaction begins. Its
    grammar-derived zero-transaction test exercises each persisted scalar
    family rather than allowing the database to turn malformed client input
    into an unavailable-service result.
  - If FastAPI/Pydantic rejects an unpaired surrogate before it reaches the
    shared service, the application validation handler emits the same safe 422
    response rather than failing while serializing the invalid input in its
    default error payload. The mounted-route test submits ASCII-escaped JSON
    and proves no decision row exists before the following valid request.
  - A grammar-derived property test drives text tokens, whitespace modifiers,
    container forms, fingerprint tokens/lengths, and decision vocabulary through
    the service. Its specification-derived oracle proves only valid scalar input
    starts the transaction; every unrecognized form fails before a decision or
    invoice write.
  - The retained snapshot row, not the joined run parent, is locked by both the
    decision writer and
    `CommercialBillingApprovalService.approve`; its focused test demonstrates
    that a committed exclusion is rejected before `_insert_invoice`, while an
    absent or latest `included` decision retains old approval behavior.
  - The decision table is append-only in the database, rejecting direct update,
    delete, and truncate attempts, and migration 380 is additive. Its revision
    scope and all approval/read queries use the global candidate/fingerprint
    identity already deduplicated by approvals, so an exclusion in one retained
    run cannot be bypassed through another matching run.
  - The commercial-invoice `BEFORE INSERT` trigger takes the same global
    advisory identity lock as approvals, requires string-typed canonical
    `candidateKey` and `sourceFingerprint` metadata to exactly match a retained
    candidate identity, and rejects a committed exclusion before an invoice row
    exists. The real PostgreSQL test simulates that old writer and proves zero
    invoice rows until a later explicit inclusion; a real PostgreSQL probe also
    proves missing, blank, malformed, oversized, non-space-whitespace, and
    non-string-scalar identity metadata cannot bypass the final writer guard.
  - Migration 380 installs a nonunique
    `commercial_billing_run_candidates(candidate_key, source_fingerprint)`
    index. The migration test observes that installed index through
    `pg_indexes`, so the global retained-identity existence lookup used by the
    commercial-invoice trigger can seek across retained runs without relying on
    an index that begins with `billing_run_id`.
  - Run detail exposes the latest global derived decision without altering stored
    candidate JSON or its source fingerprint; the added `reviewDecision` field
    is backward-compatible.
  - The focused invoicing workflow enrolls migration, service, route, and tests;
    local workflow-equivalent checks pass before publication.
- Reachability proof: the real `atlas_brain.api.invoicing.receivables` router is
  mounted under `/api/v1`; an ASGI test sends token/actor headers to the new
  route backed by the production run service and isolated PostgreSQL, then
  observes both its decision response and the persisted decision row. No
  production decision, invoice, payment, PDF, Gmail draft, email, or Square
  action is created as a probe.
- Affected surfaces: `atlas_brain/services/commercial_billing_runs.py`,
  `atlas_brain/services/commercial_billing_approvals.py`,
  `atlas_brain/api/invoicing/receivables.py`, `atlas_brain/main.py`, migration
  380, focused commercial billing tests, and the invoicing workflow.
- Risk areas: token/actor admission, route normalization, stale fingerprints,
  idempotent replay, append-only audit history, global candidate identity,
  trigger lookup performance across retained runs, rolling-deployment old
  workers, concurrent decision/approval ordering, migration rollback, and
  accidental invoice creation from an exclusion write.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R12, R13, R14.

### Closure declaration

- `decision` is CLOSED: the only review-event values are `included` and
  `excluded`, authored in the commercial-billing review contract and reflected
  by the service, route model, and migration constraint. No unlisted lifecycle
  value belongs to a candidate-review decision.
- Candidate key, fingerprint, reason, actor, and idempotency key are OPEN
  caller-supplied input families. Their membership is derived from their scalar
  grammar (reason is trimmed before nonblank bounded-text evaluation; other text
  is trimmed bounded text; every persisted text value rejects embedded NUL and
  UTF-8-unencodable Unicode; fingerprints are exactly 64 lowercase hexadecimal
  characters), not a sampled vocabulary.
- Any malformed, nested, wrapped, unrecognized, out-of-set, or database-unsafe
  text input rejects before the transaction begins. If the framework rejects
  malformed Unicode before service admission, its error response remains a
  safe 422. The safe direction is zero financial/audit mutation: ambiguity must
  never silently become an approved or excluded candidate decision.

### Boundary-change enumeration

- Boundary path/seam: authenticated `PUT` decision route -> normalized request
  model -> encoding-safe validation 422 on framework-rejected Unicode, or
  `CommercialBillingRunService.set_candidate_review_decision` -> global
  candidate/fingerprint advisory lock plus retained-candidate row lock ->
  append-only event -> derived run view. Current approvals take the same locks
  and read the global event; the database invoice trigger repeats the global
  lock/read for a rolling old worker before invoice insertion, after admitting
  only exact raw JSON-string identities present in retained candidates.
- Replaced-path behaviors:
  - no prior decision event: preserved as the derived `included` default;
  - existing direct approval of a candidate with no decision: preserved;
  - current explicit `excluded` event: intentionally rejected before invoice
    insertion from any matching retained run or old commercial-invoice worker;
  - old-worker metadata with noncanonical whitespace, a non-string scalar, or
    an identity absent from retained candidates: intentionally rejected before
    advisory locking or invoice mutation;
  - candidate/source fingerprint mismatch, unknown run/candidate, malformed
    body, missing auth, and post-approval decision request: intentionally
    rejected with no decision/invoice write.
- Guard-relevant fields: service bearer token, server-derived actor, run UUID,
  candidate key, expected source fingerprint, `decision`, reason, idempotency
  key (including raw-header padding before shared normalization), any pre-existing
  exact candidate approval, and old-worker invoice
  metadata `candidateKey`/`sourceFingerprint` JSON types and exact retained
  identity; every persisted review-text field's UTF-8 encodability.
- Caller x input shape:
  - authenticated tracker caller + retained exact candidate + `excluded` or
    `included` + nonempty reason -> one append-only decision;
  - two retained runs with the same candidate key/fingerprint -> one shared
    current decision/revision identity; either run observes the same exclusion
    and only an explicit later inclusion clears it;
  - unchanged retry with same idempotency key -> original event, no second row;
  - padded at-limit idempotency header -> shared normalized key -> original
    event on either padded or canonical retry; blank or normalized-over-limit
    header -> reject before decision mutation;
  - reused key with a different candidate/fingerprint/decision/reason ->
    conflict, no second row;
  - unknown run/candidate, stale fingerprint, unsupported decision, blank/long
    or database-unencodable persisted text after normalization, missing
    actor/token, or already approved candidate -> reject before decision or
    invoice mutation;
  - direct existing approval caller with no decision/latest included -> preserved;
  - direct existing approval caller with latest excluded -> rejection before
    `_insert_invoice`; an old worker that reaches the database writer is also
    rejected by the invoice trigger.
  - old-worker invoice metadata containing canonical JSON strings that match a
    retained candidate -> existing final-writer behavior; tabs/newlines around
    the key, non-string JSON scalar forms, or an unmatched pair -> reject.

### Deployed-config probing

- Deployed/default config values: the active full receivables router uses the
  configured SHA-256 service-token dependency; the secret value remains
  intentionally unread. Production evidence remains a later read-only auth probe.
- Explicit value probe: ASGI test uses a generated configured token and
  authenticated actor to observe the decision result.
- Absent value probe: ASGI test omits bearer/actor and asserts rejection before
  the service writer is called.
- Default-session/default-context probe: the router's normal dependency chain
  rejects an unauthenticated request before its writer.
- Side-effect ordering: parsing/auth/fingerprint/approval checks happen before
  the decision insert; the decision route delegates only to the run service,
  and approval checks current exclusion under the retained-row lock before
  `_insert_invoice`. The migration's commercial-invoice trigger repeats the
  global identity check at the final database writer boundary for an old worker.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/main.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `atlas_brain/services/commercial_billing_runs.py`
- `atlas_brain/storage/migrations/380_commercial_billing_candidate_review_decisions.sql`
- `plans/PR-EOM-Commercial-Billing-Candidate-Exclusions.md`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_commercial_billing_runs.py`

### Fix-loop disposition preflight

- Root decisions: reject undeclared review fields; enforce append-only review
  evidence; lock only the requested retained candidate; fence a committed
  exclusion at the old-worker invoice writer; scope decisions to the global
  approval identity; normalize reason before bounds apply; reject database-unsafe
  NUL and UTF-8-unencodable Unicode in persisted review text; preserve a safe
  422 when framework validation rejects malformed Unicode; prove the real
  mounted route reaches the production persistence boundary; and reject
  noncanonical or non-string old-worker invoice identity metadata at the final
  database fence; delegate idempotency-header length admission to the shared
  normalized validator; and index the trigger's retained global identity lookup.
- Source trace: request JSON -> Pydantic model -> encoding-safe validation
  response or shared persisted-text validator -> run service; raw
  `Idempotency-Key` header -> shared trim/length validator -> idempotent event
  identity; retained candidate/fingerprint -> global approval identity ->
  current/old invoice writer; migration 380 -> application-role database
  mutation, trigger identity lookup, and index enforcement.
- Upstream files: `atlas_brain/api/invoicing/receivables.py`,
  `atlas_brain/main.py`,
  `atlas_brain/services/commercial_billing_runs.py`,
  `atlas_brain/services/commercial_billing_approvals.py`, and migration 380.
- Fix strategy: upstream-root. The route normalizes/forbids its request model
  before delegation and leaves the normalized idempotency length decision to the
  shared service; the application preserves a serializable 422 for
  framework-rejected Unicode; the shared service validates every persisted
  scalar before opening a transaction and is the sole normalized idempotency
  length choke point; the
  run/approval services use the existing global identity; the migration enforces
  immutable history, an indexed retained-identity lookup, and a last-writer
  invoice fence.
- Blocking predicate: twelve current-head live-reconciliation R1/R3, R1/R4,
  R7, R8, R1/R8, R1, R1/R3, R1/R2, final database identity R1/R3,
  database-encodability R1/R3/R13, normalized idempotency-header R1/R13, and
  retained-identity index R7 threads.
- Disposition: fix all twelve in this PR; no reviewer finding is waived.
- Allowed files / Max files: the nine-file scope remains the hard
  ceiling; recovery changes use only the route, run/approval services,
  migration, existing tests, plan, workflow enrollment, and external PR body.
- Parked hardening target: #2363 remains unchanged because all twelve findings
  violate this slice's claimed money/audit/concurrency boundary rather than
  adding optional hardening.

## Mechanism

Migration 380 creates an append-only decision-event table referencing the exact
`(billing_run_id, candidate_key, source_fingerprint)` snapshot key while its
revision identity is the same global `(candidate_key, source_fingerprint)` pair
already deduplicated by approvals. A decision is a closed state value,
`included` or `excluded`; absence is derived as included. Each event stores
revision, actor, reason, idempotency key, request fingerprint, and timestamp.

The run service locks only the immutable candidate row with `FOR UPDATE OF
candidate` after the existing global candidate/fingerprint advisory lock, then
assigns the next global revision. The approval service takes the same locks,
reads the latest global decision, and raises before `_insert_invoice` when it
is excluded. Before taking its matching advisory lock, the database `BEFORE
INSERT` trigger requires JSON-string metadata and an exact retained
`(candidate_key, source_fingerprint)` match; it then enforces the exact state
for an old worker that does not yet contain the new service read. This makes the
database and existing lock namespace the closed-surface ordering primitive
rather than relying on process version or in-memory state.

Run detail reads a latest-global-event projection and attaches it as additive
`reviewDecision` state to every matching retained run. The original `snapshot`
JSON and source fingerprint remain untouched. The route trims its reason before
Pydantic length admission and leaves idempotency-key length admission to the
shared service after that service trims the header value. The shared service
rejects embedded NUL and UTF-8-unencodable text in every persisted scalar before
opening its transaction. Migration 380 also gives the old-worker trigger's
global retained-candidate existence check its own leading identity index, while
keeping it nonunique because duplicate retained runs are a supported snapshot
shape.
If Pydantic rejects an unpaired surrogate before the service, the application
uses ASCII-safe JSON serialization for that existing 422 validation response.
Its full mounted-route proof uses the production run service with isolated
PostgreSQL and observes a malformed request's 422/no row followed by an
inserted valid decision row; tracker and Website controls are separate consumer
slices after provider deployment.

## Intentional

- `included` and `excluded` are CLOSED and ENUMERATED in request validation
  and the database constraint. An absent event derives to `included`;
  unrecognized input fails closed.
- Re-inclusion is an additional append-only `included` event, not an update or
  delete, preserving who reversed an earlier exclusion and why.
- A review decision is recorded through one retained run but applies to the
  globally deduplicated candidate/fingerprint approval identity. A matching
  duplicate run therefore cannot manufacture a second, contradictory outcome.
- At the final old-worker writer boundary, review identity is CLOSED to raw
  JSON strings that exactly equal a retained candidate/fingerprint pair.
  PostgreSQL scalar coercion and whitespace normalization are not identity
  admission mechanisms.
- The invoice trigger is limited to `eom_commercial_billing` inserts. The
  deployed pre-change approval writer already persists candidate key and source
  fingerprint in that invoice metadata; a missing or malformed identity fails
  closed instead of bypassing a committed exclusion.
- Reason whitespace is normalization, not policy: route and service both admit
  the same trimmed one-to-1,000-character grammar.
- Idempotency-key whitespace is likewise normalization, not policy: only the
  shared service applies its 1-to-128-character bound after trimming. The new
  review-decision route therefore removes its duplicate raw `max_length` check;
  unrelated existing receivables routes are intentionally unchanged.
- Database-unsafe NUL and UTF-8-unencodable Unicode are malformed wire input,
  not a storage retry: every persisted scalar review field rejects them before
  the transaction starts.
- The application validation-response fallback is inert for ordinary validation
  errors. It activates only if FastAPI's default JSON serializer cannot encode
  an invalid Unicode value carried in a validation error, preserving its 422
  status and response schema without changing valid request behavior.
- The retained-candidate identity index is nonunique and additive. It supports
  the trigger's existence predicate without changing the already-supported
  duplicate-run/revision identity semantics.
- An already-approved exact candidate cannot receive a later decision. Excluding
  it would imply a financial reversal, which is a separate correction path.
- This PR does not choose editable monetary, quantity, rate, recipient, tax, or
  delivery fields. Those materially change an invoice/customer outcome and the
  current evidence does not define the permitted set.
- No tracker or Website caller is included. Provider-first deployment preserves
  all old consumers.

## Deferred

- #2362: tracker proxy and Website candidate-card controls for this durable
  exclusion/reinclusion boundary, then browser proof of the full operator flow.
- #2362: define the operator-authorized permitted draft-field override set and
  its financial/audit semantics before implementing candidate edits beyond
  include/exclude.
- #2363: existing nonessential billing hardening remains unchanged; the unresolved
  edit-field choice is a product/financial policy decision, not hardening.

Parking predicate: park additional reporting, UI polish, unrelated historical
ledger behavior, scheduler changes, and generic billing-model refactors unless
they block the exclusion decision or prove a money/security/data-correctness
failure in this exact boundary.

Parked hardening: none.

## Verification

- Passed locally against isolated PostgreSQL 16: focused header/index regression
  selection — 9 passed; the padded at-limit header reached the service, blank
  and normalized-over-limit forms returned 422 without a decision row, the
  canonical retry replayed the exact event, and `pg_indexes` exposed the new
  retained-identity index.
- Passed locally against isolated PostgreSQL 16:
  `pytest -q tests/test_commercial_billing_runs.py
  tests/test_commercial_billing_approvals.py` — 135 passed. No production
  database, financial record, Gmail draft, or customer email was used.
- Passed locally against isolated PostgreSQL 16: exact invoicing workflow test
  groups — 2 passed / 670 passed / 43 passed.
- Passed locally: `ruff check --ignore E402` on changed Python,
  `py_compile`, workflow YAML parsing, guard-class-closure strict check, and
  `git diff --check`.
- Pending after the current repair: the pinned Gitleaks PR-range scan,
  plan/body audits, and one managed local PR-review run through
  `scripts/push_pr.sh`.
- Pending after provider merge: a read-only active-route/auth registration probe;
  no financial or customer record will be created for verification.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 4 |
| `atlas_brain/api/invoicing/receivables.py` | 47 |
| `atlas_brain/main.py` | 31 |
| `atlas_brain/services/commercial_billing_approvals.py` | 52 |
| `atlas_brain/services/commercial_billing_runs.py` | 337 |
| `atlas_brain/storage/migrations/380_commercial_billing_candidate_review_decisions.sql` | 165 |
| `plans/PR-EOM-Commercial-Billing-Candidate-Exclusions.md` | 446 |
| `tests/test_commercial_billing_approvals.py` | 508 |
| `tests/test_commercial_billing_runs.py` | 877 |
| **Total** | **2467** |
