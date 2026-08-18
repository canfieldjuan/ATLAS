# PR-EOM-Commercial-Billing-Candidate-Overrides

## Why this slice exists

The completed EOM commercial billing workflow can retain a source-evidence
preview and record audited Include/Exclude decisions, but it cannot correct a
one-time quantity/rate, resolve missing hourly quantity, add a disclosed
credit/charge, or route an individual candidate to a run-only recipient before
an invoice exists. Juan explicitly approved a hybrid policy: source correction
for recurring truth, and immutable run-only overrides for legitimate one-time
exceptions. This provider slice is the financial and audit boundary required
before tracker and website consumers can expose that policy.

This intentionally exceeds the repository's 400-LOC soft target. The durable
override projection, one atomic migration, API admission, review/approval
guards, and real-Postgres recovery tests must ship together: splitting the
migration from the provider guard would create an interval where a prior
Include could approve changed money or recipient evidence. The resulting diff
is large because it closes that single financial invariant across the existing
run, approval, and database boundaries rather than adding an unrelated
product surface.

### Problem-derived contract

- Root cause: retained candidates are immutable source snapshots and approval
  is keyed only by their source fingerprint, so a direct edit would either
  rewrite evidence or leave a prior Include decision eligible for different
  money/recipient data.
- Correct fix must touch/change: append immutable override revisions outside
  the source snapshot; derive an effective candidate and a review identity;
  scope review decisions and approval to that identity; revalidate exact-cent
  money and permitted blocker recovery server-side; expose an additive ATLAS
  route with idempotency and actor audit; and prove the normal, rejected,
  replay, stale, and post-approval paths.
- Must not change: canonical contacts/services/rates/calendar evidence/tax
  policy; legacy preview shape or source fingerprints; existing no-override
  review/approval/PDF/Gmail/Square/MCP workflows; invoice sending behavior;
  tracker ledger ownership; and customer-facing email/PDF rendering.

## Scope (this PR)

Ownership lane: eom-commercial-billing/candidate-overrides
Slice phase: Vertical slice
Max files: 12

1. Add append-only, one-run-only commercial candidate override evidence and an
   effective candidate projection before approval.
2. Add an audited ATLAS override endpoint and extend review/approval identity
   additively so an override always requires a fresh explicit Include.
3. Prove exact-cent calculations, retry/concurrency rejection, stale source
   blocking, and preservation of legacy no-override behavior.
4. Review-fix iteration: bind every override review identity and final invoice
   trigger lookup to the retained billing run; canonicalize equivalent legacy
   review-decision retries before hashing; and preserve a committed exact line
   amount when a draft invoice is edited without replacing its line items.
5. Review-fix iteration: admit override descriptions only up to the shared
   invoice-line limit and validate Gmail recipients with the canonical contact
   email normalizer so an accepted, otherwise-billable override cannot strand
   its later approval or Gmail-draft path.
6. Review-fix iteration: derive the missing-billing-email blocker from the
   effective Gmail delivery state, including when a delivery-method override
   changes a no-recipient manual-Square or receipt-only candidate to Gmail.
7. Review-fix iteration: preserve recorded line amounts only when the
   controlled commercial-billing approval writer marks its exact-cent evidence;
   notes-only edits of generic/MCP invoices must continue to calculate from
   quantity and unit price.
8. Review-fix iteration: retain an `invalid_rate` blocker for a service the
   canonical producer omitted from line items, because an override cannot
   safely invent that missing source evidence.
9. Review-fix iteration: state the PostgreSQL transaction/lock execution model
   and prove a contended same-key override commits one revision and replays the
   other caller.

### Review Contract

- Acceptance criteria:
  1. `POST /receivables/commercial-billing-runs/{id}/candidates/{key}/override`
     appends one actor/audit revision and returns an effective candidate without
     creating an invoice, PDF, Gmail draft, email, Square record, or sent state;
     settled by the real route and
     `tests/test_commercial_billing_candidate_overrides.py`.
  2. The effective projection preserves the retained source snapshot and source
     fingerprint, derives a distinct review fingerprint for an override, and
     recomputes money in integer cents; settled by service tests covering minutes,
     rate/quantity, signed adjustment, and inherited tax.
  3. A review decision and approval require the submitted effective review
     fingerprint. A newly appended override has no eligible Include decision;
     settled by real-Postgres migration/service tests and approval tests.
  4. Matching retries return the original override, changed idempotency reuse
     conflicts, stale revisions/source and post-approval writes create no rows;
     settled by real-Postgres override/review/approval tests.
  5. Existing requests that omit the additive review-fingerprint field keep their
     current no-override behavior; settled by current commercial billing run and
     approval regression tests.
  6. The same candidate/source pair in separate retained runs has independent
     override review identities: a run-B override cannot stale run A at the final
     invoice trigger, and same-content run overrides cannot collide; settled by
     real-Postgres and pure identity tests.
  7. A notes- or due-date-only edit preserves an approved hourly line's explicit
     exact-cent amount only when the commercial approval writer marked that
     retained evidence; generic/MCP caller-supplied amounts recalculate from
     quantity and price instead of becoming authoritative; settled by repository
     regression tests using multiple minute/rate pairs and untrusted sources.
  8. An override with an overlong line/adjustment description or a Gmail
     recipient rejected by the downstream contact grammar fails at admission;
     a maximum-length line description still produces an approvable draft;
     settled by diverse pure override and approval regression tests.
  9. A delivery-method override to Gmail with no effective canonical recipient
     includes exactly one `missing_billing_email` blocker (and the normal
     derived zero-total blocker) rather than becoming approval-eligible;
     settled by diverse manual-Square, receipt-only, and missing-preference
     transition tests.
  10. An `invalid_rate` blocker can clear only when every affected retained
      source line has a specific valid rate override. If the canonical producer
      omitted the unpriced service entirely, valid sibling lines or an
      adjustment leave it visibly blocked and unapprovable; settled by pure
      effective-projection tests.
  11. Concurrency execution model: all durable writes occur in PostgreSQL
      transactions. Each operation first serializes its idempotency key, then
      candidate mutations and final approval share the candidate/source
      transaction advisory lock and lock the retained run candidate row. The
      approval's read-only preview/preflight may run between transactions, but
      its final transaction reacquires both authoritative locks before writing.
      Across every admitted interleaving for one operation key, exactly one
      matching request commits and all matching retries replay it; a different
      request conflicts. Across every admitted interleaving for one
      candidate/source, the candidate lock defines the commit order: a prior
      review/override is observed by approval, while a prior approval causes a
      later review/override to reject. A cancelled or failing transaction
      commits neither partial override/review state nor an invoice/approval
      pair. The migration's uniqueness constraints and invoice trigger remain
      the database backstop for direct and mixed-version writers. This is
      settled by
      `test_real_postgres_concurrent_override_replays_one_committed_revision`
      (same operation key),
      `test_real_postgres_exclusion_and_approval_serialize_at_the_same_candidate_boundary`
      (shared candidate lock),
      `test_real_postgres_override_requires_a_fresh_include_and_retries_without_invoice_side_effects`
      (retry/stale order), and
      `test_real_postgres_rolls_back_the_invoice_when_approval_audit_insert_fails`
      (transaction abort).
- Reachability proof: exercise the FastAPI receivables route with the actual
  dependency boundary and assert its returned effective candidate/audit state;
  direct service tests additionally settle the Postgres locking/migration facts.
- Affected surfaces: commercial billing run snapshots/review decisions,
  approvals/invoice metadata, receivables API models/routes, and additive
  migrations only.
- Risk areas: exact money, mutable-history prevention, review-selection reset,
  concurrency/idempotency, stale source evidence, legacy mixed-version readers,
  and no-delivery side effects.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: the new override request validator and the existing
  review-decision/approval identity admission checks, including the final
  invoice trigger's `commercialBillingRunId` metadata binding.
- Replaced-path behaviors: legacy no-override calls derive
  `reviewFingerprint == sourceFingerprint`; overridden candidates require the
  supplied current review fingerprint.
- Guard-relevant fields: billing run UUID, invoice `commercialBillingRunId`,
  candidate key, source fingerprint,
  expected override revision, review fingerprint, line key, integer cents,
  whole minutes, delivery method, recipient, closed reason code, note, actor,
  idempotency key, canonical Gmail recipient, and invoice-line description
  length.
- Caller x input shape: direct ATLAS API callers may omit the additive review
  fingerprint only when the candidate has no override; new tracker callers send
  it for every review/approval after this provider deploys.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: N/A - no environment/config fallback changes.
- Explicit value probe: N/A.
- Absent value probe: an absent review fingerprint is accepted only for an
  unoverridden candidate and rejected once an override exists.
- Default-session/default-context probe: N/A.
- Side-effect ordering: all validation, locks, source/revision checks, and
  idempotency lookup occur before the override insert; override itself is the
  only successful write and approval still separately creates the first invoice.

### Guard class-closure declaration

- `OVERRIDE_REASON_CODES` and `OVERRIDE_DELIVERY_METHODS` are **CLOSED** policy
  sets authored in `commercial_billing_candidate_overrides.py`; API, service,
  and migration constraints admit only their enumerated members. An out-of-set
  value rejects before any write because billing evidence must never infer a
  delivery or exception policy.
- The source fingerprint/review fingerprint language is a **CLOSED** structural
  format (64 lowercase hexadecimal characters); malformed or stale values
  reject at route/service/database admission. The effective snapshot is not an
  open customer-text classifier: the route permits only its typed fields and
  the service projects them over retained source evidence.
- The strict heuristic's changed `_validate_*` definitions therefore do not
  represent a trigger-A open-input safety guard. Its requested grammar-derived
  property test is inapplicable; the PR body records the supported
  `guard-class-closure: waived` marker with this rationale while the direct
  malformed/closed-set/real-Postgres tests remain required evidence.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/main.py`
- `atlas_brain/services/commercial_billing_approvals.py`
- `atlas_brain/services/commercial_billing_candidate_overrides.py`
- `atlas_brain/services/commercial_billing_runs.py`
- `atlas_brain/storage/migrations/382_commercial_billing_candidate_overrides.sql`
- `atlas_brain/storage/repositories/invoice.py`
- `plans/PR-EOM-Commercial-Billing-Candidate-Overrides.md`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_commercial_billing_candidate_overrides.py`
- `tests/test_commercial_billing_runs.py`

## Mechanism

The migration creates an append-only override history keyed to an immutable
run candidate and its source fingerprint. Each revision stores the validated
effective snapshot, exact request fingerprint, actor, reason code, note, and
operation key. Its review identity includes that retained run, and the final
invoice trigger validates the invoice's `commercialBillingRunId` before looking
up an active override. The base candidate row is never changed.

The run reader overlays only the latest override and exposes both source and
effective evidence. An override calculates its review fingerprint from the
retained run, source identity, and effective snapshot. Review-decision and
approval records carry that identity; legacy rows use their existing source
fingerprint as the review fingerprint. This makes an older Include decision
ineligible as soon as an override revision exists.

For a draft edit that does not replace line items, the invoice repository
preserves a persisted explicit line `amount` only when the controlled
commercial-billing approval writer marked its exact-cent evidence. This retains
the exact cents approved from whole minutes while generic/MCP caller-supplied
amount fields still recalculate from quantity and unit price. Supplying
replacement line items always retains the existing recalculation behavior.

Override admission shares the approval line-description limit and canonical
contact-email normalizer. This rejects an invalid description or Gmail
recipient before it can clear a blocker, receive a review decision, or create
an invoice that the PDF/Gmail delivery path cannot process.

The effective projection also recomputes the Gmail-recipient blocker after a
delivery-method override. A candidate that becomes Gmail-routed without a
canonical recipient stays visibly blocked instead of failing later during
invoice approval or Gmail draft creation.

An `invalid_rate` blocker is service-specific evidence, not a summary of the
currently materialized lines. The canonical producer intentionally omits an
unpriced service line, so an override can clear that blocker only when every
affected retained source line carries its own valid rate override. An omitted
service has no targetable source line and remains blocked; a sibling line or
one-time adjustment cannot silently bill around it.

The migration is atomic and preserves mixed-version safety: legacy provider
INSERTs that omit the new review-fingerprint column receive the source
fingerprint in a database trigger, while the invoice trigger rejects any
legacy source identity once a newer effective review identity exists. A
rollback therefore cannot silently approve an override through an older
provider binary.

Concurrency is intentionally modeled at the PostgreSQL transaction boundary.
Override and review-decision writers take an operation-key transaction advisory
lock before idempotency lookup, then the candidate/source transaction advisory
lock shared with approval, followed by `FOR UPDATE` on the retained run
candidate. Approval performs a no-write preflight, then reacquires its
operation and candidate locks and the retained row in the final write
transaction before it inserts the invoice and approval audit row. Therefore a
same-key retry has one durable linearization point, candidate mutations and
approval have one shared commit order, and any cancellation or exception before
commit releases transaction-scoped locks with no partial durable result. The
real-Postgres contention, serialization, stale/retry, and injected-rollback
tests exercise the representative lock waits and both commit/abort outcomes;
the stated lock/transaction invariant, rather than sampling alone, covers all
admitted orderings.

Permitted source line edits are description, rate cents, and quantity. Hourly
quantities are whole minutes and calculate a line amount with Decimal
ROUND_HALF_UP; visit/month quantities are positive integers. One explicit
credit or charge adjustment is allowed per effective revision and inherits the
existing candidate tax basis points. Recipient/delivery overrides are run-only:
Gmail accepts a valid named address, Manual Square needs no recipient, and
receipt-only delivery is not admitted. Unresolvable source blockers remain.

## Intentional

- A recurring rate/service/calendar correction remains a source-data task;
  overrides do not write canonical records or carry to a later run.
- A stale source candidate is regenerated and the override is deliberately
  reapplied or discarded; this slice does not silently transplant overrides.
- One signed adjustment line is intentionally narrower than arbitrary custom
  lines and makes credits/charges legible in invoice evidence.
- Tax is inherited from the candidate's existing tax basis points. The slice
  does not introduce a tax-policy editor.
- The provider stores an effective snapshot rather than mutating the original,
  even though that duplicates data, because it preserves financial review
  evidence and a deterministic approval artifact.

## Deferred

- Tracker proxy and Website edit/review UI are separate dependent PRs after
  this provider deploys.
- Canonical service/rate/calendar editing and any multi-line adjustment editor
  remain out of scope; the existing source workflows own recurring corrections.

Parking predicate: UI polish, expanded adjustment catalogues, source-data
editing, and reporting features that do not affect the provider's money/audit
correctness are parked in ATLAS issue #2363.

Parked hardening: none.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=<isolated local Postgres> python -m
  pytest tests/test_commercial_billing_candidate_overrides.py
  tests/test_commercial_billing_runs.py tests/test_commercial_billing_approvals.py -q`
  — **191 passed, 1 warning** against disposable local PostgreSQL.
- The complete current local equivalent of the receivables list in
  `.github/workflows/atlas_invoicing_checks.yml` was run against the same
  isolated database — **726 passed, 1 warning** — including payment/receipt,
  candidate/run/approval, Gmail/Square, repository/PDF, and canonical
  contact-email grammar paths.
- `python -m pytest tests/test_monthly_invoice_generation.py -k
  "update_invoice_clears_needs_hours_when_line_items_are_billable or
  line_items_are_billable_requires_all_positive_quantities" -q` — **2 passed,
  41 deselected**.
- `python -m pytest tests/test_invoicing_readonly_mcp.py
  tests/test_invoicing_readonly_oauth.py tests/test_invoicing_draft_writer_mcp.py
  tests/test_invoicing_draft_writer_oauth.py -q` — **43 passed**.
- `git diff --check`, `python -m py_compile` on changed Python paths, targeted
  `ruff check`, `ruff check --ignore E402 atlas_brain/main.py`, Black checks
  for the new Python files, and a YAML parse of the changed workflow all pass.
  `E402` is pre-existing in `main.py` because dotenv loading deliberately
  precedes imports.
- `bash scripts/push_pr.sh <body> -u origin HEAD` will run the required local
  mechanical review bundle once before publishing.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 174 |
| `atlas_brain/main.py` | 4 |
| `atlas_brain/services/commercial_billing_approvals.py` | 305 |
| `atlas_brain/services/commercial_billing_candidate_overrides.py` | 632 |
| `atlas_brain/services/commercial_billing_runs.py` | 506 |
| `atlas_brain/storage/migrations/382_commercial_billing_candidate_overrides.sql` | 284 |
| `atlas_brain/storage/repositories/invoice.py` | 72 |
| `plans/PR-EOM-Commercial-Billing-Candidate-Overrides.md` | 361 |
| `tests/test_commercial_billing_approvals.py` | 733 |
| `tests/test_commercial_billing_candidate_overrides.py` | 578 |
| `tests/test_commercial_billing_runs.py` | 317 |
| **Total** | **3973** |
