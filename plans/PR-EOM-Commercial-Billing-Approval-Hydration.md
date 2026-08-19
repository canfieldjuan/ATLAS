# PR-EOM-Commercial-Billing-Approval-Hydration

## Why this slice exists

Website PR #219's post-merge review found that reopening a saved commercial
billing run clears the portal's page-local approval map.  The page can then
offer a candidate adjustment even though ATLAS has already created the
candidate's immutable draft invoice.  The user asked to address that thread in
the EOM Billing & Payments lane.  ATLAS owns approvals and invoices, so the
durable state must be exposed by the existing saved-run reader before the
Website consumes it through the already-deployed tracker proxy.

### Problem-derived contract

- Root cause: `CommercialBillingRunService._run_view` returns source/effective
  candidate evidence and review decisions but not the durable
  `commercial_billing_candidate_approvals` record.  A browser reload therefore
  has no ATLAS-owned evidence with which to distinguish an unapproved candidate
  from an already-invoiced one.
- Correct fix must touch/change: the saved-run reader must project one matching
  durable approval and its linked invoice for the candidate's exact
  candidate/source/review identity; the projection must also find an approval
  that ATLAS idempotently reused from an equivalent run.  Contract tests must
  prove the normal no-approval read, same-run approval read, equivalent-run
  reuse, and a mismatched review identity failing closed.  A real authenticated
  `/api/v1/receivables/commercial-billing-runs/{id}` read must expose the same
  data without a financial or delivery write.
- Must not change: no migration, approval/invoice writer, invoice lifecycle,
  review/override writer, Gmail/Square behavior, customer-visible invoice/PDF
  content, or tracker authentication is changed.  The existing saved-run JSON
  remains compatible; `candidate.approval` is additive and may be `null`.

## Scope (this PR)

Ownership lane: eom-billing-payments
Slice phase: Production hardening
Max files: 4

1. Add the read-only durable approval projection to each retained commercial
   billing candidate returned by ATLAS.
2. Preserve the projection's exact identity fence and linked invoice evidence,
   including an equivalent-run idempotency reuse.
3. Add service and full API-entrypoint regression proof.  The Website-side
   validation/rendering repair remains the dependent follow-up PR.

### Review Contract

- Acceptance criteria:
   - [ ] Every saved candidate has an additive `approval: null` when ATLAS has
     no matching approval; settled by the real-Postgres service test in
     `tests/test_commercial_billing_runs.py`.
   - [ ] A matching durable approval projects its immutable approval identity
     and linked invoice identity/amount into the saved candidate; settled by
     the real-Postgres saved-run tests in
     `tests/test_commercial_billing_approvals.py`.
   - [ ] The reader joins candidate key and source fingerprint, then proves the
     approval's review fingerprint against the current effective candidate, so
     an approval from an equivalent run is visible but a mismatched review
     identity fails the saved-run read closed; settled by the equivalent-run and
     mismatch test cases in
     `tests/test_commercial_billing_approvals.py`.
   - [ ] The reader performs no invoice, approval, override, review-decision,
     Gmail, or Square write; settled by the before/after row-count assertions
     in `tests/test_commercial_billing_approvals.py`.
   - [ ] Existing commercial billing service and API behavior remains green;
     settled by the focused invoicing CI-equivalent command and static checks
     recorded below.
- Reachability proof: an authenticated in-process request through
  `atlas_brain.main.app` to the real `/api/v1/receivables/commercial-billing-runs/{id}`
  route returns the persisted approval projection, without mutating the test
  schema.
- Affected surfaces: `CommercialBillingRunService` saved-run read model; the
  existing receivables GET route; persisted approval/invoice joins; the Website
  consumer contract via the existing tracker proxy.
- Risk areas: financial lifecycle interpretation, equivalent-run idempotency,
  read-model backward compatibility, stale/malformed identity evidence, and
  mixed provider deployment.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R7, R8, R10, R12, R13, R14.

### Boundary-change enumeration

The saved-run projection is a CLOSED, DERIVED identity boundary.  Membership is
derived from durable approval rows rather than an enumerated list: matching
rows have the candidate's key and source fingerprint, then must prove the
active effective-review fingerprint.  No matching approval row produces
`approval: null`; a persisted row with a mismatched identity fails the
saved-run read closed.

- Boundary path/seam: `CommercialBillingRunService._run_view` -> the existing
  authenticated saved-run GET route.
- Replaced-path behaviors: retained candidates previously exposed only
  source/effective review evidence; they now also expose one matched durable
  approval or `null`.
- Guard-relevant fields: approval candidate key, source fingerprint, review
  fingerprint, approval state, linked invoice source/context, and linked invoice
  identity/amount.
- Caller x input shape: no browser-supplied approval fields; a UUID route path
  selects a persisted run and the reader derives the projection from ATLAS.

### Deployed-config probing

N/A - no config, environment fallback, or credential behavior changes.  The
reader uses the existing ATLAS database pool and receives no external-service
credentials.

### Files touched

- `atlas_brain/services/commercial_billing_runs.py`
- `plans/PR-EOM-Commercial-Billing-Approval-Hydration.md`
- `tests/test_commercial_billing_approvals.py`
- `tests/test_commercial_billing_runs.py`

## Mechanism

`_run_view` already materializes every retained candidate in one query and
derives its active effective-review fingerprint from the latest append-only
override.  Extend that query with one lateral, bounded approval/invoice join.
The join deliberately has no request-side input and matches durable candidate
and source identity; the reader then proves the active review fingerprint
before returning the projection.  It returns an approval view only when ATLAS
can prove the invoice belongs to the EOM commercial billing source and business
context.  The response decorates the existing candidate with `approval`,
leaving its source snapshot, effective candidate, review decision, and invoice
lifecycle untouched.

## Intentional

- Add an additive field to the existing saved-run response instead of a second
  approval-list URL: reopening a review already loads this response, so a
  separate reader would add race/loading state without improving authority.
- Match an equivalent-run approval by exact candidate/source/review identity,
  not only the current run id: ATLAS deliberately reuses the financial approval
  for that identity and the UI must not offer a duplicate path.
- Do not project Gmail or Square delivery artifacts: they belong to their own
  recovery readers and are not needed to lock a financial approval.
- Do not add a migration: all durable fields already exist, and this is a
  backwards-compatible read-model addition.

## Deferred

- The dependent Website PR validates the new read evidence, rejects malformed
  stored override relationships, binds override responses to submitted change
  sets, and updates the operator guide.  It is held until this provider change
  is deployed.
- The tracker proxy already forwards this GET response as untyped JSON; no
  tracker change is required.  A future typed response schema, if adopted, must
  preserve this additive field.

Parking predicate: implementation-polish and delivery-artifact hydration that
does not affect whether an approved candidate can be adjusted or re-approved
are parked.  A discovered financial identity, authorization, duplicate-invoice,
or data-loss defect remains in scope.

## Verification

- Passed locally before push (Python 3.13; hosted workflow will repeat these
  under its pinned Python 3.11):
  - Ruff and compileall passed for changed Python;
  - the extended billing regression command with its real local Postgres
    fixture: `509 passed` in 83.80s, including saved-run, approval, receipt,
    Gmail-draft, manual-Square, repository, and PDF coverage;
  - the exact three Atlas invoicing workflow commands from
    `.github/workflows/atlas_invoicing_checks.yml`: `2 passed, 41 deselected`
    for the monthly blocker proof; `726 passed` in 92.49s for the receivables
    ledger/repository proof; and `43 passed` for the MCP/OAuth proof;
  - `git diff --check`.
- Pending before push: diff-budget, audit-format, and the one local PR review
  run through `scripts/push_pr.sh`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/services/commercial_billing_runs.py` | 153 |
| `plans/PR-EOM-Commercial-Billing-Approval-Hydration.md` | 178 |
| `tests/test_commercial_billing_approvals.py` | 58 |
| `tests/test_commercial_billing_runs.py` | 5 |
| **Total** | **394** |
