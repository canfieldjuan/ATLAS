# PR-EOM-Approved-Invoice-PDF-Artifact

## Why this slice exists

ATLAS PR #2381 deliberately stops after an explicit commercial-candidate
approval creates one draft invoice.  It creates no PDF because the only current
PDF callers are the legacy monthly/MCP flows, which combine disk writes with
service markers, CRM activity, email, and/or sent state.  H-15 in the Billing &
Payments deferred issue requires a durable, retry-safe PDF artifact before a
later no-send Gmail-draft slice can attach one.

Diff-budget override: The additive retention schema, transaction-locked exact
byte writer, authenticated real route, and disposable-PostgreSQL
render/failure/retry proof are one independently deployable provider behavior.
Splitting them would publish either a PDF route without durable recovery or
binary financial evidence without a safe authenticated operation boundary.

### Problem-derived contract

- Root cause: ATLAS has a pure in-memory `render_invoice_pdf()` renderer, but
  no durable artifact identity or operation receipt for an invoice created by
  an explicit commercial-billing approval.  Reusing the legacy callers would
  introduce delivery and financial side effects into PDF generation.
- Correct fix must touch/change: add one additive ATLAS artifact/operation
  persistence contract, an exact approved-invoice PDF service, and an
  authenticated receivables route.  It must lock/reconcile a request identity,
  render only the linked EOM draft invoice, persist bytes plus immutable
  render-input/hash evidence atomically, and expose only artifact metadata.
  Tests must prove creation, unchanged replay, recovery after renderer or
  transaction failure, same-approval reuse, changed-invoice rejection, and
  route authentication.
- Must not change: invoice content/branding, the existing renderer, legacy
  MCP/monthly paths, invoice status, payment balances, billing-run candidate
  evidence, Gmail drafts, email sending, CRM/service markers, Square state,
  browser credentials, or any tracker/Website consumer.

## Scope (this PR)

Ownership lane: eom/billing-approved-gmail-drafts
Slice phase: Vertical slice

1. Add a durable `invoice_pdf` artifact with immutable SHA-256 and render-input
   fingerprint, plus a separate idempotency/actor operation receipt keyed to
   the already-approved invoice.
2. Add `POST /receivables/commercial-billing-approvals/{approval_id}/invoice-pdf`.
   The existing server-to-server token, actor header, and idempotency header are
   required.  The response exposes metadata only; no PDF bytes or delivery
   action cross the browser boundary.
3. Generate a PDF only for the approval's `eom_commercial_billing` invoice
   while it remains `draft`; replay returns the same artifact, and a fresh
   request refuses a changed or non-draft invoice instead of silently attaching
   stale money/customer data later.
4. Keep failures recoverable: failed render or failed artifact/operation
   insertion commits neither row, so the same key can retry; a committed
   operation returns the original artifact without rendering again.

### Review Contract

- Acceptance criteria:
  1. The service's advisory-lock transaction creates at most one artifact for
     one approval, byte-hash/fingerprint evidence matches the renderer input,
     and a same-key replay returns the original artifact without calling the
     renderer again -- settled by
     `tests/test_commercial_billing_approvals.py` service tests.
  2. A failed renderer and an injected artifact/operation insert failure leave
     no durable artifact or operation; retrying the identical request can
     create exactly one ready artifact -- settled by the focused test and its
     disposable PostgreSQL rollback assertion.
  3. A new operation key may reuse an unchanged artifact without rerendering,
     while changed render-input evidence or a non-draft invoice returns a
     conflict before any artifact/operation insert -- settled by the focused
     test's zero-write assertions.
  4. The authenticated FastAPI route requires the existing bearer token,
     `X-EOM-Actor`, and `Idempotency-Key`; the real full-application mount
     (`atlas_brain.main.app -> /api/v1 -> receivables router`) invokes the
     service and exposes artifact metadata only -- settled by the ASGI route
     test with only the auth configuration and service dependency overridden.
     The bounded idempotency-key validator is also checked against a
     grammar-derived tokens/wrappers/families oracle without a database call.
  5. The new service imports neither Gmail, email transport, the legacy monthly
     task, CRM writer, nor service-marker writer, and it never updates invoice
     status -- settled by the import/source contract test and SQL assertions.
  6. The migration is additive, restricts one artifact to one approval (and
     therefore its approval's unique invoice), keeps its references on
     `ON DELETE RESTRICT`, and stores no
     failed/delivery/sent state -- settled by migration contract and disposable
     PostgreSQL tests.
- Reachability proof: invoke the real mounted
  `POST /api/v1/receivables/commercial-billing-approvals/{approval_id}/invoice-pdf`
  route through `atlas_brain.main.app`; override only the existing receivables
  auth configuration and PDF-service dependency, then assert the service
  receives UUID, actor, and idempotency key and the response has no PDF-byte
  field.
- Affected surfaces: `atlas_brain/api/invoicing/receivables.py`, a new
  `commercial_billing_invoice_pdfs` service, additive migration 373, the
  existing `commercial_billing_candidate_approvals`/`invoices` tables, and the
  existing explicit invoicing workflow test enrollment.
- Risk areas: financial evidence linkage; exact cents/render input; durable
  idempotency; partial transaction failures; non-draft/stale invoice reuse;
  PII-bearing binary storage; auth; accidental delivery imports; additive
  mixed-version deployment.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: authenticated receivables PDF-artifact route ->
  `CommercialBillingInvoicePDFService.generate_or_reuse()` -> PostgreSQL
  artifact/operation rows -> existing in-process renderer.
- Replaced-path behaviors: none; legacy MCP/monthly file-write/send paths
  remain callable and are deliberately not used.
- Guard-relevant fields: approval UUID, idempotency key, authenticated actor,
  exact approval/invoice linkage, `draft` status, renderer bytes, content hash,
  and render-input fingerprint.
- Caller x input shape: tracker is the future authenticated server caller;
  this provider accepts UUID path identity plus bounded headers only.  Browser
  clients do not receive ATLAS credentials or bytes.

### Deployed-config probing

N/A - no new environment/config fallback.  The artifact is PostgreSQL-owned
instead of using the legacy configurable filesystem path, which avoids an
unverified per-host storage dependency.  Deployment verification will use the
existing protected route only with an unauthenticated rejection probe and a
read-only schema/health check; it will not generate a real customer PDF.

### Closure Declaration

- **Approval/invoice state predicate — CLOSED / DERIVED:** allowed generation
  membership is the existing approval row linked by foreign key to an invoice
  whose source is `eom_commercial_billing` and whose status is exactly `draft`.
  The service derives it from the joined database row rather than copying a
  caller-provided vocabulary.  Any absent, mismatched, or non-draft member is
  rejected before renderer or insert work because a stale or sent financial
  artifact is costlier than a recoverable operator conflict.
- **Artifact kind and MIME vocabulary — CLOSED / AUTHORED HERE:** this migration
  creates only `invoice_pdf` / `application/pdf`; no unlisted artifact kind is
  admitted.  The outside default is reject/no-write, preventing this PDF slice
  from becoming a generic delivery state machine.
- **Render-evidence field inventory — CLOSED / DERIVED:** the fingerprint is
  built from precisely the invoice fields read by
  `atlas_brain/services/invoice_pdf.py::render_invoice_pdf`; the service's
  snapshot builder is the single derived choke point.  Any changed derived
  fingerprint makes a fresh reuse request fail closed rather than silently
  returning a PDF whose customer, line, total, date, or status no longer
  matches ATLAS.
- The renderer's bytes are in-process output, not an open user-input admission
  classifier.  A bounded `%PDF-`/EOF/size sanity check only prevents persisting an
  obviously invalid renderer result; it never chooses an external delivery
  destination.  No trigger-A open-input classifier is introduced.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_invoice_pdfs.py`
- `atlas_brain/storage/migrations/373_commercial_billing_invoice_pdf_artifacts.sql`
- `plans/PR-EOM-Approved-Invoice-PDF-Artifact.md`
- `tests/test_commercial_billing_approvals.py`

## Mechanism

The service resolves the immutable approval-to-invoice foreign-key link inside
one PostgreSQL transaction.  It takes an advisory lock for the idempotency key,
then an approval lock.  A matching operation receipt returns its artifact;
otherwise a matching artifact is reused only when the current derived render
fingerprint still matches.  When no artifact exists, the service validates the
draft invoice, invokes the existing exact-cent renderer inside the transaction,
validates bounded PDF bytes, hashes them, inserts one immutable `BYTEA` row and
one operation receipt, and returns metadata.

The artifact and operation inserts share the same transaction.  Thus a database
failure rolls both back, while an earlier renderer failure has no persistent
effect and can be retried with the unchanged key.  Artifact bytes stay in the
ATLAS financial database for future server-side Gmail attachment work; this
slice adds neither a public download nor a delivery integration.

### Execution model and invariant

Every request executes in one PostgreSQL transaction.  It first takes a
transaction-scoped advisory lock for its idempotency key and then one for the
approval; PostgreSQL releases both locks on commit, rollback, cancellation, or
connection loss.  The approval's unique artifact constraint is the durable
backstop.  Across every interleaving admitted by those primitives, a committed
approval has at most one artifact, each committed operation refers to that
artifact, and a `(source, idempotency_key)` maps to exactly one approval
fingerprint.  A crash or cancellation before commit leaves neither newly
inserted row; after commit both rows are visible.  Same-key callers therefore
replay one committed result, while concurrent different-key callers serialize
on the approval and one reuses the finished artifact.  The real-PostgreSQL
concurrency test holds the first renderer invocation inside its transaction,
proves the second caller reaches and waits at the approval lock, then verifies
one artifact, two receipts, and one render after release.

## Intentional

- PostgreSQL `BYTEA`, rather than the legacy `auto_invoice_save_path`, is the
  durable artifact store: a per-worktree filesystem path cannot prove reuse or
  survive provider cutovers, and it would not give a later Gmail writer one
  audited source of attachment bytes.
- A same-key replay returns its committed original even if someone later edits
  the invoice; a new operation key detects that change and fails closed.  This
  preserves idempotency while preventing a new delivery attempt from quietly
  using obsolete financial content.
- No status transition is added to `commercial_billing_candidate_approvals`.
  The immutable artifact row is the PDF state; changing the approval lifecycle
  now would entangle later Gmail, sent-mail, and Square recovery states.
- Manual-Square draft invoices may have a PDF artifact because it is a private
  ATLAS document action, not a delivery-channel inference.  This slice creates
  no Square invoice/reference or sent state.

## Deferred

- H-15 / #2363: no-send Gmail draft identity, attachment recovery state,
  verified sent-mail reconciliation, and manual Square sent action remain
  separate provider slices.
- An authenticated artifact download/read UI is deferred until the Website's
  operator workflow has an accepted product-shape contract.  The future Gmail
  provider can retrieve bytes server-side without exposing them to the browser.
- Parking predicate: generic artifact formats, external object storage,
  delivery-channel semantics, post-send correction workflows, and product PDF
  redesign are parked by default.  They require another state machine or a
  customer/operator-visible decision and are not needed to prove one durable
  ATLAS PDF artifact.  Parked hardening: `HARDENING.md` "Isolate receivables
  config failures from unrelated public routes" remains parked because it is a
  pre-existing full-app startup-isolation concern, not an artifact behavior or
  regression caused by this slice.

## Verification

- `python -m compileall -q atlas_brain/services/commercial_billing_invoice_pdfs.py
  atlas_brain/api/invoicing/receivables.py tests/test_commercial_billing_approvals.py`
  -- passed locally.
- `python3.11 -m compileall -q atlas_brain/services/commercial_billing_invoice_pdfs.py
  atlas_brain/api/invoicing/receivables.py tests/test_commercial_billing_approvals.py`
  -- passed locally (workflow interpreter syntax check).
- ruff check `atlas_brain/services/commercial_billing_invoice_pdfs.py`,
  `atlas_brain/api/invoicing/receivables.py`, and
  `tests/test_commercial_billing_approvals.py` -- passed locally.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest -q
  tests/test_commercial_billing_approvals.py` -- 78 passed using disposable
  PostgreSQL 16 schemas; no live financial or email state was touched.
- The exact three commands in `.github/workflows/atlas_invoicing_checks.yml`
  passed locally: 2 approval-blocker tests (41 deselected), the nine-file
  invoicing suite (exit 0), and the 43 MCP/OAuth surface tests.  The installed
  local test stack is Python 3.13; the separate Python 3.11 compile check above
  confirms source syntax but Python 3.11's pytest dependencies are not locally
  installed.
- `python scripts/maturity_sweep.py atlas_brain/api ...` with the workflow's
  baseline and sensitive globs -- passed: `ratchet gate passed: no new
  brittleness above baseline`.
- Final local plan, diff-budget, reconciliation, and guard-closure checks passed
  before push; the repository local-review wrapper remains the final pre-push
  mechanical gate and records its receipt in the PR body.
- Deployment: protected-route unauthenticated rejection plus read-only schema,
  process cwd, and health checks only; no customer invoice, PDF, Gmail draft,
  email, payment, or service marker will be created as a probe.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 4 |
| `atlas_brain/api/invoicing/receivables.py` | 59 |
| `atlas_brain/services/commercial_billing_invoice_pdfs.py` | 592 |
| `atlas_brain/storage/migrations/373_commercial_billing_invoice_pdf_artifacts.sql` | 65 |
| `plans/PR-EOM-Approved-Invoice-PDF-Artifact.md` | 268 |
| `tests/test_commercial_billing_approvals.py` | 576 |
| **Total** | **1564** |
