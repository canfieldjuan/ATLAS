# PR-EOM-Commercial-Invoice-Gmail-Drafts

## Why this slice exists

The Billing & Payments coordinating issue #2362 and H-15 in #2363 require a
website-initiated commercial invoice to become a Gmail draft only after an
explicit commercial-billing approval.  PR #2381 creates the exact draft ATLAS
invoice and PR #2382 retains one exact PDF artifact, but neither persists a
Gmail draft identity or can recover an uncertain Gmail API result.  The legacy
MCP and invoice action paths combine sending/marking-sent with delivery, so
they cannot be used by the operator workspace.

Diff-budget override: the additive draft/operation persistence, existing
Gmail API recovery lookup, authenticated provider route, and disposable
PostgreSQL failure/concurrency proof are one deployable no-send behavior.
Splitting them would either call Gmail without durable recovery evidence or
publish a durable state machine that cannot safely create/reconcile its
external draft.

### Problem-derived contract

- Root cause: ATLAS has a `GmailTransport.create_draft()` API primitive, but
  it has no durable link to an explicit EOM commercial approval/PDF artifact,
  no idempotent operation receipt, no stable message identity to find a draft
  after a process/database failure, and no guard against the legacy
  send/mark-sent paths.
- Correct fix must touch/change: add an additive ATLAS Gmail-draft/operation
  persistence contract, a no-send draft service that only accepts the
  approval's unchanged `gmail_pdf` draft invoice plus its ready retained PDF,
  and an authenticated receivables route.  It must persist intent before
  external work, attach the retained bytes, store Gmail draft/message/thread
  identifiers only after a successful response or lookup, and use a stable
  RFC Message-ID to reconcile an uncertain outcome without a second create.
  Tests must cover normal creation, same-key and new-key reuse, transport
  failure, database-confirm failure, recovery lookup, stale/non-Gmail/Square
  rejection, concurrent callers, route auth, and the actual Gmail query shape.
- Must not change: invoice amount/status/sent state, payment balances,
  candidate/run evidence, the immutable PDF artifact, Gmail sending, the
  existing MCP/manual invoice flows, CRM/service markers, Square execution,
  browser credential exposure, invoice/PDF/email product copy, or tracker and
  Website consumers.

## Scope (this PR)

Ownership lane: eom/billing-approved-gmail-drafts
Slice phase: Vertical slice

1. Add one durable no-send Gmail draft operation for an already-approved
   `gmail_pdf` commercial invoice that already has its immutable PDF artifact.
   Persist the Gmail draft/message/thread identity, recipient/subject evidence,
   actor and idempotency receipts, but never update invoice financial state.
2. Extend only the existing Gmail draft transport with a stable RFC Message-ID
   search and safe deterministic headers, then add the authenticated provider
   route and tests that prove normal/retry/failure/recovery/concurrency behavior
   without contacting a real mailbox.

### Review Contract

- Acceptance criteria:
  1. The service creates one `draft_created` Gmail draft for an explicitly
     approved, still-draft `gmail_pdf` invoice with a ready unchanged PDF
     artifact; it uses the existing invoice template and attaches the retained
     bytes, then returns durable draft/message/thread identifiers without
     changing any invoice status -- settled by the focused disposable
     PostgreSQL service tests and source assertions.
  2. A matching idempotency key returns the original operation result, while a
     fresh key for the same completed approval records a second receipt and
     reuses the same Gmail draft without another transport call; a key reused
     for another approval fails before a write -- settled by the focused tests.
  3. Intent and operation evidence are committed before the Gmail call.  If
     transport acceptance succeeds but final database confirmation fails, a
     retry searches Gmail by its exact persisted RFC Message-ID and stores the
     found draft rather than calling create again; an ambiguous or missing
     result remains visible `recovery_required`, never inferred sent -- settled
     by the fake-gateway and disposable PostgreSQL failure tests.
  4. The execution model serializes every operation request on its key and
     every approval on its durable draft row.  Across admitted interleavings,
     an approval links to at most one persisted Gmail draft identity, and a
     caller that sees another request's `creating` state may lookup/recover but
     cannot issue a second Gmail create -- settled by the real PostgreSQL
     concurrency test plus the service's transaction/transition code.
  5. Non-`gmail_pdf`, missing/invalid recipient, missing/mismatched artifact,
     changed PDF evidence, and non-draft/sent invoice inputs fail before a
     Gmail create or draft/operation insert.  Delivery eligibility comes from
     the approval invoice's retained metadata and canonical recipient
     normalizer, not a new caller-provided channel -- settled by negative
     zero-write tests.
  6. The authenticated full application route requires the existing bearer
     token, `X-EOM-Actor`, and `Idempotency-Key`; it reaches the service through
     `atlas_brain.main.app -> /api/v1 -> receivables`, returns draft metadata
     only, and does not expose PDF bytes or credentials -- settled by an ASGI
     full-mount test.
  7. The Gmail transport's draft lookup calls the documented
     `users/me/drafts` search with `q=rfc822msgid:<stable-id>`, recognizes zero
     or exactly one result, and rejects malformed/ambiguous responses; the
     tests fake only HTTP at this third-party seam.  No test authenticates to
     Gmail or creates a real customer draft.
- Reachability proof: invoke the real mounted
  `POST /api/v1/receivables/commercial-billing-approvals/{approval_id}/gmail-draft`
  route through `atlas_brain.main.app`; override only the existing receivables
  auth configuration and Gmail-draft-service dependency, then assert UUID,
  actor, and idempotency key reach the service and no attachment bytes occur in
  the response.
- Affected surfaces: `atlas_brain/api/invoicing/receivables.py`, a new
  `commercial_billing_invoice_gmail_drafts` service, the existing PDF artifact
  reader, the existing Gmail transport, additive migration 374, the explicit
  invoicing workflow enrollment, and focused service/transport/route tests.
- Risk areas: external exactly-once boundary; invoice/PDF staleness; customer
  recipient and attachment PII; mailbox recovery; no-send financial lifecycle;
  concurrent retries; migration/rollback; auth; legacy compatibility.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R13, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: authenticated receivables Gmail-draft route -> durable
  Gmail-draft service -> PostgreSQL intent/operation rows -> existing
  GmailTransport `drafts.create` / `drafts.list` third-party seam.
- Replaced-path behaviors: none; legacy MCP and invoice action
  send/approve-and-send behaviors stay callable but are deliberately not
  imported by this service.
- Guard-relevant fields: approval UUID, idempotency key, authenticated actor,
  linked invoice/approval/artifact identities, artifact hash/fingerprint,
  `draft` invoice status, retained `deliveryMethod`, canonical recipient,
  RFC Message-ID, Gmail draft/message/thread identifiers, and the closed draft
  state vocabulary.
- Caller x input shape: future tracker is the authenticated server caller;
  this provider accepts a UUID path identity and bounded headers only.  The
  browser receives no ATLAS/Gmail credentials or PDF bytes.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: existing Gmail OAuth credentials and
  `gmail.modify` scope remain the only transport configuration; this slice
  introduces no new environment variable or fallback sender.
- Explicit value probe: the default transport is injected in tests and is not
  invoked by deployment/route auth probes; fake transport tests cover valid
  create and lookup responses.
- Absent value probe: a configured transport failure leaves `retryable` or
  `recovery_required` evidence, never a sent invoice or a second create.
- Default-session/default-context probe: no production Gmail call occurs in
  tests or deployment verification; the protected route is checked only with
  no credentials to prove rejection.
- Side-effect ordering: one transaction persists a `creating` intent plus
  operation receipt before any Gmail call; final `draft_created` confirmation
  happens only after response/lookup validation.  Uncertain outcomes must
  reconcile by the persisted RFC identity before any new create is allowed.

### Closure Declaration

- **Delivery-method predicate — CLOSED / DERIVED:** only the existing exact
  `gmail_pdf` value retained in the approval invoice's metadata is admitted.
  `manual_square` and every other/missing value fail before any draft row or
  Gmail call; the service does not accept a delivery-channel argument.
- **Draft-state vocabulary — CLOSED / AUTHORED HERE:** `creating`,
  `retryable`, `recovery_required`, and `draft_created` are the only persisted
  states.  The durable row is a delivery-state record, not invoice state; no
  state means sent.
- **External result identity — CLOSED / DERIVED:** the service derives one
  RFC Message-ID from the immutable approval UUID and writes it before the
  first external call.  It accepts only exactly one Gmail Draft list result
  carrying nonblank draft/message/thread IDs; zero results are not treated as
  sent and more than one is ambiguous/recovery-required.
- **Recipient admission — CLOSED / DERIVED:** the recipient is the retained
  invoice email, normalized by the canonical EOM contact-email normalizer
  already used by candidate selection.  No route body can choose or override a
  customer address.  An absent/invalid result is a no-write conflict.
- The route idempotency key is bounded by the existing strict 1--128-character
  header schema and service validator.  It is not a free-text classifier;
  grammar-derived admission tests cover strings, padding, boundaries, and
  non-string container shapes.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py`
- `atlas_brain/services/commercial_billing_invoice_pdfs.py`
- `atlas_brain/storage/migrations/374_commercial_billing_invoice_gmail_drafts.sql`
- `atlas_brain/tools/gmail.py`
- `plans/PR-EOM-Commercial-Invoice-Gmail-Drafts.md`
- `tests/test_commercial_billing_gmail_drafts.py`

## Mechanism

The Gmail-draft service first uses the PDF artifact service's server-only read
boundary to validate the approval-to-draft-invoice link, current render
fingerprint, retained PDF bytes/hash, and invoice metadata.  It rejects an
unapproved, changed, sent, missing-PDF, non-Gmail, or invalid-recipient state
without contacting Gmail.

It then opens a short PostgreSQL transaction: lock the idempotency key, lock
the approval, insert/reuse one draft intent and one operation receipt, and
commit.  The intent has a deterministic RFC Message-ID and frozen recipient /
subject metadata.  Gmail work occurs outside that transaction.  A successful
`create_draft` response is validated and confirmed in a second short
transaction.  If confirmation fails or transport returns an uncertain result,
the next same-key request performs only `drafts.list` by that Message-ID; it
stores the one found draft or leaves explicit recovery evidence rather than
creating another draft.  Definite pre-create/rejection failures move to the
safe `retryable` state; ambiguous/missing recovery results do not.

The email body/HTML comes from the existing EOM invoice template and the
attachment is the already-retained `BYTEA`, so this slice adds neither new
customer-facing copy nor a filesystem dependency.  The service invokes only
the Gmail draft API.  It never invokes `send`, an email provider, CRM, Square,
or an invoice-status writer.

### Execution model and invariant

The first and confirmation transitions use transaction-scoped PostgreSQL
advisory locks: operation key before approval, released at commit/rollback.
The draft row's unique approval/artifact foreign keys are the durable backstop.
No transaction spans a Gmail HTTP call.  Across every admitted interleaving,
one approval can own one Gmail-draft intent and at most one confirmed Gmail
draft identity; every idempotency key maps to that intent's fixed approval
fingerprint.  Only the caller that changes `retryable` to `creating` may call
`create_draft`; callers observing `creating` or `recovery_required` may only
look up the stable RFC identity.  A crash after Gmail acceptance but before
confirmation therefore becomes a lookup/recovery operation rather than a
second-create operation.  A deleted/missing draft remains recovery evidence;
it is never sent.

## Intentional

- The service requires an existing retained PDF rather than calling the PDF
  writer itself.  This keeps PDF creation and external Gmail delivery as two
  independently retryable provider transitions and avoids generating a PDF for
  a rejected Square candidate.
- The existing EOM invoice template is reused; this accepted workspace
  behavior does not silently redesign customer email content.
- An uncertain Gmail call is not automatically retried.  Gmail's create API
  has no caller-provided idempotency key, so a second create could duplicate a
  draft.  The stable Message-ID lookup is the visible recovery path.
- The generic transport gains a draft lookup and safe extra-header support,
  not any send behavior.  Its documented Draft list query can return an
  external draft ID and message/thread IDs without treating Gmail as the
  financial ledger.
- No sent-mail reconciliation occurs here.  A Gmail draft can disappear when
  sent or deleted; deciding between those outcomes is a separate H-15 slice.

## Deferred

- H-15 / #2363: verifiable sent-mail reconciliation, final sent message ID /
  timestamp, deleted-draft classification, and recovery actions after an
  unresolved `recovery_required` state remain separate slices.
- H-15 / #2363: Manual Square queue, external Square invoice reference, and
  explicit audited `sent via Square` action remain separate provider/UI work.
- Tracker proxy and Website review/recovery UI remain dependent consumer
  slices after this provider contract deploys.
- Parking predicate: new delivery channels, Gmail mailbox background sweeps,
  invoice/email redesign, manual-reconciliation UX, generic artifact download,
  and sender/credential redesign are parked by default because they introduce
  a new state machine or product decision beyond one no-send draft creation.
  Parked hardening: none.

## Verification

- Completed locally against the isolated `atlas_receivables_test` PostgreSQL
  container: `python -m pytest -q tests/test_commercial_billing_gmail_drafts.py`
  — 75 passed (the route uses ASGI plus a fake dependency; no Gmail account or
  live financial record is contacted).
- Completed local `Atlas Invoicing Checks` equivalents: 2 approval blockers,
  417 receivables/PDF/billing regressions, and 43 MCP/OAuth regressions passed.
- Completed: targeted ruff, Python source compilation, and `git diff --check`.
- Completed: maturity-sweep unit suite (86 passed) and API/storage/Gmail
  ratchets; the Gmail ratchet passes without a baseline update, and the new
  state-machine service has no maturity findings after replacing runtime
  assertions with explicit conflict handling.
- Pending before push: final plan/body/diff-budget/AI-reconciliation/guard
  closure checks and one local PR-review wrapper run.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 9 |
| `atlas_brain/api/invoicing/receivables.py` | 69 |
| `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py` | 984 |
| `atlas_brain/services/commercial_billing_invoice_pdfs.py` | 114 |
| `atlas_brain/storage/migrations/374_commercial_billing_invoice_gmail_drafts.sql` | 110 |
| `atlas_brain/tools/gmail.py` | 212 |
| `plans/PR-EOM-Commercial-Invoice-Gmail-Drafts.md` | 291 |
| `tests/test_commercial_billing_gmail_drafts.py` | 791 |
| **Total** | **2580** |
