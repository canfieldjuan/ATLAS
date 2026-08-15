# PR-EOM-Missing-Gmail-Draft-Replacement

## Why this slice exists

H-15 remains open in [the Billing & Payments coordinating issue #2362](https://github.com/canfieldjuan/ATLAS/issues/2362): after sent-mail reconciliation proves a commercial Gmail draft is missing, the operator needs a deliberate recovery path. The deployed provider currently retains that missing observation and correctly leaves the invoice `draft`, but the original table permits only one durable draft record per approval/PDF. Recalling the ordinary create endpoint therefore returns the vanished record rather than creating an auditable replacement.

### Problem-derived contract

- Root cause: `commercial_billing_invoice_gmail_drafts` stores one mutable current Gmail identity per immutable approval/PDF (`374_commercial_billing_invoice_gmail_drafts.sql:18-80`), while reconciliation records `draft_missing` on that same row (`375_commercial_billing_invoice_gmail_sent_reconciliation.sql:14-77`). The normal create/reuse preparation returns a `draft_created` row even when reconciliation later proves the remote draft has gone (`atlas_brain/services/commercial_billing_invoice_gmail_drafts.py:351-442`). Reusing that row would either create no recovery draft or overwrite the only evidence of the missing one.
- Correct fix must touch/change: add an atomic, append-only replacement-event table plus current-generation fields; add an authenticated, idempotent `replace_missing` draft-service action and full receivables route; retain the old identity snapshot before changing the one compatibility-preserving root row; bind every replay to its immutable replacement-event generation and reject it before Gmail if the mutable root has advanced; prove normal, malformed/precondition, retry, concurrent, uncertain-create recovery, stale-invoice, stale-generation replay, and no-send behavior in the real PostgreSQL contract suite; enroll the new migration in the invoicing workflow.
- Must not change: invoices stay `draft` unless the existing sent-mail reconciler later finds verifiable `SENT` evidence; this action never calls Gmail send, MCP invoicing, CRM/service markers, Square, or payment writers; ordinary draft creation and legacy callers stay compatible; tracker/Website recovery controls, scheduled observation, and the existing cross-linked draft/PDF hardening item are not folded into this provider slice.
- Diff-budget exception: this provider recovery is one indivisible financial-safety transition, so its migration, authenticated route, idempotency/recovery state-machine extension, and real-PostgreSQL failure/concurrency proof ship together. Splitting the event/intent mutation from its no-send retry and stale-proof tests would deploy a customer-facing recovery action without the evidence required to establish that it cannot duplicate a draft or mark an invoice sent.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: vertical slice
Max files: 6

1. Permit an authenticated operator to explicitly replace only the currently durable, reconciliation-proven `draft_missing` commercial Gmail draft for an approval whose invoice remains `draft`.
2. Write an append-only snapshot/event of the missing generation and its actor/time before resetting the compatibility-preserving current root record to a new generation and a new stable RFC Message-ID.
3. Use the existing committed-intent → Gmail-draft → confirm/recover state machine for the new generation, keyed by a replacement-specific idempotency source, so retry cannot produce another Gmail draft; use the append-only event's replacement generation as the replay fence and reject a key after a later generation takes over the current root.
4. Keep the original table's one-row-per-approval/PDF uniqueness intact so an older provider revision can still read a single current root row after deployment rollback. The replacement event is additive audit history, not a second competing root row.
5. Expose the action only through the existing private full receivables boundary and add proof that it has the same authentication/actor/idempotency behavior as ordinary no-send draft creation.

### Review Contract

- Acceptance criteria:
  1. `CommercialBillingInvoiceGmailDraftService.replace_missing` admits only a `draft_created` root with `reconciliation_state == 'draft_missing'` and a still-draft approved invoice, records the predecessor snapshot/event, then calls only `create_draft`; this is settled by the real PostgreSQL normal/precondition tests in `tests/test_commercial_billing_gmail_drafts.py`.
  2. The replacement action assigns a distinct generation-scoped RFC Message-ID and leaves no `UPDATE invoices` / send path in the Gmail-draft service; this is settled by the event-row, gateway-header, and invoice-state assertions in `tests/test_commercial_billing_gmail_drafts.py`.
  3. The same replacement idempotency key returns/recoveries the one committed replacement intent only while its immutable event generation remains the current root; a replay after a later replacement rejects before Gmail. A concurrent fresh key cannot produce a second Gmail create, and a replacement refuses to race a pending Sent-mail reconciliation of the predecessor. This is settled by the retry, uncertain outcome, generation-advance, advisory-lock, and pending-reconciliation tests in `tests/test_commercial_billing_gmail_drafts.py`.
  4. The root table keeps its existing approval/PDF unique constraints while migration 377 adds only a transactional audit/event table and columns; this is settled by the migration contract test plus the local production-shape preflight recorded in #2362.
  5. `POST /api/v1/receivables/commercial-billing-approvals/{approval_id}/gmail-draft/replace-missing` requires existing receivables auth, actor, and `Idempotency-Key`, routes to the service, and never returns PDF bytes; this is settled by the full ASGI route test.
- Reachability proof: the real full FastAPI route invokes the service under existing token/actor dependencies; real PostgreSQL fixtures prove the committed intent and event state; a recording Gmail gateway proves the only external operation is no-send `create_draft` with the replacement RFC Message-ID.
- Affected surfaces: `atlas_brain/api/invoicing/receivables.py`; `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py`; migration 377; the existing Gmail-draft contract suite and invoicing workflow enrollment.
- Risk areas: financial lifecycle separation; missing-versus-sent proof; idempotency/retry; concurrent replacement attempts; migration atomicity and rollback compatibility; credential boundary; PII-free logs/responses.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R13, R14.

### Boundary-change enumeration

- Boundary path/seam: the new authenticated `POST .../gmail-draft/replace-missing` handler → `replace_missing` service method → committed current-generation intent/event → Gmail Drafts create/lookup/confirm.
- Replaced-path behaviors: `create_or_reuse` remains the ordinary first-draft/reuse path and must not auto-replace a missing draft; sent-mail reconciliation remains the only route that can move an invoice to `sent`.
- Guard-relevant fields: root draft `state`, durable `reconciliation_state`, pending Sent-mail reconciliation operations, approval/invoice identity, invoice lifecycle fields, replacement idempotency key/source, authenticated actor, current RFC Message-ID, and generation counter.
- Caller x input shape: existing tracker/Website calls continue to use ordinary `gmail-draft`; only an authenticated operator calling the new UUID-path route with a nonblank `Idempotency-Key` and actor can request replacement. All absent/malformed/stale/non-missing combinations return validation/conflict before Gmail creation.

### Closure declaration

- Eligibility status set: **CLOSED**. Membership comes from the existing persisted state vocabularies in migrations 374/375 plus the invoice lifecycle values checked by the service; the new route admits only the conjunctive `draft_created` + `draft_missing` + invoice `draft` predicate. Any unrecognized, inactive, malformed, or future status takes the safer conflict path with no Gmail call because creating another customer-facing draft is the expensive error.
- Operation sources: **CLOSED**. The normal source remains `_DRAFT_SOURCE = 'eom_admin'`; this slice introduces one authored replacement source used only by `replace_missing`. A source outside those explicit service queries cannot be replayed as either ordinary creation or replacement, which safely rejects cross-operation idempotency reuse.
- Generation set: **OPEN positive integers**, derived from the current root's stored generation at the locked replacement transition. A missing/non-positive/non-matching generation fails closed before an event or Gmail call; this avoids enumerating retry/replacement count while preserving a single current root record.

### Deployed-config probing

- Deployed/default config values: no new flag, credential, mailbox scope, scheduler, or service account is added; the action uses the already-deployed private receivables/Gmail draft transport boundary.
- Explicit value probe: read-only production preflight at `6340b1cc` found zero root draft rows, zero missing rows, zero sent rows, and the existing approval/artifact unique constraints.
- Absent value probe: unchanged Gmail transport failure continues to record `retryable` or `recovery_required` on the committed replacement generation and returns 503/409; it never falls through to send or an unrecorded second create.
- Default-session/default-context probe: the existing full app route uses `require_receivables_api` and `require_actor`; unauthenticated/no-actor/no-key ASGI cases remain 401/422 before service invocation.
- Side-effect ordering: the event plus replacement intent commits before `create_draft`; any later Gmail failure leaves durable recovery state. No invoice lifecycle write is in this service.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py`
- `atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql`
- `plans/PR-EOM-Missing-Gmail-Draft-Replacement.md`
- `tests/test_commercial_billing_gmail_drafts.py`

## Mechanism

Migration 377 uses `-- atlas: atomic-bookkeeping` because it couples the event table, root generation columns, and bookkeeping. It leaves the original root table's approval/PDF unique constraints in place. On a valid replacement, the service locks the approval and replacement idempotency key, snapshots the exact prior root into an append-only event row, increments the root's generation, resets only its **current delivery projection** to `creating` with a generation-scoped RFC Message-ID, and writes a replacement-source operation. The database transaction commits before the existing Gmail Drafts call.

The normal create/reuse code remains the transport/recovery engine for that current root: a definite Gmail rejection becomes retryable; an uncertain result is searched by the newly persisted RFC Message-ID; confirmation saves one Gmail draft/message/thread identity. The immutable replacement event is one-to-one with its replacement operation and records its target generation; replay first compares that target to the mutable root and fails closed if a later replacement advanced it. A fresh concurrent key cannot pass the missing-draft predicate after the first committed transition, and a pending Sent-mail recheck blocks replacement until that proof attempt completes.

The prior missing identity is not deleted or silently discarded: the event snapshot records its root ID, prior/new generation, full durable draft/reconciliation state, actor, timestamp, and replacement operation. An older provider revision continues to see one root record rather than encountering duplicate approval/PDF rows; because it cannot derive a later generation's RFC Message-ID, it fails closed instead of issuing a duplicate draft.

## Intentional

- A replacement is explicit only. `create_or_reuse` does not infer that `draft_missing` authorizes a new customer-visible draft.
- The root record is a current-delivery projection, while migration 377's event table is the append-only identity/history ledger. This preserves the existing one-row compatibility contract without losing the missing generation.
- A replacement gets a new generation-scoped RFC Message-ID; it does not reuse the missing draft's mailbox identity.
- A stale replacement idempotency key is rejected rather than reconstructed against a later current root; the append-only event is the durable generation claim for that key.
- The service does not query or send Sent mail during replacement. The existing reconciliation path remains the sole proof-gated invoice-sent writer; a stale invoice/non-missing reconciliation state rejects the action.
- No UI copy or operator interaction shape is decided here. Tracker and Website will consume the deployed provider action in a separate slice.

## Deferred

- H-15 tracker/Website recovery action and history presentation after this provider is deployed.
- H-15 scheduled mailbox polling/webhook reconciliation remains an operations/credential slice.
- The #2386 cross-linked approval/PDF draft hardening remains deferred: it needs its own production-data assessment and must not be disguised as part of generation recovery.
- Append-only Square-reference correction/reversal remains a separate Manual Square hardening item.

Parking predicate: unrelated UI/product-shape work, scheduler/webhook operations, Gmail configuration, and cross-link schema hardening are parked unless they directly prevent the replacement from retaining missing evidence or safely avoiding a duplicate draft.

Parked hardening: none.

## Verification

- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest -q tests/test_commercial_billing_gmail_drafts.py -k stale_generation_replay` — 1 passed, 120 deselected.
- `ATLAS_RECEIVABLES_TEST_DATABASE_URL=... python -m pytest -q tests/test_commercial_billing_gmail_drafts.py` — 121 passed.
- Local equivalent of the invoicing workflow receivables/repository group — 587 passed; MCP/OAuth group — 43 passed; migration runner — 30 passed, 1 skipped.
- Scoped Ruff, Python compile, and `git diff --check` passed.
- Pinned redacted Gitleaks scan of `origin/main..HEAD` with the trusted-base baseline — 1 commit scanned, no leaks. Local `check_diff_budget.py` accepts the explicit indivisibility override.
- `bash scripts/push_pr.sh .tmp/h15-pr-body.md --force-with-lease -u origin HEAD` will run the repository local-review/unit gate on the exact final commit before publication; current-head thread reconciliation follows publication.
- Pre-implementation production evidence: `BEGIN TRANSACTION READ ONLY` inspection of the live `atlas` database at the #2387 runtime returned 0 root draft/missing/sent rows and confirmed the existing unique-constraint names; it made no data change.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/api/invoicing/receivables.py` | 25 |
| `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py` | 670 |
| `atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql` | 67 |
| `plans/PR-EOM-Missing-Gmail-Draft-Replacement.md` | 117 |
| `tests/test_commercial_billing_gmail_drafts.py` | 629 |
| **Total** | **1510** |
