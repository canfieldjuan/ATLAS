# PR-EOM-Missing-Gmail-Draft-Replacement

## Why this slice exists

H-15 remains open in [the Billing & Payments coordinating issue #2362](https://github.com/canfieldjuan/ATLAS/issues/2362): after sent-mail reconciliation proves a commercial Gmail draft is missing, the operator needs a deliberate recovery path. The deployed provider currently retains that missing observation and correctly leaves the invoice `draft`, but the original table permits only one durable draft record per approval/PDF. Recalling the ordinary create endpoint therefore returns the vanished record rather than creating an auditable replacement.

### Problem-derived contract

- Root cause: `commercial_billing_invoice_gmail_drafts` stores one mutable current Gmail identity per immutable approval/PDF (`374_commercial_billing_invoice_gmail_drafts.sql:18-80`), while reconciliation records `draft_missing` on that same row (`375_commercial_billing_invoice_gmail_sent_reconciliation.sql:14-77`). The normal create/reuse preparation returns a `draft_created` row even when reconciliation later proves the remote draft has gone (`atlas_brain/services/commercial_billing_invoice_gmail_drafts.py:351-442`). Reusing that row would either create no recovery draft or overwrite the only evidence of the missing one.
- Correct fix must touch/change: add an atomic, append-only replacement-event table plus current-generation fields; add an authenticated, idempotent `replace_missing` draft-service action and full receivables route; retain the old identity snapshot before changing the one compatibility-preserving root row; bind ordinary, replacement, and Sent-mail reconciliation operations to their immutable generation and reject stale replays/finalizations before Gmail or an invoice write; guard every asynchronous draft confirmation/recovery transition by the prepared generation; let a definitive Gmail rejection restore `retryable` even if an overlapping same-key recovery lookup first records `recovery_required`; allow a later completed same-generation reconciliation to supersede only an expired pending claim; make all invoice writes share the approval lock and take the sent-reconciliation invoice row before that lock; prove normal, malformed/precondition, retry, concurrent, uncertain-create recovery, stale-invoice, stale-generation replay/finalization, abandoned-claim recovery, lock ordering, invoice-writer fencing, and no-send behavior in the real PostgreSQL contract suite; enroll the new migration in the invoicing workflow.
- Must not change: invoices stay `draft` unless the existing sent-mail reconciler later finds verifiable `SENT` evidence; this action never calls Gmail send, MCP invoicing, CRM/service markers, Square, or payment writers; ordinary draft creation and legacy callers stay compatible; tracker/Website recovery controls, scheduled observation, and the existing cross-linked draft/PDF hardening item are not folded into this provider slice.
- Diff-budget exception: this provider recovery is one indivisible financial-safety transition, so its migration, authenticated route, idempotency/recovery state-machine extension, and real-PostgreSQL failure/concurrency proof ship together. Splitting the event/intent mutation from its no-send retry and stale-proof tests would deploy a customer-facing recovery action without the evidence required to establish that it cannot duplicate a draft or mark an invoice sent.

## Scope (this PR)

Ownership lane: eom/billing-payments
Slice phase: vertical slice
Max files: 7

1. Permit an authenticated operator to explicitly replace only the currently durable, reconciliation-proven `draft_missing` commercial Gmail draft for an approval whose invoice remains `draft`.
2. Write an append-only snapshot/event of the missing generation and its actor/time before resetting the compatibility-preserving current root record to a new generation and a new stable RFC Message-ID.
3. Use the existing committed-intent → Gmail-draft → confirm/recover state machine for the new generation, keyed by a replacement-specific idempotency source, so retry cannot produce another Gmail draft; persist every ordinary/replacement/reconciliation operation's target generation and reject it after a later generation takes over the current root. Pass the prepared generation through every asynchronous draft confirmation/recovery transition and Sent-mail finalization. A definitive no-create result wins over an overlapping same-key recovery lookup and restores the same current generation to `retryable`.
4. Let a completed same-generation Sent-mail reconciliation supersede only pending claims older than the durable five-minute lease, retaining the old key, authenticated supersession actor, and timestamp. This makes a crashed lookup recoverable without disrupting concurrent healthy keys.
5. At the database boundary, make every invoice mutation share the existing approval advisory lock and reject it while a replacement generation is `creating`, `retryable`, or `recovery_required`. Sent reconciliation first locks the invoice row and then the shared approval lock, matching the trigger's order and preserving the approved PDF/invoice lifecycle evidence from legacy writers through the external no-send call.
6. Keep the original table's one-row-per-approval/PDF uniqueness intact so an older provider revision can still read a single current root row after deployment rollback. The replacement event is additive audit history, not a second competing root row.
7. Expose the action only through the existing private full receivables boundary and add proof that it has the same authentication/actor/idempotency behavior as ordinary no-send draft creation.

### Review Contract

- Acceptance criteria:
  1. `CommercialBillingInvoiceGmailDraftService.replace_missing` admits only a `draft_created` root with `reconciliation_state == 'draft_missing'` and a still-draft approved invoice, records the predecessor snapshot/event, then calls only `create_draft`; this is settled by the real PostgreSQL normal/precondition tests in `tests/test_commercial_billing_gmail_drafts.py`.
  2. The replacement action assigns a distinct generation-scoped RFC Message-ID and leaves no `UPDATE invoices` / send path in the Gmail-draft service; this is settled by the event-row, gateway-header, and invoice-state assertions in `tests/test_commercial_billing_gmail_drafts.py`.
  3. Every ordinary, replacement, and Sent-mail reconciliation idempotency key is bound to one persisted generation. A replay/finalization after a later replacement rejects before Gmail, a completion/recovery transition cannot touch a new generation, and an expired pending reconciliation claim is superseded only by a later completed observation of that same generation. A concurrent healthy key remains live. This is settled by the retry, uncertain outcome, generation-advance, stale-finalization, expired-claim, and duplicate-execution real PostgreSQL tests in `tests/test_commercial_billing_gmail_drafts.py`.
  4. An invoice status or render-field writer started after replacement preparation cannot commit through the pending replacement claim; the database trigger takes the same approval advisory lock and rejects while the replacement generation is unresolved. A sent-mail finalization locks the invoice row before that advisory lock, so it cannot deadlock with that trigger. This is settled by `test_missing_gmail_draft_replacement_blocks_invoice_mutations_until_creation` and `test_sent_reconciliation_locks_invoice_before_approval_to_avoid_trigger_cycle` using two real PostgreSQL connections.
  5. The root table keeps its existing approval/PDF unique constraints while migration 377 adds only transactional audit/event/fence bookkeeping and never writes an invoice sent lifecycle value; this is settled by the migration contract test plus the local production-shape preflight recorded in #2362.
  6. `POST /api/v1/receivables/commercial-billing-approvals/{approval_id}/gmail-draft/replace-missing` requires existing receivables auth, actor, and `Idempotency-Key`, routes to the service, and never returns PDF bytes; this is settled by the full ASGI route test.
- Reachability proof: the real full FastAPI route invokes the service under existing token/actor dependencies; real PostgreSQL fixtures prove the committed intent and event state; a recording Gmail gateway proves the only external operation is no-send `create_draft` with the replacement RFC Message-ID.
- Affected surfaces: `atlas_brain/api/invoicing/receivables.py`; `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py`; `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py`; migration 377; the existing Gmail-draft contract suite and invoicing workflow enrollment.
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
- Generation set: **OPEN positive integers**, derived from the current root's stored generation at the locked operation/replacement transition. Every operation persists the claimed value; a missing/non-positive/non-matching generation fails closed before an event, Gmail call, or invoice write. This avoids enumerating retry/replacement count while preserving a single current root record.

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
- `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py`
- `atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql`
- `plans/PR-EOM-Missing-Gmail-Draft-Replacement.md`
- `tests/test_commercial_billing_gmail_drafts.py`

## Mechanism

Migration 377 uses `-- atlas: atomic-bookkeeping` because it couples the event table, root generation columns, and bookkeeping. It leaves the original root table's approval/PDF unique constraints in place. On a valid replacement, the service locks the approval and replacement idempotency key, snapshots the exact prior root into an append-only event row, increments the root's generation, resets only its **current delivery projection** to `creating` with a generation-scoped RFC Message-ID, and writes a replacement-source operation. The database transaction commits before the existing Gmail Drafts call.

The normal create/reuse code remains the transport/recovery engine for that current root: a definite Gmail rejection becomes retryable even when an overlapping duplicate request first logged recovery-required; an uncertain result is searched by the newly persisted RFC Message-ID; confirmation saves one Gmail draft/message/thread identity. Every ordinary and replacement operation persists its target generation, and the prepared generation is carried through retry claims, Gmail confirmation, and best-effort recovery transitions. The immutable replacement event is one-to-one with its replacement operation and records its target generation; replay first compares that target to the mutable root and fails closed if a later replacement advanced it. A fresh concurrent key cannot pass the missing-draft predicate after the first committed transition, and a pending Sent-mail recheck blocks replacement until that proof attempt completes.

The Sent-mail reconciler writes the root's current generation onto each reconciliation operation when it is created. Migration 377 backfills the already-deployed generation-1 operations through a non-null default. On a same-key replay and again under the finalization lock, the reconciler compares the operation generation to the current root before it can return a completed result, make a Gmail lookup, or write reconciliation evidence. A later replacement therefore makes the old reconciliation key an explicit stale conflict rather than pairing its historical outcome with a new delivery projection. A pending claim has a durable five-minute lease based on its request timestamp; when a different same-generation operation completes after that lease, it records an append-only supersession actor/timestamp on the abandoned key. Active concurrent keys are not superseded.

Migration 377 also adds a `BEFORE UPDATE` trigger on `invoices`. It first acquires the same approval advisory lock held by replacement preparation, then rejects any invoice mutation while the corresponding event's current generation remains `creating`, `retryable`, or `recovery_required`. Sent-mail finalization acquires the invoice row before that same advisory lock, matching PostgreSQL's update/trigger order and avoiding a lock cycle. A legacy MCP/repository writer that arrives before the replacement lock commits first, so preparation rereads its changed invoice and fails closed; one that arrives afterward waits, observes the committed replacement claim, and cannot change lifecycle or render fields before the Gmail outcome is known.

The prior missing identity is not deleted or silently discarded: the event snapshot records its root ID, prior/new generation, full durable draft/reconciliation state, actor, timestamp, and replacement operation. An older provider revision continues to see one root record rather than encountering duplicate approval/PDF rows; because it cannot derive a later generation's RFC Message-ID, it fails closed instead of issuing a duplicate draft.

### Execution model and invariant

The provider state machine runs in PostgreSQL's default `READ COMMITTED` transaction model. Gmail calls are deliberately outside those transactions. The protocol uses only database-provided closed-surface components: transaction-scoped advisory locks serialize a given operation key and the shared approval transition; unique operation keys and persisted generation bindings attach each committed intent to one durable identity; conditional row updates protect state transitions; atomic transactions persist either the full intent/evidence change or none of it; the Sent-mail finalizer takes the invoice row before the shared approval advisory lock; and the invoice `BEFORE UPDATE` trigger then takes that same advisory lock to fence legacy invoice writers while an unresolved replacement is current.

For every interleaving admitted by that model, each `(operation source, idempotency key)` has one durable intent bound to exactly one `(draft record, generation)`. A root generation has at most one unresolved replacement intent; distinct Sent-mail reconciliation keys may observe that same generation concurrently, and active keys remain live through their five-minute lease. A replay, asynchronous confirmation, recovery transition, or finalization may return, mutate, or make an external Gmail lookup/create call only while its persisted/prepared generation equals the current root generation. A replacement refuses to advance a root while any reconciliation intent is pending. Once a later replacement commits, every operation bound to an earlier generation fails before Gmail or an invoice write. A completed same-generation reconciliation can supersede only expired pending claims, retaining their actor/time rather than deleting history. An invoice writer either commits before replacement preparation reads the invoice (so the replacement precondition fails), or waits for the approval lock and is rejected by the trigger while the replacement remains unresolved; sent reconciliation uses the matching invoice-then-advisory order. No path in this service marks an invoice sent; only the pre-existing Sent-mail reconciler can do so after verifiable Gmail `SENT` evidence.

This model assumes PostgreSQL advisory locks and row/trigger semantics are available to all application writers, migrations have been applied before this provider version serves traffic, and privileged direct database administration does not disable the trigger or edit rows outside the application protocol. Gmail can be unavailable, ambiguous, duplicate a transport response, or lose a request response; those outcomes are represented by the persisted retry/recovery state and do not authorize a new generation or send operation. The model does not assume Gmail supports caller idempotency.

## Intentional

- A replacement is explicit only. `create_or_reuse` does not infer that `draft_missing` authorizes a new customer-visible draft.
- The root record is a current-delivery projection, while migration 377's event table is the append-only identity/history ledger. This preserves the existing one-row compatibility contract without losing the missing generation.
- A replacement gets a new generation-scoped RFC Message-ID; it does not reuse the missing draft's mailbox identity.
- A stale replacement idempotency key is rejected rather than reconstructed against a later current root; the append-only event is the durable generation claim for that key.
- An ordinary create/recover key and every asynchronous draft transition are equally generation-bound; an older coroutine cannot attach its Gmail identity or recovery state to a replacement generation.
- A stale completed Sent-mail reconciliation key is rejected rather than paired with a later root's mutable reconciliation projection; its operation stores the generation it observed.
- A crashed Sent-mail lookup becomes recoverable after its five-minute lease only when a later completed same-generation observation records a durable supersession. Active concurrent keys retain their original retry behavior.
- A commercial invoice is temporarily immutable while its explicit replacement generation is unresolved. Rejecting a legacy mutation is safer than permitting a stale PDF or an out-of-band sent status; the trigger lifts automatically once the draft is confirmed.
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

- Ephemeral local PostgreSQL (`postgres:16`, loopback-only port) ran `python -m pytest -q tests/test_commercial_billing_gmail_drafts.py -k 'ordinary_gmail_draft_replay_rejects_stale_replacement_generation_before_gmail or stale_gmail_create_completion_and_recovery_transition_cannot_touch_new_generation or sent_reconciliation_finalization_rejects_a_completed_old_generation_after_replacement or completed_reconciliation_supersedes_an_abandoned_pending_claim_and_unblocks_replacement or sent_reconciliation_locks_invoice_before_approval_to_avoid_trigger_cycle or replacement_migration_backfills_existing_sent_reconciliation_operations'` — 6 passed, 124 deselected in 2.66s.
- The same isolated database ran `python -m pytest -q tests/test_commercial_billing_gmail_drafts.py` — 130 passed, 1 warning in 19.63s.
- Local equivalents of the billing/invoicing workflow groups passed: receivables, candidates, runs, approvals, Gmail drafts, Square, repository, and PDF group — 596 passed, 1 warning in 72.76s; MCP/OAuth group — 43 passed in 0.56s; approval blocker — 2 passed, 41 deselected in 0.19s; migration runner — 30 passed, 1 skipped in 0.42s.
- Scoped `ruff check`, Python compile, and `git diff --check` passed. The managed `scripts/push_pr.sh` local-review/unit mirror and pinned redacted Gitleaks scan will be re-run against the final corrective commit before publication; hosted checks are observed only, per the operator's local-verification direction.
- Pre-implementation production evidence: `BEGIN TRANSACTION READ ONLY` inspection of the live `atlas` database at the #2387 runtime returned 0 root draft/missing/sent rows and confirmed the existing unique-constraint names; it made no data change.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/api/invoicing/receivables.py` | 25 |
| `atlas_brain/services/commercial_billing_invoice_gmail_drafts.py` | 839 |
| `atlas_brain/services/commercial_billing_invoice_gmail_sent_reconciliation.py` | 129 |
| `atlas_brain/storage/migrations/377_commercial_billing_gmail_draft_replacements.sql` | 169 |
| `plans/PR-EOM-Missing-Gmail-Draft-Replacement.md` | 136 |
| `tests/test_commercial_billing_gmail_drafts.py` | 1584 |
| **Total** | **2884** |
