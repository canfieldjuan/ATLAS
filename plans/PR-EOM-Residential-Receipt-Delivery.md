# PR-EOM-Residential-Receipt-Delivery

## Why this slice exists

The EOM Billing & Payments coordinator [#2362](https://github.com/canfieldjuan/ATLAS/issues/2362) requires a committed residential payment to have a retry-safe receipt-email lifecycle. The prior outbox slice deliberately stopped before transport: migration 369 calls itself non-sending and requires a future explicit claim plus verifiable transport result. The Website can display its `pending`/`skipped` projection, but neither ATLAS API profile has a dispatch/recovery operation.

Diff-budget override: this 3,927-added-line provider slice is deliberately over the 400-LOC soft cap because one safe, independently deployable receipt-dispatch contract requires its additive durable state, the exact Gmail transport boundary, both established ATLAS service profiles, and executable crash/concurrency proof together. Splitting those pieces would either publish a state machine with no safe operator boundary or a sender without the durable no-duplicate recovery invariant; no consumer is enabled by this release.

### Problem-derived contract

- Root cause: the ledger owns a deterministic receipt snapshot but has no durable, actor-attributed operation that can safely decide whether Gmail received a send request. Calling the generic Gmail sender directly would make a timeout indistinguishable from a duplicate customer email.
- Correct fix must touch/change: add additive ATLAS receipt-dispatch persistence; extend the shared Gmail transport with server-owned immutable headers, definite-vs-uncertain send errors, and receipt metadata lookup; add one authenticated, idempotent dispatch route to both existing EOM receivables API profiles; and prove migration, normal, retry, failed, concurrent, crash-recovery, and authorization behavior with non-customer doubles.
- Must not change: customer payment, allocation, deposit, clearing, return, void, invoice, receipt body, generic MCP, commercial Gmail draft/sent-mail, Square, Resend fallback, or automatic-mail behavior. No test or deployment probe sends a real customer email.

### Readiness repair contract

- Root cause: receipt-delivery readiness still names only the migration-369 outbox objects even though dispatch now depends on migration-378 columns, tables, and indexes. A deployment with startup migrations disabled can therefore report ready before the first dispatch fails on an absent operation/reconciliation object.
- Correct fix must touch/change: extend only the additive EOM receipt-delivery readiness contract with the closed migration-378 schema set; use the production migration runner for real-schema fixtures that exercise migration 378's concurrent DDL; and prove a missing migration-378 column, table, or required index returns not-ready while legacy receivables readiness remains usable.
- Must not change: the startup migration order, payment/allocation behavior, receipt dispatch/reconciliation state machine, Gmail transport, public route shape, or legacy `is_ready()` contract.

### Invalid-proof recovery repair contract

- Root cause: recovery treats Gmail lookup outage as ambiguous but lets malformed or mismatched Sent evidence escape as a conflict, leaving an already-attempting operation without durable recovery evidence.
- Correct fix must touch/change: in only the post-attempt recovery paths, treat invalid proof as unverified evidence and persist `recovery_required`; prove immediate uncertain-send and later crash-recovery paths do so without a second send.
- Must not change: preflight behavior before an attempt, the exact Gmail proof grammar, payment state, Gmail transport, route shape, or retry semantics.

### Mixed-version migration cutover repair contract

- Root cause: migration 378 contains concurrent index DDL, so the production runner executes its SQL statements separately in autocommit mode. Its current order backfills `rfc_message_id` before installing the default and `NOT NULL` invariant, leaving a committed interval in which an already-deployed legacy receipt writer can insert a new null identity and make the later `SET NOT NULL` fail.
- Correct fix must touch/change: install the server-side `rfc_message_id` default before backfilling historical nulls, then enforce `NOT NULL` only after the backfill; prove through the production migration runner and a second schema-scoped connection that a legacy writer which omits the new column in the former seam receives the default and migration 378 completes.
- Must not change: the deterministic historical identity format, payment/receipt facts or status, the migration runner's execution model, the concurrent-index retry behavior, migration-378 tables/constraints, or legacy writer call shapes.

### HTTP timeout outcome repair contract

- Root cause: `GmailTransport.send` classifies every HTTP 4xx response as definitely not accepted, including HTTP 408 even though the raw message body may have reached Gmail or an intermediary before the timeout response. The receipt service treats that explicit classification as a definite failure and permits a new idempotency key to send again.
- Correct fix must touch/change: classify HTTP 408 alone as an ambiguous send outcome so the existing receipt attempt/recovery path performs Sent-mail reconciliation rather than enabling retry; prove the transport emits `definitely_not_sent=False` for 408 while ordinary rejected 4xx behavior remains unchanged.
- Must not change: Gmail request construction, OAuth/authentication failure handling, explicit 403 permission guidance, other HTTP-status classifications, receipt service state transitions, idempotency rules, routes, payment state, or any generic Gmail caller API.

### Late definite-rejection finalization repair contract

- Root cause: an original Gmail send can outlive the attempt lease. A concurrent reconciliation then safely changes its still-ambiguous operation from `attempting` to `recovery_required`; if the original request later receives a definite pre-acceptance rejection, `_record_definite_failure` currently returns that recovery result rather than recording the authoritative retryable `failed` outcome.
- Correct fix must touch/change: in only the existing direct Gmail definite-failure finalizer, accept a still-active `recovery_required` operation under the current delivery lock, complete it as `failed`, and clear its recovery markers; prove the interleaving of a blocked original send, post-lease no-proof reconciliation, late definite rejection, and a later new-key retry.
- Must not change: the attempt lease, reconciliation's no-proof behavior, uncertain/timeout recovery, verified-Sent completion, lock order, external-send count before the retry, payment state, routes, or generic Gmail transport send/error behavior.

### Recipient log-redaction repair contract

- Root cause: the shared successful-send log interpolates the raw `to` recipient list. The new receipt dispatch supplies a persisted residential customer email to that shared transport, so successful dispatches retain customer email addresses in application logs.
- Correct fix must touch/change: redact recipients at the shared `GmailTransport.send` success-log boundary by logging only the recipient count, and prove a successful multi-recipient transport call records neither recipient address while retaining the delivery identifier, count, and existing subject observability.
- Must not change: the `to` argument sent to Gmail, raw MIME recipient construction, OAuth, headers, request/response/error classification, service state machine, routes, recipient persistence, payment state, or existing subject-log treatment.

### Full-profile reconciliation reachability repair contract

- Root cause: the full ATLAS reconciliation route is implemented with the real bearer/actor dependency and service dependency, but its current test calls the handler directly. That bypasses routing, header parsing, authentication, and dependency injection, so the existing slim-profile HTTP proof does not settle the full-profile reachability claim.
- Correct fix must touch/change: replace only the direct full reconciliation handler invocation with an ASGI request through the mounted full router; prove unauthenticated rejection, authenticated actor forwarding, observable reconciliation result, and the no-dispatch property using the existing in-process transport fake.
- Must not change: full or slim route implementation, dependencies, authentication rules, response mapping, service behavior, Gmail calls, recipient data, payment state, or the established slim-profile HTTP proof.

## Scope (this PR)

Ownership lane: eom/billing-payments-receipt-delivery
Slice phase: vertical slice
Max files: 14

1. Add migration 378, extending the existing receipt-outbox record with a stable RFC Message-ID and verified Gmail sent identity, plus a separate idempotent receipt-dispatch operation table. Existing rows derive their identity from the receipt-delivery UUID; a server-side default is installed before the historical backfill so mixed-version legacy writers receive a new stable identity without changing payment facts, receipt text, or delivery status. A migration retry drops any same-named invalid concurrent index before rebuilding it, and completed operations retain an immutable delivery-result snapshot.
2. Add `ResidentialPaymentReceiptDeliveryService`. It may dispatch only an existing `pending` or definitely `failed` residential receipt with a canonical persisted recipient. The service records the request/actor and marks an attempt in PostgreSQL before Gmail is called; it queries Gmail Sent by that stable identity before a new send and after an uncertain result.
3. Extend `GmailTransport.send` with the same narrow immutable-header validation used by drafts, make known rejected sends distinguishable from uncertain send outcomes, and log only a recipient count after a successful send; HTTP 408 is explicitly uncertain rather than retryable. Receipt dispatch calls `GmailTransport` directly and never routes through the Gmail-to-Resend composite provider. No-send reconciliation writes append-only actor/timestamp/outcome evidence.
4. Add the idempotent `POST /api/v1/receivables/payments/{payment_id}/receipt-delivery` send boundary plus `POST /api/v1/receivables/payments/{payment_id}/receipt-delivery/reconcile` to both full and slim service-to-service EOM profiles. Dispatch requires the existing bearer/actor boundary and `Idempotency-Key`; reconciliation is authenticated, performs only an exact Gmail Sent-mail lookup for an already-ambiguous operation, and cannot send email. Both expose only persisted receipt-delivery data, never credentials or raw Gmail responses.
5. Add an isolated local-PostgreSQL migration/service suite, full and slim HTTP reachability tests for both dispatch and reconciliation, Gmail transport byte/header tests using `httpx.MockTransport`, and CI enrollment for the new test surface. Invalid/mismatched Sent evidence after an attempt must persist recovery-required evidence rather than strand an `attempting` operation; a late definite rejection after post-lease reconciliation must still complete as retryable `failed`.
6. Make the existing EOM receipt-delivery readiness probe fail closed unless every migration-378 dispatch column, table, and safety index is present; retain the legacy receivables readiness contract unchanged; keep pre-378 fixtures explicit and apply the full current schema through the production migration runner where receipt-aware behavior is exercised.

### Review Contract

- Acceptance criteria:
  - Migration 378 preserves every prior payment/receipt row, installs the `rfc_message_id` default before backfilling historical nulls, derives a deterministic identity for those historical rows, supplies the same stable default to a mixed-version legacy writer in that cutover seam, and enforces one dispatch operation per `(source, idempotency_key)`; a failed concurrent unique-index build is removed and rebuilt on migration replay. Its rollback is application-first and retention-preserving. Settled by isolated PostgreSQL migration/replay tests in `tests/test_residential_payment_receipt_delivery.py` through the production migration runner and a second schema-scoped writer connection.
  - The delivery service persists its operation before the Gmail fake is invoked, sends the already-persisted subject/body to only the persisted recipient, and returns `sent` with Gmail identity. A Sent-mail proof can safely complete a still-`prepared` operation without dispatching, and later no-send reconciliation records its actor/time/outcome append-only. Both full and slim dispatch and reconciliation HTTP boundaries preserve their bearer/actor requirements, forward the authenticated actor to the service, map service outcomes without exposing Gmail internals, and reconciliation never calls dispatch. Settled by full and slim ASGI plus service tests.
  - A same `(source, idempotency_key, payment_id)` returns that operation's immutable completed result without another Gmail call; reuse of that key for another payment conflicts before external I/O. A later retry may advance the receipt row without changing the original key's replay. Settled by service retry tests and HTTP boundary forwarding/auth tests.
  - A known Gmail rejection leaves the committed payment untouched, records `failed`, and permits a later new idempotency key to retry that existing receipt, including when a post-lease reconciliation changed the still-active operation to `recovery_required` before the original request obtained its rejection. HTTP 408 instead produces an ambiguous `GmailSendError`, which follows the existing no-resend recovery path; existing non-408 rejected behavior is retained. Settled by Gmail transport and service interleaving tests.
  - A successful `GmailTransport.send` log includes the Gmail delivery identifier, recipient count, and existing subject observability but never a raw recipient address. Settled by the multi-recipient successful transport-log assertion in `tests/test_residential_payment_receipt_delivery.py`.
  - A timeout, cancellation, malformed accepted response, invalid/incomplete Sent proof, or process interruption after the durable `attempting` marker cannot invoke a second send. The operation remains `pending`/`recovery_required`; the authenticated reconcile route gives a reloaded Website a no-send recovery path by performing only a Sent lookup against that existing operation, and only records `sent` after exact `SENT`, Message-ID, and receipt-id header evidence. This is the selected PostgreSQL/unique-index execution model: every state transition takes the receipt advisory lock before operation and delivery row locks, while the active-operation partial uniqueness invariant remains a database backstop; tests exercise concurrent callers, crash-shaped `attempting`, lookup-hit, lookup-miss, invalid proof, actor audit, and the no-ambiguous-attempt guard.
  - EOM `/receivables/ready` can report receipt-delivery-ready only after the migration-378 dispatch columns, operation/reconciliation tables, and required indexes are present and valid. A missing one returns not-ready without changing the legacy receivables readiness result. Settled by the isolated local-PostgreSQL readiness test in `tests/test_receivables.py`.
  - `skipped/no_email`, absent receipt records, invalid/incomplete Gmail proof, unauthenticated requests, and unavailable Gmail all fail closed without a payment mutation or a customer send. Settled by negative route/service/transport tests.
  - Existing payment creation, legacy receipt projections, commercial draft/sent-mail behavior, and generic Gmail/Resend caller interfaces retain their inputs and call shapes. HTTP 408 alone now carries ambiguous rather than definite Gmail-send metadata; non-408 classifications retain their behavior. Settled by focused receipt regressions, Gmail transport regression tests, and the local invoicing-check selection.
- Reachability proof: both service-to-service route profiles are exercised through their mounted ASGI routers with their real bearer/actor dependencies and generated bearer tokens. Their observable effect is an operation row plus a receipt `sent`/`failed`/`pending` projection in an isolated ledger schema.
- Affected surfaces: EOM receivables migrations/readiness, full and slim authenticated receivables routers, Gmail raw-message transport, receipt-ledger projection, local PostgreSQL tests, and the invoicing-check workflow path/run lists.
- Risk areas: customer-email duplication, uncertain external outcome, payment immutability, race/crash ordering, customer address exposure, migration backfill/rollback, auth bypass, API compatibility, and CI enrollment.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

Closure declaration for the new decision-driving sets:

- Receipt delivery status is **CLOSED** and **DERIVED** from migration 369's canonical check constraint: `pending|sent|failed|skipped`. A status outside that set is a database rejection and a service conflict; no fallback is admitted because falsely calling a receipt sent is costlier than leaving it pending.
- Dispatch operation state is **CLOSED** and **AUTHORED HERE** by migration 378: `prepared|attempting|completed|recovery_required`; unknown state is a service conflict and cannot reach Gmail. Completion outcome is likewise closed to `sent|failed|already_sent` with null only for active states.
- Gmail proof headers are **CLOSED** and **DERIVED** from server constants (`Message-ID`, `To`, and `X-Atlas-EOM-Payment-Receipt`); unrecognized/missing/ambiguous evidence remains pending/recovery-required rather than creating an email or marking it sent.
- Gmail's producer-supplied Sent-mail payload grammar is **OPEN**: values, containers, and duplicate/unknown header shapes cannot be enumerated. Its one required proof-header family is **CLOSED** and **DERIVED** from the three server constants above. Anything outside the documented JSON-list shape containing each exact proof once takes the safe direction—reject the evidence before `sent` can be recorded, because a false sent assertion is more costly than leaving a valid customer receipt pending for reconciliation. `test_sent_mail_proof_grammar_uses_a_spec_derived_oracle` generates tokens × containers × key families and compares to that independent rule.

- Boundary path/seam: receipt delivery moves from a non-sending durable outbox to one explicit, authenticated `/payments/{payment_id}/receipt-delivery` side-effect boundary and a separate `/payments/{payment_id}/receipt-delivery/reconcile` recovery boundary. Gmail's external request is behind `ResidentialPaymentReceiptDeliveryService`; raw client bodies cannot choose recipient, body, status, Message-ID, provider, or headers. Reconciliation cannot reach the send method.
- Replaced-path behaviors: none. Existing payment creation still only enqueues; existing generic Gmail sender/draft routes retain their existing call shapes. This endpoint is additive and no caller invokes it until the tracker/Website consumer slice deploys.
- Guard-relevant fields: path `payment_id`, bearer service credential, `X-EOM-Actor`, dispatch `Idempotency-Key`, persisted delivery id/status/recipient/subject/body, stable RFC Message-ID, Gmail message/thread identifiers, and exact mailbox headers.
- Caller x input shape: existing full and slim authorized service callers x a valid UUID path and 1-128-character idempotency key may dispatch only an existing dispatchable receipt. Missing/unauthorized/malformed/skipped/sent/recovery-required state never invokes Gmail send. An authenticated reconciler admits only an existing `attempting` or `recovery_required` operation and reaches Gmail Sent lookup, never Gmail send. Same dispatch key plus a different payment conflicts before external I/O.

### Deployed-config probing

- Deployed/default config values: no new configuration, feature flag, or credential source is introduced. The existing direct `GmailTransport` token-store boundary is used only on this explicit authenticated route; the composite Gmail-to-Resend fallback is intentionally not called.
- Explicit value probe: a fake direct Gmail gateway with valid `send` and Sent lookup proof yields `sent`; transport tests prove the raw MIME contains only server-supplied immutable headers.
- Absent value probe: missing/invalid Gmail credentials, 403, missing token, unavailable token refresh, and unavailable mailbox lookup produce controlled unavailable/failed/recovery outcomes with no payment or additional send.
- Default-session/default-context probe: a payment write with no later dispatch request remains the existing queued `pending`/`skipped` behavior; ordinary `GmailTransport.send` callers without `headers` preserve their old MIME shape.
- Side-effect ordering: validate request and load/lock delivery -> persist `prepared` operation -> read-only Sent lookup -> atomically persist `attempting` marker -> Gmail `messages.send` -> atomically record `sent` or definite `failed`; uncertain outcomes perform lookup only and retain recovery evidence. Financial payment state is never inside or after the transport transaction.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/services/residential_payment_receipt_delivery.py`
- `atlas_brain/storage/migrations/378_receivables_payment_receipt_delivery.sql`
- `atlas_brain/tools/gmail.py`
- `plans/PR-EOM-Residential-Receipt-Delivery.md`
- `tests/test_commercial_billing_gmail_drafts.py`
- `tests/test_eom_payment_receipts.py`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`
- `tests/test_residential_payment_receipt_delivery.py`

## Mechanism

Payment creation remains an atomic financial transaction that queues a receipt. Dispatch is a later operation rooted at the receipt row, never at mutable CRM data. Migration 378 derives historical identities as `<atlas-eom-payment-receipt-{delivery-id}@effinghamofficemaids.com>` and gives future legacy writers a generated, stable server-owned identity; it stores any verified Gmail message/thread/timestamp separately from the financial payment.

Each dispatch request hashes only the target payment identity and persists `(source, idempotency_key, actor, fingerprint)` in an operation row under the receipt's PostgreSQL advisory lock. The primary key, `(source, key)` unique constraint, and one-active-operation partial unique index are the selected closed-surface component for concurrency; no file lease, in-memory map, or ad-hoc retry worker is used. Every state transition takes that receipt lock before the operation and delivery row locks. Before a call can reach Gmail it must atomically move that operation from `prepared` to `attempting`. Only the request that wins that conditional transition can call `send`.

The operation marker commits before the external request. Therefore a crash after the marker but before/after Gmail is conservatively ambiguous: later callers lookup Sent mail by the stable RFC Message-ID and exact receipt header, but never issue another send. A successful API response or verified Sent lookup records `sent`. A known pre-acceptance rejection records `failed`, allowing a new idempotency key to retry the same persisted receipt. A completed operation also snapshots its own resulting receipt status/timestamp, so replay cannot be silently rewritten by a later retry. A timeout, cancellation, 5xx, malformed result, missing proof, or crash-shaped `attempting` state remains `pending` with `recovery_required`; the separate authenticated reconcile route gives the next consumer a durable no-send recovery path without needing the original dispatch key and appends the reconciliation actor/time/outcome.

This is one execution surface: PostgreSQL coordinates durable receipt-dispatch state, while Gmail is the single external delivery provider. The invariant for every admitted interleaving is: once any operation has entered `attempting`, no other operation for that receipt can reach `GmailTransport.send`; after a verified `sent`, no operation can reach it at all. The residual assumption is Gmail's documented Sent lookup is read-only evidence for messages it accepted; if that evidence is temporarily unavailable, the service takes the safe non-send branch.

## Intentional

- Dispatch is operator-triggered only. The enqueue on payment commit is unchanged, and this provider PR creates no background worker, cron, automatic send, or tracker/Website call.
- An ambiguous outcome is not relabeled `failed` just to make a retry button easy. It stays `pending` with recovery evidence because re-sending could duplicate a customer receipt.
- Generic `CompositeEmailProvider` is intentionally rejected because its Gmail-to-Resend fallback changes delivery identity and defeats Gmail Sent-mail reconciliation.
- The full and slim routes intentionally share one provider service but retain their established router/auth profiles; no client can directly call ATLAS from the Website.

## Deferred

- Coordinator #2362 follow-up: eom-timetracker proxy, then Website recipient confirmation/status/retry controls after this provider deploys.
- Coordinator #2362 follow-up: an explicit audited replacement policy for a permanently unresolved Gmail attempt, if protected operational evidence shows it is needed; it must use a new generation/identity and never silently re-send.
- #2363 H-15 remains the unrelated commercial Gmail-draft/PDF cross-link schema prevention and scheduled mailbox observation work.

Parked hardening: none.

## Verification

- 2026-08-15 local, no customer Gmail: `python -m py_compile atlas_brain/services/residential_payment_receipt_delivery.py tests/test_residential_payment_receipt_delivery.py tests/test_eom_payment_receipts.py && ruff check atlas_brain/services/residential_payment_receipt_delivery.py tests/test_residential_payment_receipt_delivery.py tests/test_eom_payment_receipts.py` — pass.
- 2026-08-15 local PostgreSQL (`ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas`): `python -m pytest tests/test_residential_payment_receipt_delivery.py tests/test_eom_payment_receipts.py -q` — 34 passed; one environment-only `torch`/`pynvml` deprecation warning.
- 2026-08-15 local: `python -m py_compile atlas_brain/services/receivables.py tests/test_receivables.py && ruff check atlas_brain/services/receivables.py tests/test_receivables.py` — pass.
- 2026-08-15 local PostgreSQL: the readiness/migration/ledger regression subset in `tests/test_receivables.py` — 7 passed; no real Gmail call.
- 2026-08-15 local PostgreSQL: invalid-proof recovery subset in `tests/test_residential_payment_receipt_delivery.py` — 5 passed; no real Gmail call.
- 2026-08-16 local: `python -m py_compile atlas_brain/tools/gmail.py tests/test_residential_payment_receipt_delivery.py && ruff check atlas_brain/tools/gmail.py tests/test_residential_payment_receipt_delivery.py` — pass.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_residential_payment_receipt_delivery.py::test_migration_installs_default_before_backfill_for_mixed_version_legacy_writer tests/test_residential_payment_receipt_delivery.py::test_gmail_transport_send_preserves_immutable_headers_and_classifies_outcomes -q` — 2 passed; the first test runs migration 378 through the production runner while a second schema-scoped connection writes a legacy row in the former cutover seam, and the second asserts HTTP 408 is ambiguous.
- 2026-08-16 local: `python -m py_compile atlas_brain/services/residential_payment_receipt_delivery.py tests/test_residential_payment_receipt_delivery.py && ruff check atlas_brain/services/residential_payment_receipt_delivery.py tests/test_residential_payment_receipt_delivery.py` — pass.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_residential_payment_receipt_delivery.py::test_late_definite_rejection_after_lease_reconciliation_is_retryable tests/test_residential_payment_receipt_delivery.py::test_unknown_gmail_result_never_resends_and_can_later_reconcile_sent_evidence tests/test_residential_payment_receipt_delivery.py::test_definite_gmail_rejection_is_failed_and_a_new_key_can_retry -q` — 3 passed; a blocked original send is reconciled after the lease, then a late definitive rejection completes it as retryable `failed` without an extra send.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_residential_payment_receipt_delivery.py -q` — 22 passed; no real Gmail call.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_receivables.py tests/test_eom_payment_receipts.py tests/test_residential_payment_receipt_delivery.py tests/test_commercial_billing_gmail_drafts.py tests/test_eom_render_profile.py tests/test_eom_billing_recipients.py -q` — 403 passed; no real Gmail call.
- 2026-08-16 local: `python -m py_compile atlas_brain/tools/gmail.py tests/test_residential_payment_receipt_delivery.py && ruff check atlas_brain/tools/gmail.py tests/test_residential_payment_receipt_delivery.py` — pass.
- 2026-08-16 local: `python -m pytest tests/test_residential_payment_receipt_delivery.py::test_gmail_transport_send_preserves_immutable_headers_and_classifies_outcomes -q` — 1 passed; a successful five-recipient Gmail transport call preserves all five recipients in raw MIME but logs only its Gmail id, recipient count, and subject.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_residential_payment_receipt_delivery.py -q` — 22 passed; no real Gmail call.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_receivables.py tests/test_eom_payment_receipts.py tests/test_residential_payment_receipt_delivery.py tests/test_commercial_billing_gmail_drafts.py tests/test_eom_render_profile.py tests/test_eom_billing_recipients.py -q` — 403 passed; no real Gmail call.
- 2026-08-16 local: `python -m py_compile tests/test_eom_payment_receipts.py && ruff check tests/test_eom_payment_receipts.py && python -m pytest tests/test_eom_payment_receipts.py::test_receipt_reconciliation_routes_are_authenticated_no_send_boundaries -q` — 1 passed; the slim and full mounted routers reject unauthenticated reconciliation, forward their authenticated actors to reconciliation, and do not dispatch.
- 2026-08-16 local PostgreSQL: `python -m pytest tests/test_eom_payment_receipts.py -q` — 14 passed; no real Gmail call.
- Before publication: `python scripts/sync_pr_plan.py plans/PR-EOM-Residential-Receipt-Delivery.md origin/main`, `python scripts/sync_pr_plan.py --check plans/PR-EOM-Residential-Receipt-Delivery.md origin/main`, `git diff --check`, and `bash scripts/push_pr.sh <current-pr-body> -u origin HEAD` run the repository’s local mechanical review. No hosted GitHub status is treated as the acceptance gate under Juan's local-check direction.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/api/invoicing/receivables.py` | 88 |
| `atlas_brain/eom_api/receivables.py` | 83 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/receivables.py` | 113 |
| `atlas_brain/services/residential_payment_receipt_delivery.py` | 1451 |
| `atlas_brain/storage/migrations/378_receivables_payment_receipt_delivery.sql` | 298 |
| `atlas_brain/tools/gmail.py` | 84 |
| `plans/PR-EOM-Residential-Receipt-Delivery.md` | 190 |
| `tests/test_commercial_billing_gmail_drafts.py` | 2 |
| `tests/test_eom_payment_receipts.py` | 193 |
| `tests/test_eom_render_profile.py` | 1 |
| `tests/test_receivables.py` | 183 |
| `tests/test_residential_payment_receipt_delivery.py` | 1269 |
| **Total** | **3963** |
