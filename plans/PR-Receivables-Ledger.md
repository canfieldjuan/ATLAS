# EOM Receivables Ledger and Authenticated Payment API

## Why this slice exists

Atlas stores each invoice application as an unrelated payment row. That loses the identity of a physical check covering several invoices, cannot preserve unapplied credit or deposit state, and performs balance/status updates outside a transaction. The current HTTP invoice actions are also unauthenticated, so extending them directly would widen a money-write risk.

Post-open review found four invariant-placement gaps in the first implementation: the retired quick-pay route inherited the live API's router-level auth gate, the shared currency helper rounded sub-cent inputs, event-key lookup and locking were scoped too narrowly to prevent cross-payment reuse, and payment summaries treated every unreversed allocation as active even after return or void. The fixes place each rule at its authoritative boundary rather than special-casing the reported examples.

Exact-head review and CI then exposed two more root causes in the follow-up: allocation adjustment treated mutable payment status as authoritative before consulting the immutable idempotency event ledger, so a completed adjustment could stop replaying after a later return or void; and several new tests replaced first-party components or asserted fake SQL text, increasing the repository's maturity debt instead of proving behavior through supported boundaries. This fix moves replay ownership ahead of state-dependent admission and replaces those new internal mocks with explicit request/runtime dependency seams plus real PostgreSQL behavior checks. It does not accept a weaker maturity baseline.

The exact-head security scan also identified the literal test operation key `mcp-boundary-1` as a generic API key at the commit where that fixture entered the branch. It was not a credential, but the PR scanner evaluates the full commit range, so deleting the literal only at the branch tip could not reconcile the historical false positive. A PR-owned `.gitleaksignore` entry was also not a trustworthy fix because the trusted `pull_request_target` job ran the base branch's checker, which did not yet protect ignore growth. The corrected publication is one clean commit based directly on the reviewed base, contains no `.gitleaksignore`, and retains the UUID fixture already present in the final source. The future trusted checker rejects every first or later ignore addition unless a labeled security-only rotation authorizes it.

The repo-wide Unit Gate subsequently exposed shared-process test contamination rather than an MCP contract regression. Legacy B2B test modules install a passthrough `FastMCP` fake with `sys.modules.setdefault(...)` during collection; when that fake wins the collection order, the new receivables test sees no real tool registry and fails while the same schema and validation contract pass with the CI-pinned MCP 1.28.1/Pydantic 2.13.4 runtime. The regression proof therefore runs the real MCP boundary in a clean child interpreter and uses the supported `list_tools`/`call_tool` surface instead of reading the process-global private registry.

Live reconciliation then found two deployment-readiness omissions on the same reviewed head. The invoicing workflow enrolled migration 344 but not its required follow-up migration 345, so a future 345-only change could skip the PostgreSQL receivables suite. The readiness query also treated table existence plus `invoice_payments.payment_id` as the complete schema contract even though the write path dereferences additional allocation-reversal, deposit-clear, lifecycle, and audit columns. The fix enrolls 345 in both workflow triggers and makes readiness compare the current schema to one explicit manifest of every receivables table column introduced by migration 344.

The next exact-head review found three related outcome-classification gaps. Returned or voided receipts suppressed active allocations but reported the entire inactive receipt as available unapplied credit; idempotent singular-MCP retries deduplicated money but repeated their CRM side effect; and compatibility-only non-check methods were classified as `received`, placing cash/card/Zelle/Venmo/other rows in the check-deposit queue that later rejects them. The root fix makes availability and deposit eligibility derive from lifecycle/method state, and carries the transaction's replay outcome through the repository boundary so secondary effects run only on the first commit without changing the MCP response shape.

A later exact-head review exposed four deployment-boundary gaps. The standalone invoicing MCP initializes a pool but, unlike the FastAPI process, never applies the migrations required by its payment compatibility writer. Readiness models only columns even though write safety and `ON CONFLICT` inference depend on migration-created indexes. The HTTP adapter recognizes an uninitialized pool but not live asyncpg connection, capacity, shutdown, interface, or timeout failures. Finally, the exact historical `.gitleaksignore` exception is consumed directly from a PR checkout while the trusted `pull_request_target` guard protects only the JSON baseline. These are root contract omissions: each boundary advertises or permits a finance write without proving the prerequisite it actually relies on. The fix makes MCP startup migration-gated, expands readiness to the required migration indexes, maps only database-availability runtime classes to 503, and extends the trusted-base Gitleaks rotation policy to reject unapproved ignore growth.

Cold review of that repair found two remaining overclaims. First, its availability classifier was broader than the contract: every `asyncpg.InterfaceError` and every `OSError` became a 503 even though those families also represent transaction/API misuse and non-network programming failures. Second, its readiness manifest covered only literal `CREATE INDEX` statements, not the primary-key and unique indexes created by migration table constraints. The corrected boundaries recognize only unavailable pool/connection interface states—including a connection resource unwinding after its backend closes—plus concrete network and timeout failures, and require the complete explicit plus constraint-backed migration index set; unrelated interface/OS errors still escape and any missing primary-key or unique index now fails readiness.

The latest exact-head review found four final boundary-ordering gaps. The proposed ignore bootstrap was still PR-controlled on its first adoption; standalone MCP startup ran migrations but did not prove the full receivables schema ready; the singular payment tool accepted an unbounded idempotency key even though storage is limited to 128 characters; and allocation suggestions inherited customer-name-first ordering from the open-invoice feed instead of honoring their oldest-due-first contract. The root fixes remove the ignore entirely through clean-history publication, require service readiness before the MCP yields, reject empty or oversized singular keys before repository access (with a matching service invariant), and order the authoritative invoice feed by due date before customer name.

### Problem-derived contract

- Root cause: the persistence and service contract model an invoice-local row, not a customer payment with allocations and lifecycle. Transport-specific writers call that non-atomic repository method directly.
- A correct fix must add one authoritative payment parent, explicit allocations, unapplied credit, deposit/audit lifecycle, idempotent transactional writes, deterministic invoice recomputation, a fail-closed server-to-server API, and compatibility wrappers for existing callers.
- Completed operation keys must be reconciled from the immutable event ledger before mutable lifecycle status can reject the same request; mismatched reuse must still conflict after the lifecycle advances.
- CI proof must exercise SQL behavior against PostgreSQL and inject configuration, service, repository, or transport collaborators only through explicit supported boundaries rather than patching first-party module globals.
- Secret-scan reconciliation must not rely on a PR-controlled ignore: the published range must contain no flagged fixture, and the trusted checker must reject every new ignore unless an explicit labeled security-only rotation authorizes it.
- MCP boundary proof must be independent of collection-order `sys.modules` fakes and must exercise the registered tool through public FastMCP APIs.
- Every receivables migration file covered by the PostgreSQL bundle must trigger the invoicing workflow, and readiness must fail closed when any required ledger column is missing.
- Returned or voided receipts must expose neither active allocation nor available unapplied credit; only physical checks may enter the received/check-deposit queue.
- An idempotent singular payment retry must reuse the receipt without repeating CRM history; the atomic replay outcome must cross the service/repository boundary without changing the public MCP payload.
- Every standalone finance-writer startup must apply pending migrations and prove the complete receivables schema ready before exposing MCP tools; migration or readiness failure must prevent that writer from starting.
- Readiness must prove both required columns and every explicit or constraint-backed migration index used for identity, uniqueness, idempotency, lookup, and deposit invariants.
- Live database availability failures must map to the structured 503 contract without converting validation, not-found, conflict, transaction/API misuse, or unrelated OS/programming failures.
- Gitleaks ignore growth must be evaluated by the trusted-base `pull_request_target` guard; any first or later addition requires a labeled, security-only rotation.
- Singular MCP idempotency keys must contain 1 to 128 characters and fail at the transport/service boundary before repository access.
- Allocation suggestions must be deterministic oldest-due-first even when one contact has invoices under different customer-name snapshots.
- It must also repair overdue selection because returned payments must re-enter the real reminder path.
- It must not change invoice generation, PDFs, email content, CRM ownership, customer-facing invoice shape, SaaS authentication, the read-only/draft-writer MCP permissions, or unrelated product lanes.

## Scope (this PR)

Ownership lane: eom-receivables

Slice phase: Vertical slice

Max files: 18

1. Add an additive, rerunnable payment-parent/allocation/deposit/audit migration with conservative legacy backfill.
2. Centralize atomic payment creation, allocation adjustment, deposit, clear, return, and void behavior in a receivables service.
3. Add a typed, fail-closed `/api/v1/receivables` surface for the EOM backend.
4. Preserve singular MCP behavior and add one explicit multi-invoice MCP tool through the same service.
5. Remove the unauthenticated quick-pay write and protect retained legacy invoice actions.
6. Fix balance-derived overdue/reminder selection and enroll the new behavior in invoicing CI.
7. Close the exact-head adjustment-replay finding and restore all maturity ratchets without updating their baselines.
8. Reconcile the historical test-fixture Gitleaks false positive by publishing a clean one-commit range with no ignore, without weakening the trusted baseline or scanner rule.
9. Make the strict multi-invoice MCP contract regression proof hermetic against repo-wide test-module contamination.
10. Enroll migration 345 in invoicing CI and close partial-schema readiness gaps without broadening the feature surface.
11. Make inactive-credit, non-check deposit eligibility, and singular-MCP replay side effects agree with the authoritative payment outcome.
12. Run all pending migrations before the standalone invoicing MCP exposes payment tools.
13. Require every explicit and constraint-backed receivables migration index in `/receivables/ready`, not columns alone.
14. Preserve the structured database-unavailable 503 contract for live driver/runtime outages.
15. Extend the trusted Gitleaks rotation gate to reject unexpected `.gitleaksignore` fingerprint growth.
16. Require full receivables readiness before standalone MCP serving, bound singular idempotency keys before repository access, and preserve oldest-due-first allocation suggestions.

### Review Contract

- Acceptance criteria:
  - [x] One customer payment atomically allocates across two or three invoices and preserves an unapplied remainder.
  - [x] Same-request retries do not duplicate money; mismatched or cross-payment event-key reuse conflicts, including concurrent deposit/clear attempts.
  - [x] An allocation-adjustment retry returns the committed result even after a later return or void, while changed or cross-payment replay payloads still conflict.
  - [x] Cross-customer, draft/void, non-positive, sub-cent, duplicate, and over-allocation inputs fail before mutation.
  - [x] Received, deposited, cleared, returned, and voided transitions deterministically recalculate balances and invoice statuses; returned/voided payment summaries expose allocations as history with zero active allocation and zero available credit.
  - [x] Legacy payment rows retain their public fields and continue contributing without inferred grouping.
  - [x] New HTTP routes reject missing/wrong credentials before repository access and report database unavailability.
  - [x] Legacy quick-pay returns 410 without writing even when the finance feature is disabled or unauthenticated; retained legacy actions require the new credential.
  - [x] Existing singular MCP calls retain their payload shape, reject idempotency keys outside 1 to 128 characters before repository access, log CRM history once across retries, bypass the check-deposit queue for compatibility-only non-check methods, and delegate multi-invoice writes to the domain service.
  - [x] Every balance recomputation locks affected invoices before its aggregate snapshot; a real two-connection PostgreSQL test covers concurrent create-versus-return.
  - [x] Already-overdue invoices remain visible to reminder processing.
  - [x] SQL behavior tests use real PostgreSQL, request/runtime tests use explicit dependency boundaries, and the API, storage, and MCP maturity ratchets pass without baseline changes.
  - [x] The clean one-commit PR history contains neither the flagged test-code occurrence of `mcp-boundary-1` nor a root `.gitleaksignore`, and the secret scan still uses the trusted base-branch baseline.
  - [x] The multi-invoice MCP schema and strict validation proof passes under the repo-wide Unit Gate even when the parent pytest process contains a legacy `FastMCP` fake.
  - [x] Both pull-request and main-push workflow filters enroll migration 345 in the receivables PostgreSQL checks.
  - [x] `/receivables/ready` returns unavailable when any migration-added ledger column is absent, including deposit-clear and allocation-reversal fields.
  - [x] Standalone invoicing MCP startup applies migrations after pool initialization, proves full receivables readiness, and fails before serving tools when migration or readiness fails.
  - [x] `/receivables/ready` fails closed for each required explicit or constraint-backed migration index, including every new table primary key, deposit-item uniqueness, and the partial unique index required by payment `ON CONFLICT` inference.
  - [x] Database connection, shutdown, capacity, closed/uninitialized pool, connection, or connection resource, OS-network, and timeout failures map to structured 503 responses while domain, released-resource/API-misuse, and unrelated OS/programming errors retain their existing handling.
  - [x] The trusted-base Gitleaks guard rejects every first or later ignore addition unless a labeled security-only rotation authorizes it.
  - [x] Suggested allocations are deterministic oldest-due-first for the selected contact even when invoice customer-name snapshots differ.
- Reachability proof: an authenticated FastAPI route test proves bearer/actor/idempotency propagation into the service; PostgreSQL tests prove parent/allocation persistence, same-key replay, invoice recomputation, deposit/clear/return, allocation adjustment, anomaly rejection, and create-versus-return serialization against the real query path.
- Affected surfaces: database migration, invoice repository/service, API, authentication, MCP, reminders, configuration, CI.
- Risk areas: money correctness, migration/backfill, authorization, concurrency, idempotency, backward compatibility, deployment ordering.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12, R14.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `CLAUDE.md`
- `atlas_brain/api/invoicing/__init__.py`
- `atlas_brain/api/invoicing/actions.py`
- `atlas_brain/api/invoicing/auth.py`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/config.py`
- `atlas_brain/main.py`
- `atlas_brain/mcp/invoicing_server.py`
- `atlas_brain/services/receivables.py`
- `atlas_brain/storage/migrations/344_receivables_payments.sql`
- `atlas_brain/storage/migrations/345_receivables_event_key_lookup.sql`
- `atlas_brain/storage/repositories/invoice.py`
- `plans/PR-Receivables-Ledger.md`
- `scripts/check_gitleaks_baseline_rotation.py`
- `tests/test_check_gitleaks_baseline_rotation.py`
- `tests/test_invoice_repository.py`
- `tests/test_receivables.py`

## Mechanism

`customer_payments` owns the physical/electronic receipt. Existing `invoice_payments` rows become compatible allocations by gaining a nullable parent ID and reversal fields. Deposit membership and lifecycle events are additive tables. New writes validate and lock all affected invoices in stable order inside one database transaction, write the parent/allocations/event, and recalculate invoice balances and state before commit.

Every event-producing mutation also takes one global advisory lock for its idempotency key before querying all payment events, so the same key cannot transition another payment under either sequential or concurrent requests. Currency normalization rejects values that are not already exact cents. Returned and voided responses retain allocation rows for audit while excluding them from active allocated/unapplied calculations. The retired quick-pay tombstone is registered on a separate ungated router, while all live invoice and receivables actions remain fail-closed.

Only checks begin in `received`; API ACH/Square and compatibility-only non-check methods begin in `cleared`, so deposit worklists cannot contain a method the deposit writer rejects. Payment creation also returns a typed internal first-write/replay outcome to the invoice repository. The singular MCP consumes that internal outcome to skip duplicate CRM logging on a retry, then emits the same payment dictionary as before.

Allocation adjustment reads and validates any existing event immediately after locking and loading the payment, before applying status-dependent admission to a new mutation. The replay contract therefore survives later lifecycle changes, while fingerprint validation continues to reject changed or cross-payment reuse. Repository SQL behavior is verified through a connection-scoped real PostgreSQL adapter; FastAPI configuration/service overrides and MCP core helpers expose only deliberate test/runtime seams, so tests no longer replace first-party module globals.

The receivables router is feature-gated and uses a dedicated typed service token with constant-time bearer comparison. The EOM backend supplies an idempotency key and trusted admin actor headers. New HTTP and multi-invoice MCP write-request inputs accept strict integer cents while PostgreSQL retains `NUMERIC(12,2)`; response dictionaries retain legacy dollar-valued compatibility fields alongside authoritative `*_cents` fields, and the portal consumes only cents. The existing singular MCP keeps its dollar-float argument for compatibility.

The MCP contract test starts a clean child interpreter so repo-wide collection-time `sys.modules` fakes cannot replace the installed SDK. Inside that process it discovers `record_customer_payment` through `FastMCP.list_tools`, checks the published input schema, and calls the tool through `FastMCP.call_tool` to prove coercive, non-positive, oversized, and empty inputs are rejected at the registered transport boundary. The existing real-PostgreSQL MCP test separately proves that valid multi-invoice input reaches the receivables service and persists correctly.

Readiness owns a declarative `(table, column)` manifest for every relation introduced or extended by migration 344. A single `information_schema.columns` anti-join returns false if any manifest entry is absent, so partially created preview tables cannot expose money controls. Real-PostgreSQL regression proof derives the expected table set independently from migration SQL, matches every introduced column, and proves that either missing fields or a wholly missing event table fail closed. Migration 345 is named explicitly in both invoicing workflow path lists; the test parses those event blocks directly and proves removing either enrollment is detected.

The standalone MCP lifecycle now initializes the pool, applies the repository migration runner, constructs the receivables service, and proves its complete schema readiness before yielding the tool server; any migration or readiness failure closes the pool and aborts startup. Readiness also compares the live catalog to a declarative signature for every explicit and constraint-backed index in migrations 344 and 345, including its table, uniqueness, ordered key columns, predicate, validity/readiness, and required primary-key/unique constraint type. The API adapter keeps domain errors ahead of a narrow database-availability classifier covering asyncpg connection/capacity/shutdown failures, explicit closed/closing/uninitialized pool or connection states, connection resources whose underlying connection closed, and concrete runtime network/timeout failures. Released-to-pool resources and other `InterfaceError` or `OSError` values remain visible as 500s with ordinary PostgreSQL/data/programming errors.

The existing `pull_request_target` Gitleaks growth job checks out and executes the trusted base script against the fetched PR head. The future trusted script treats non-comment `.gitleaksignore` fingerprints as a second protected set: removals are safe, while every first or later addition requires the `security-rotation` label plus the existing security-only path restriction. This PR publishes a clean single-commit range without a root ignore, so its own checker change is not relied upon to excuse this PR's scan.

The singular MCP validates its optional idempotency key at 1 to 128 characters in the registered schema and normalizes it before any repository lookup; the service repeats the limit as the authoritative write invariant. Open invoices are ordered by due date, issue date, customer name, and invoice number, so the greedy suggestion loop cannot prefer a newer invoice merely because its customer-name snapshot sorts first.

## Intentional

- Legacy rows are backfilled one-to-one as `legacy`; matching references are never grouped.
- `payment_id` remains nullable for one release; balance recomputation treats a null parent as an active legacy allocation and the adoption trigger gives a late legacy row a visible parent. This is a data-visibility bridge only. Old and new finance writers must not overlap because the pre-receivables writer performs an unlocked, separate balance refresh that can overwrite a newer total.
- Unapplied credit is visible at customer/payment level but does not reduce an invoice until explicitly allocated.
- Check images and automatic ACH/Square provider sync are not part of this slice.
- The dedicated internal token is separate from SaaS auth and MCP auth because those dependencies have different fail-open/scope contracts.
- Maturity baselines are not updated; the new debt is removed at its source.
- No root `.gitleaksignore` is added. The historical false-positive commit is removed from the published PR range while the trusted JSON baseline and scanner rule remain untouched.
- The required status context remains named `Gitleaks baseline growth guard`; its trusted checker now covers both the JSON baseline and root ignore growth so branch-protection configuration does not drift.

## Deferred

- A later contract migration may make `invoice_payments.payment_id` non-null after every writer has rolled forward.
- Provider webhooks, bank-file matching, parent-payer accounts, and check-image storage require separate operator-approved slices.
- EOM proxy and portal changes ship in their own repository branches but are part of the coordinated release.
- Parked hardening: none; auth, idempotency, migration safety, and lifecycle audit are required money-risk controls for this vertical path.

## Coordinated deployment and rollback

Production token and environment provisioning are external prerequisites and are not asserted by this code change. The code-derived safe rollout order is:

1. Start a finance-write maintenance window: disable portal/MCP payment entry and drain every pre-receivables Atlas application and invoicing-MCP writer. Do not enable the new API while any old writer can still accept payment calls.
2. Deploy this Atlas revision to every finance-writer process with the receivables API disabled. Let startup apply migrations 344 and 345 in both application and standalone MCP processes, and verify that no old process remains. Then configure a non-placeholder `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN`, set `ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true`, restart, and require an authenticated `/api/v1/receivables/ready` response before ending the maintenance window.
3. Deploy the EOM backend with the matching `ATLAS_RECEIVABLES_SERVICE_TOKEN` and an `ATLAS_RECEIVABLES_BASE_URL` ending in `/api/v1`. Require `/api/health` and an authenticated `/api/admin/receivables/ready` response before exposing the portal UI.
4. Deploy the EOM portal last and run the operator smoke against the backend. The base portal has no receivables UI, so deploying it before the proxy would expose controls whose routes do not yet exist.

Normal disablement reverses the consumer order: roll back the portal first to remove money-write controls, roll back/disable the EOM proxy second, and disable the Atlas feature flag last. Migration 344 and this Atlas writer implementation must remain in place once new receipt, reversal, or allocation-history rows exist; restarting the pre-receivables writer can miscount reversed allocations and is not a supported rollback. Rollback never deletes receipt history. For an emergency write stop, disabling the Atlas feature flag immediately makes the finance API fail closed with 503 while the UI/backend may remain deployed for diagnosis.

## Verification

- Clean-process MCP contract probe passes under Python 3.11.15 with the CI-pinned MCP 1.28.1/Pydantic 2.13.4 runtime; the exact pytest node also passes with the parent process preloaded with the legacy passthrough `FastMCP` fake.
- `python -m pytest tests/test_receivables.py tests/test_invoice_repository.py -q` with `ATLAS_RECEIVABLES_TEST_DATABASE_URL` against ephemeral PostgreSQL 16 (62 passed, including the real-PostgreSQL tests and the review regressions). The entrypoint proof observes one first-write/one replay outcome under concurrency, one CRM call across a singular retry, zero inactive available credit, a compatibility card receipt absent from the received-check worklist, and oldest-due-first suggestion across differing customer-name snapshots while preserving the public MCP payload.
- Migration/readiness proof parses both exact workflow path lists with negative controls, derives the table set from migration 344, matches every introduced column, and proves missing deposit-clear fields, allocation-reversal fields, or the complete events table each fail closed.
- MCP lifecycle proof observes initialization, migration, and full readiness in order, and proves either migration failure or incomplete-schema readiness aborts startup and closes the pool.
- Real-PostgreSQL readiness proof transactionally removes every explicit and constraint-backed migration index, proving each omission fails closed before rollback restores the schema.
- Parametrized API proof covers multiple asyncpg connection/capacity/shutdown classes, explicit unavailable pool/connection interface states, network errnos, and timeouts. A real PostgreSQL probe terminates a backend inside a live transaction and proves the resulting closed-resource interface error maps to 503; negative controls preserve domain statuses and let released-resource/API misuse, unrelated OS failures, and unexpected errors escape.
- Trusted Gitleaks rotation tests reject unapproved initial and later growth, permit comments/removals, and allow only labeled security-only rotation.
- Focused receivables, repository, and rotation unit suite passes (72 passed, six expected database skips); security policy/workflow coverage passes (85 passed, 43 subtests).
- Invoicing MCP/OAuth surface suite (43 passed).
- Monthly invoice approval-blocker selection (2 passed).
- API, storage, and MCP maturity ratchets pass without baseline changes.
- Gitleaks PR-range scan passes the clean single-commit branch against the trusted baseline with no `.gitleaksignore` and reports no leaks.
- Real-PostgreSQL migration parse, one-to-one legacy backfill, and rolling-writer adoption in a temporary schema dropped after the test.
- Real-PostgreSQL receipt/replay/deposit/clear/return, adjustment/anomaly, and concurrent-key flows in temporary schemas dropped after each test.
- Companion backend suite `python -m pytest -q` (116 passed) covers canonical deposit retries, duplicate-ID rejection, Pydantic v2 runtime requirements, and retryable non-JSON upstream outages in addition to the original proxy contract.
- Companion portal suite `npm test` (12 passed) includes a jsdom reachability test that loads the real `portal.html`, initializes the real receivables module, clicks the actual payment control, verifies the two-invoice cents payload/idempotency key, and observes in-flight control locking; it also proves failed refreshes invalidate every stale money surface and definitive 4xx payment rejections refresh authoritative invoices before unlocking. The production-like coordinated browser smoke remains a release-time gate in the deployment sequence above and is not asserted by this PR.
- `python scripts/sync_pr_plan.py plans/PR-Receivables-Ledger.md --check` after synchronizing the plan.
- `git diff --check`.
- Atlas pre-push/local-review gate through the repository wrapper if publishing is requested.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 37 |
| `CLAUDE.md` | 20 |
| `atlas_brain/api/invoicing/__init__.py` | 10 |
| `atlas_brain/api/invoicing/actions.py` | 67 |
| `atlas_brain/api/invoicing/auth.py` | 73 |
| `atlas_brain/api/invoicing/receivables.py` | 341 |
| `atlas_brain/config.py` | 8 |
| `atlas_brain/main.py` | 3 |
| `atlas_brain/mcp/invoicing_server.py` | 283 |
| `atlas_brain/services/receivables.py` | 1629 |
| `atlas_brain/storage/migrations/344_receivables_payments.sql` | 221 |
| `atlas_brain/storage/migrations/345_receivables_event_key_lookup.sql` | 7 |
| `atlas_brain/storage/repositories/invoice.py` | 211 |
| `plans/PR-Receivables-Ledger.md` | 215 |
| `scripts/check_gitleaks_baseline_rotation.py` | 99 |
| `tests/test_check_gitleaks_baseline_rotation.py` | 91 |
| `tests/test_invoice_repository.py` | 225 |
| `tests/test_receivables.py` | 2101 |
| **Total** | **5641** |
