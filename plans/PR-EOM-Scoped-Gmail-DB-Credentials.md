# PR-EOM-Scoped-Gmail-DB-Credentials

## Why this slice exists

Issue #2196 is the database-backed half of the scoped-mailbox work split out of
PR #2184. That merge deliberately shipped only IMAP because the abandoned
scoped-Gmail implementation turned one global file token store into a bespoke
cross-process filesystem protocol. Scoped Gmail still cannot be enabled
without either borrowing the global account or reintroducing that protocol.

This is production hardening because it closes a live privacy/authorization
risk: per-context OAuth refresh state must survive restarts and concurrent
workers without crossing tenants or losing a rotated token.

The estimated diff exceeds the 400-LOC soft target because the security
boundary is not safe to split into a floating repository or an unverified
runtime adapter. The additive schema, encrypted repository, scoped provider
wiring, real-entrypoint PostgreSQL proof, CI enrollment, and operator docs must
land together. The implementation is capped to this one credential row and
one read path; it does not add a provisioning API, UI, generic OAuth framework,
or file-lock compatibility layer.

The review/CI fix round closes two root causes rather than baselining symptoms:
the cancellation drain let a later child exception replace an already-recorded
caller cancellation, and two new tests retained config classes/settings across
the existing BYOK suite's intentional `atlas_brain.config` reload. The runtime
now gives recorded cancellation precedence while retaining the child failure
as its cause; the tests resolve the current config classes and patch the
settings object actually owned by the global token store.

### Problem-derived contract

- Root cause: the global `GoogleTokenStore` owns one shared file/env credential
  and a process-local lock, while scoped inbox reads need exact-context,
  encrypted, restart-durable state whose refresh-token rotation is serialized
  across processes. The current scoped factory is IMAP-only and synchronous,
  and the global Gmail client/provider cannot be reused because it fetches the
  shared account, caches globally, persists rotation to the shared file, and
  returns message IDs without the sender evidence required by scoped
  admission.
- Correct fix must touch/change: add a narrow replay-safe table keyed by exact
  `business_context_id` and provider; store one authenticated-encrypted Gmail
  credential bundle with a KEK id, monotonic generation, and durable revoke
  state; add repository bind/read/rebind/revoke operations plus a
  transaction-held `FOR UPDATE` refresh lease; allow a secret-free
  `provider="gmail"` context binding; inject that lease into a fresh scoped
  Gmail client/provider; preserve every Gmail `From` header value for the
  existing strict sender admission; make scoped provider resolution async and
  fail closed before external Gmail I/O when credentials are absent/revoked;
  enroll a real PostgreSQL test that drives the CRM MCP entrypoint through
  bind, read, rotation, reconstructed-service restart, concurrent refresh,
  rebind, and revoke.
- Must not change: the global file/env `GoogleTokenStore`, global Gmail client
  singleton semantics, Calendar OAuth, unscoped/composite inbox reads, outbound
  Gmail/Resend routing, IMAP binding behavior, MCP response shape, contact
  ownership policy, `business_contexts` rows/repository, or customer-facing
  product shape. OAuth provisioning UI/API and KEK-rotation jobs are outside
  this slice.

## Scope (this PR)

Ownership lane: eom-crm/email-tenancy
Slice phase: Production hardening
Max files: 17
<!-- raised from 14: round-2 fixes added
     tests/test_eom_scoped_gmail_hardening.py and CLAUDE.md; round 3 added
     the migration-runner advisory lock -->

1. Add one encrypted, generation-aware scoped Gmail credential row and
   transaction-serialized refresh path for exact CRM business contexts.
2. Extend the already-shipped scoped inbox seam with a fresh Gmail provider
   that never reaches the global token store or global composite provider.
3. Prove the durable lifecycle and authorization boundary through the real
   `crm_server.get_customer_context` entrypoint with only PostgreSQL and Google
   HTTP as the real/faked external boundaries.
4. Archive the merged predecessor plan by its exact filename and refresh the
   plan index, as required by AGENTS.md teardown.

### Review Contract

- Acceptance criteria:
  - Migration 350 is additive and replay-safe, enforces exact nonblank context,
    provider `gmail`, positive generations, and one row per
    `(business_context_id, provider)`.
  - Client ID, client secret, and refresh token are stored only inside one
    Fernet-authenticated ciphertext tagged with its KEK id; lookup/decrypt
    failures return no usable credential and never log plaintext/ciphertext.
  - Bind/rebind and revoke atomically increment a monotonic generation. Revoke
    cannot create an ABA generation, and active reads exclude revoked rows.
  - Refresh holds a PostgreSQL row lock across the external token exchange and
    persists any rotated bundle before releasing the transaction, so two
    processes cannot refresh/rotate one context concurrently.
  - The execution model covers arbitrary concurrent scoped reads, refreshes,
    rebinds, and revokes against one credential row; the PostgreSQL row lock
    and transaction define their order, while cancellation and process-loss
    behavior fail closed under the explicit external-token assumption below.
  - `provider="gmail"` bindings contain no OAuth secret material. Unmapped,
    missing, undecryptable, and revoked credentials fail closed without using
    the global token store/composite provider.
  - Each scoped request constructs a fresh provider whose refresh lease reads
    the current row after any rebind; no authorization cache survives requests.
  - Gmail candidate hydration preserves all `From` header values, so the
    existing duplicate/ambiguous/exact-address admission guard applies before
    caller result limits.
  - Global Gmail, Calendar, outbound email, IMAP, unscoped customer context,
    and MCP response fields remain behaviorally unchanged.
  - The real PostgreSQL/MCP proof observes an admitted exact Gmail message,
    durable rotation and restart reuse, serialized concurrent refresh,
    immediate rebind, and revoked omission.
- Reachability proof: call the real async
  `atlas_brain.mcp.crm_server.get_customer_context` with a tenant contact and
  `business_context_id`, use the real credential repository and PostgreSQL
  transaction path, fake only Google's token/Gmail HTTP endpoints, and assert
  the serialized MCP response plus persisted generation/ciphertext state.
- Affected surfaces: email settings/binding validation, scoped email-provider
  construction, Gmail read-client dependency injection and sender metadata,
  customer-context scoped provider resolution, PostgreSQL migration/repository,
  EOM PostgreSQL CI enrollment, documentation, and focused tests.
- Risk areas: plaintext/error leakage, exact-context aliasing, global-credential
  fallback, stale provider state after rebind/revoke, refresh races/cancellation,
  generation ABA, Gmail duplicate-header collapse, pagination starvation,
  migration replay/deploy order, and accidental global Gmail/outbound changes.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12,
  R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `CLAUDE.md`
- `atlas_brain/autonomous/tasks/gmail_digest.py`
- `atlas_brain/config.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/customer_context.py`
- `atlas_brain/services/email_provider.py`
- `atlas_brain/storage/migrations/350_scoped_mailbox_credentials.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `atlas_brain/storage/repositories/scoped_mailbox_credential.py`
- `plans/INDEX.md`
- `plans/PR-EOM-Scoped-Gmail-DB-Credentials.md`
- `plans/archive/PR-EOM-Mailbox-Context-Binding.md`
- `tests/test_eom_mailbox_context_binding.py`
- `tests/test_eom_scoped_gmail_credentials.py`
- `tests/test_eom_scoped_gmail_hardening.py`
- `tests/test_migrations_runner.py`

## Mechanism

Migration 350 creates one row per exact context/provider. The row stores a
single encrypted JSON credential bundle, its KEK id, a generation counter, and
revocation timestamps. Rebinding overwrites that same row and increments its
generation; revocation marks it revoked and increments again, so a later
rebind cannot reuse a stale generation.

The repository reuses Atlas's existing BYOK Fernet boundary. Its refresh lease
opens a normal `DatabasePool.transaction()`, selects the active row `FOR
UPDATE`, decrypts only inside the repository, and yields a narrow lease whose
rotation method rewrites the encrypted bundle and increments generation on the
same connection. The Gmail client holds that lease while exchanging a refresh
token, so PostgreSQL—not a file or process lock—serializes cross-process
rotation and transaction cleanup handles exceptions/cancellation.

`InboxMailboxBinding(provider="gmail")` is a secret-free selector. The scoped
binding normalizer preserves the provider-only shape for both raw mappings and
typed binding models. The scoped provider factory exact-looks up that selector,
confirms an active decryptable row, and returns a fresh read-only
`ScopedGmailEmailProvider` backed by a fresh injected `GmailClient`; the global
singleton/store remain the default when no source is injected. Decryption and
provider-setup failures are contained without interpolating secret-bearing
configuration exceptions into logs. Scoped Gmail listing hydrates candidate
metadata while preserving every `From` value under the existing private
evidence key, then the existing customer-context
admission/limit/address-race code filters and strips it.

### Execution model

The selected closed-surface component is one PostgreSQL transaction over one
credential row, using the database's atomic upsert/update, `FOR UPDATE` row
locking, commit/rollback, and connection-loss lock release. No application
file lease, clock, expiring lock, or background retry protocol is added. The
one execution surface in this slice is refresh/rebind/revoke ordering for that
row; Gmail message reads occur only after the refresh transaction commits.

The model admits arbitrary concurrent read requests and repository
bind/rebind/revoke calls from multiple processes, cancellation at any
application await, repeated or out-of-order requests, HTTP failure/timeout,
and process loss. The initial availability read is advisory, never the
authorization decision: the refresh transaction rereads the exact active row
under `FOR UPDATE`. Operations on the same `(business_context_id, provider)`
row therefore take the database-defined lock order; operations on different
rows remain independent. A waiting refresh reads the predecessor's committed
rotation, while a waiting rebind/revoke takes effect before the next refresh.
Generation changes occur in the same atomic row mutation as the encrypted
bundle or revoke marker, so every admitted interleaving preserves exact-context
ownership, monotonic generation, and fail-closed active-row selection.

**Stated restrictions on the provisioning surface.** Two restrictions bound
this model rather than being enforced by it, because the only caller of
`bind_gmail`/`revoke_gmail` today is an operator running a snippet in a
protected runtime (CLAUDE.md), not a retrying or queued API:

1. *One active credential per underlying Google identity.* The row lock
   serializes by `(business_context_id, provider)`. Two contexts provisioned
   with the SAME refresh token take different locks, so a rotation persisted
   for context A leaves context B holding the superseded token; B then fails
   closed until an operator rebinds it. Binding one Google credential to two
   contexts is therefore out of contract. Enforcing it needs a deterministic
   credential fingerprint plus a partial unique index over active rows, which
   belongs with the provisioning API that can actually admit the mistake.
2. *No mutation fencing by generation or idempotency identity.* `bind_gmail`
   clears `revoked_at` unconditionally, so if bind/revoke were ever delivered
   out of order or retried by a queue, arrival order would decide the
   authorization outcome. No such caller exists in this slice; the monotonic
   generation records order but does not gate on it. Fencing belongs with the
   provisioning API for the same reason.

In-process cancellation is drained only after the shielded child completes the
lease, token exchange, optional encrypted rotation, and transaction exit.
Recorded caller cancellation takes precedence even when that drained child
fails; the child failure is retained as the cancellation cause rather than
turning cancellation into an ordinary mailbox error.
Ordinary exceptions and process/connection loss roll the database transaction
back and release its lock; no partial database mutation becomes visible. A
replayed request starts from the currently committed row rather than cached
authorization state. The external Google exchange and PostgreSQL commit cannot
form one distributed transaction: this model assumes a refresh token returned
by a successful Google refresh is not made the sole usable token before the
client can durably store it. If Google independently invalidates the committed
refresh token, including during a process crash after exchange but before
commit, the next read fails closed and requires operator rebind; this slice
does not promise cross-system crash recovery or availability in that case.
Direct SQL writers outside this repository are also outside the model.

## Intentional

- Reuse `ATLAS_SAAS_BYOK_ENCRYPTION_KEK` and the existing Fernet helper rather
  than inventing a second cipher/key format. Scoped Gmail fails closed until a
  non-sentinel KEK is configured; no plaintext fallback exists.
- Keep one encrypted JSON bundle instead of three independent ciphertext/kid
  pairs. Refresh-token rotation re-encrypts the complete bundle under the
  current write KEK, keeping key rotation coherent.
- Hold a database row lock during the external token refresh. Access-token
  refresh is infrequent, scoped to one context/provider row, and correctness
  under rotation is more important than allowing concurrent refresh calls for
  the same mailbox; other contexts remain independent.
- Add no admin credential API or settings response. Operational provisioning
  calls the repository directly for now, so this security slice does not
  create an unaudited secret-write HTTP surface.
- Apply the predecessor review's one-line ASCII NIT in the already-touched
  mailbox test by spelling the SMTPUTF8 address with an ASCII escape; behavior
  is unchanged.
- Archive only `plans/PR-EOM-Mailbox-Context-Binding.md`; do not bulk-archive
  concurrent sessions' plans.

## Deferred

- An authenticated operator provisioning/rotation/revocation CLI or API is a
  separate product/operations slice; this PR exposes no new secret-write
  network surface.
- Online KEK re-encryption/retirement remains the existing BYOK operational
  concern; the stored kid keeps rows readable during configured key overlap.
- Gmail batch metadata optimization is deferred unless live latency shows the
  bounded candidate hydration is material; the correctness path limits
  candidates to the existing scoped cap.

Parked hardening: none.

## Review follow-up (round: four Codex findings, all fixed)

1. **Sanitized setup-failure reason** -- provider setup failures now log the
   exception class name (never the message, which can carry credential text),
   so a dead pool is distinguishable from a missing migration in production
   logs. `tests/test_eom_scoped_gmail_hardening.py` proves the class appears
   and a secret-bearing message does not.
2. **Late revocation is an omitted source** -- `ScopedMailboxCredentialUnavailable`
   raised between the advisory availability check and the locked read now
   propagates to the aggregation, which records `inbox_email_source_omitted`
   instead of returning an indistinguishable empty inbox. A transient provider
   failure still reads as an ordinary empty result (both directions tested).
3. **Standalone CRM server migrates at startup** -- `_lifespan` now applies
   migrations when the pool is initialized (mirroring the invoicing MCP), so
   migration 350 is present before the first scoped Gmail read; a disabled-DB
   deployment logs a warning instead of skipping silently. CLAUDE.md states no
   separate migration step is needed.
4. **Refresh waiters are bounded** -- an in-process per-(loop, context) gate in
   front of `locked_gmail` queues concurrent refreshes WITHOUT pool
   connections; cross-process serialization stays on FOR UPDATE. Ten
   concurrent scoped reads for one context now hold at most one connection per
   process. The paired probe defeats the gate and reproduces the reviewed
   pool-exhaustion profile, proving the test measures the gate. Residual,
   stated: each *process* still contributes one blocked connection per context
   while another process refreshes -- the same bound the FOR UPDATE design
   always had.

## Verification

- PASS — changed-module compilation:

      python -m py_compile atlas_brain/autonomous/tasks/gmail_digest.py atlas_brain/services/email_provider.py atlas_brain/services/customer_context.py atlas_brain/storage/repositories/scoped_mailbox_credential.py atlas_brain/config.py
- PASS — focused mailbox/Gmail behavior:
  `python -m pytest tests/test_eom_scoped_gmail_credentials.py
  tests/test_eom_mailbox_context_binding.py -q` (42 passed), including
  cancellation drainage, unconfigured-global no-allocation behavior, typed
  Gmail binding normalization, malformed-KEK log redaction, and the unchanged
  global token-store rotation path.
- PASS — cancellation-plus-child-failure regression proving recorded caller
  cancellation remains the primary outcome after the child and lease finish
  (2 passed with the existing cancellation-success regression).
- PASS — exact config-reload pollution reproducer:
  BYOK reload -> mailbox config -> global Gmail rotation (3 passed under CI's
  Pydantic/Pytest versions).
- PARTIAL — local repo-wide unit ratchet no longer reports either PR-added EOM
  test as a regression. The Python 3.13 host does not exactly match CI's
  Python 3.11 baseline: it reports one unrelated async-test regression and 20
  environment-dependent baseline tests passing, so the exact CI ratchet
  remains the publication gate; no baseline file was changed.
- PASS — real PostgreSQL/MCP lifecycle with an ephemeral PostgreSQL 16
  container and
  `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas:atlas@localhost:55432/atlas_migration_tests`;
  the live test passed bind, missing/undecryptable refusal, encrypted read,
  rotation, restart, concurrent serialization, rebind, and revoke.
- PASS — exact expanded EOM workflow command from
  `.github/workflows/atlas_eom_lead_pipeline_checks.yml` (167 passed).
- PASS — EOM workflow plus auth-email configuration and tenant-stamping
  regression tests (181 passed).
- PASS — Ruff lint on all changed Python files; autonomous-task maturity
  ratchet unchanged; new storage repository maturity sweep has zero findings;
  migration-prefix test passes; `git diff --check` passes.
- PASS — cold reconstruction found no untraced change, missing contract item,
  or forbidden product/global-mail touch; plan/code sync and claim audit pass.
- PASS — managed local-review bundle with the canonical PR body (all mechanical
  gates passed; cross-layer caller hints were inspected against the focused
  global-Gmail and real MCP proofs).
- PASS — independent review findings reproduced and fixed: malformed KEK
  configuration now fails closed without secret-bearing logs, typed Gmail
  binding models normalize to the accepted provider-only shape, recorded
  cancellation survives a later child failure, and reload-order evidence
  resolves the live config/settings owners.
- PASS — independent pre-publication review returned LGTM with 0 NITs after
  held-out cancellation races, a real-PostgreSQL cancellation/rollback/unlock
  probe, the BYOK reload sequence, caller sweeps, and all triggered rules.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 21 |
| `CLAUDE.md` | 33 |
| `atlas_brain/autonomous/tasks/gmail_digest.py` | 199 |
| `atlas_brain/config.py` | 20 |
| `atlas_brain/mcp/crm_server.py` | 73 |
| `atlas_brain/services/customer_context.py` | 51 |
| `atlas_brain/services/email_provider.py` | 86 |
| `atlas_brain/storage/migrations/350_scoped_mailbox_credentials.sql` | 29 |
| `atlas_brain/storage/migrations/__init__.py` | 175 |
| `atlas_brain/storage/repositories/scoped_mailbox_credential.py` | 373 |
| `plans/INDEX.md` | 1 |
| `plans/PR-EOM-Scoped-Gmail-DB-Credentials.md` | 363 |
| `plans/archive/PR-EOM-Mailbox-Context-Binding.md` | 0 |
| `tests/test_eom_mailbox_context_binding.py` | 39 |
| `tests/test_eom_scoped_gmail_credentials.py` | 723 |
| `tests/test_eom_scoped_gmail_hardening.py` | 678 |
| `tests/test_migrations_runner.py` | 394 |
| **Total** | **3258** |
