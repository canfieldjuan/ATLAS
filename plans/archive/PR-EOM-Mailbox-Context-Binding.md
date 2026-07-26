# PR Plan: EOM Mailbox Context Binding

## Why this slice exists

Issue #2177 requires scoped CRM customer context to read inbox history only
from a mailbox explicitly authorized for that exact `business_context_id`.
Today the unscoped email provider is a global IMAP/Gmail composite, so using it
under a tenant scope could expose another operator mailbox.

The original implementation widened this privacy slice into a bespoke durable
Gmail refresh-token protocol. Review rounds then clustered on that subsystem:
cross-process leases, file-system race resistance, cancellation draining,
generation ancestry, and cache synchronization. Those concerns are real, but
they are not required to prove the authorization boundary. This revision
removes scoped Gmail from this PR and ships the smallest complete proof with
explicit IMAP bindings.

The slice remains above the 400-LOC soft cap because the authorization
boundary is indivisible across typed configuration, provider construction,
aggregation, MCP serialization, and a real-entrypoint regression. Most of the
churn is the focused behavioral proof; splitting those tests from the runtime
change would leave the privacy boundary unproven at merge.

### Problem-derived contract

Root cause: scoped CRM aggregation has no exact tenant-to-mailbox authorization
mapping and otherwise reaches the global email reader.

A correct fix must:

1. Define a validated IMAP credential binding keyed by the exact, nonblank CRM
   business-context ID.
2. Refuse an unmapped context before any mailbox connection or query.
3. Construct only the bound IMAP reader; never fall back to global IMAP or
   Gmail for a scoped request.
4. Admit only messages with one structurally valid `From` author that exactly
   matches the contact address. Legal RFC folding must be unfolded, while bare
   CR/LF injection, groups, duplicate fields, and multiple authors fail closed.
   The final result cap must not truncate IMAP candidates before admission.
5. Lift the scoped inbox omission flag only for a mapped source, and discard
   results if the contact ownership or queried address changes during I/O.
6. Preserve unscoped email behavior and every unrelated CRM source limit.

This fixes the authorization root directly. It must not add scoped Gmail token
rotation, durable token files, cross-process locking, a scoped provider cache,
new outbound behavior, or a database-secret schema by implication.

## Scope (this PR)

Ownership lane: eom-crm/email-tenancy

Slice phase: Production hardening

1. Add exact-key, IMAP-only typed inbox bindings with redacted secrets.
2. Build a fresh bound IMAP reader per scoped resolution; no authorization
   cache or invalidation protocol.
3. Route scoped customer-context inbox reads through that reader and preserve
   fail-closed omission semantics.
4. Enforce exact raw-sender admission with legal header unfolding.
5. Prove the real CRM MCP entrypoint reads only the bound mailbox and refuses
   unmapped contexts.
6. Remove every scoped Gmail/token-store change from this PR.

### Files touched

- `CLAUDE.md`
- `atlas_brain/config.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/customer_context.py`
- `atlas_brain/services/email_provider.py`
- `plans/PR-EOM-Mailbox-Context-Binding.md`
- `tests/test_eom_mailbox_context_binding.py`

### Review Contract

Acceptance criteria:

- Exact, whitespace-distinct context IDs cannot alias one binding.
- Invalid or unsupported bindings fail validation without exposing credential
  values in string or structured errors.
- An unmapped scoped request performs zero IMAP/Gmail operations and reports
  `inbox_emails` as omitted.
- A mapped scoped request constructs only its bound IMAP reader and never uses
  the global composite fallback.
- Legal `CRLF + WSP` and `LF + WSP` folding is admitted after unfolding; bare
  CR/LF, duplicate fields, groups, and multiple authors remain rejected.
- A collision or malformed candidate cannot hide a later exact sender, and a
  malformed candidate cannot discard the rest of the batch.
- Mailbox domains compare case-insensitively while local parts remain
  case-sensitive in both sender admission and the post-I/O address guard.
- Rebinding takes effect on the next request because no scoped provider is
  cached.
- Unscoped inbox behavior and non-mailbox aggregation limits are unchanged.
- The real `get_customer_context` MCP entrypoint returns an admitted bound
  message and clears the omission flag; the same entrypoint refuses an
  unmapped context before mailbox I/O.

Affected surfaces:

- Email configuration validation.
- Scoped IMAP provider construction and raw sender evidence.
- Customer-context aggregation and CRM MCP serialization.

Risk areas:

- Tenant authorization and secret redaction (R3).
- Legacy compatibility and exact sender parsing (R5).
- Source-local errors and observable omission state (R6).
- Bounded mailbox work and no stale authorization cache (R7/R8).
- Configuration and deployment documentation (R11/R12).
- Parser guard class closure and cold reconstruction (R13/R14).

Triggered reviewer rules: R1, R2, R3, R5, R6, R7, R8, R10, R11, R12,
R13, R14.

Reachability proof: call the real CRM MCP `get_customer_context` function with
a tenant-owned contact and an injected bound IMAP adapter; assert the serialized
inbox result and omission fields. Repeat with an unmapped context and assert
zero mailbox access.

## Mechanism

`EmailConfig.inbox_context_bindings` maps exact context strings to an
`InboxMailboxBinding` containing only IMAP connection fields. The scoped
binding preflights every allowed scalar into a typed value or a secret-free
sentinel before Pydantic can render structured errors. The scoped factory
validates the context, looks up the current binding on every call, and returns
`IMAPEmailProvider(binding)`. Construction is in-memory and performs no mailbox
I/O, so no cache or global construction lock is needed.

The customer-context service resolves the binding before scheduling inbox work.
IMAP operations already run in worker threads. A scoped read requests at most
50 candidates, isolates parsing per message, and stops when the caller's result
cap is filled. Scoped results carry private raw `From` evidence; admission
unfolds only legal continuation sequences, rejects any remaining CR/LF, parses
one non-group author, and compares the normalized domain case-insensitively
without folding local-part case.

## Intentional

- Scoped Gmail is unsupported in this slice and fails typed validation.
- A fresh scoped IMAP provider is cheap and makes binding changes effective on
  the next request without cache invalidation.
- Network/provider failures remain source-local and return an empty mapped
  inbox; absence of a binding is separately reported as omission.
- Unscoped email tooling retains its existing global composite behavior.
- Sender admission remains fail-closed after legal folding is normalized.
- Candidate discovery is capped at 50 independently of the final result cap,
  so rejected near matches cannot starve exact senders within that bound.
- The MCP reachability proof leaves unrelated repository factories real and
  exercises their existing unavailable-store isolation through `_safe`; it
  mocks only the CRM fixture and external IMAP edge.

## Deferred

- #2196 is the scoped Gmail follow-up with database-backed token state and its
  own migration, repository contract, concurrency semantics, and integration
  proof.
- The existing `BusinessContextRepository` is not directly suitable for OAuth
  secrets: it uses broad `SELECT *`, has no token column or encryption
  boundary, and its `upsert` rewrites the full business-context row. The
  follow-up must design a narrow credential store rather than placing secrets
  in the existing row by convention.
- Production IMAP credential provisioning remains an operator deployment
  action; no secrets are committed.
- Parked hardening: none.

## Verification

- pytest -q tests/test_eom_mailbox_context_binding.py -- 37 passed.
- pytest -q tests/test_eom*.py tests/test_crm_read_scoping.py
  tests/test_customer_context.py -- 158 passed, 5 skipped.
- pytest -q tests/test_auth_api_keys.py tests/test_byok_keys.py
  tests/test_eom_mailbox_context_binding.py -- 108 passed; proves a config
  module reload does not reject a previously constructed typed binding.
- `pytest -q tests/test_mcp_servers.py::TestCompositeEmailProvider
  tests/test_mcp_servers.py::TestCompositeProviderIMAPPreference` -- 5 passed;
  the unscoped IMAP-to-Gmail composite behavior is unchanged.
- The broad `tests/test_mcp_servers.py` run reports 74 passed and 6 failures
  already present in the current `origin/main` source/test pairing: two
  redacted-error expectation failures, two stale IMAP executor-call
  expectations, and two calendar mock-shape failures. None touches this
  scoped factory or its focused proof.
- `python scripts/check_guard_class_closure.py --base origin/main --strict` --
  passed.
- Changed-module compilation, `git diff --check`, and
  bash scripts/check_ascii_python.sh -- passed.
- Full unit ratchet -- 163 failing/errored nodes against a baseline of 182,
  with 20 stale baseline entries and one intermittent unbaselined
  `tests/test_nemotron_stt.py::test_nemotron_stt`; the exact node passed on
  immediate isolated replay.
- Storage maturity ratchet -- passed after removing four first-party
  repository mocks from the MCP reachability proof; no baseline debt added.
- Independent exact-snapshot re-review -- LGTM; held-out probes cover the
  candidate-50 boundary, candidate-51 exclusion, early result-cap stop,
  arbitrary parser exceptions, and domain/local-part case semantics.

## Estimated diff size

| File | LOC |
|---|---:|
| `CLAUDE.md` | 11 |
| `atlas_brain/config.py` | 154 |
| `atlas_brain/mcp/crm_server.py` | 42 |
| `atlas_brain/services/customer_context.py` | 205 |
| `atlas_brain/services/email_provider.py` | 67 |
| `plans/PR-EOM-Mailbox-Context-Binding.md` | 209 |
| `tests/test_eom_mailbox_context_binding.py` | 513 |
| **Total** | **1201** |
