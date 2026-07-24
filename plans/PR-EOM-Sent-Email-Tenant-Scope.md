# PR-EOM-Sent-Email-Tenant-Scope

## Why this slice exists

Issue #2171 records a privacy boundary that #2157 deliberately left fail
closed: scoped CRM customer context drops sent email, inbox email, and B2B
enrichment because none can prove CRM tenant ownership. Repository inspection
found a second root problem beneath that omission: `sent_emails` has no
production writer, so adding a tenant column alone would still leave the EOM
acknowledgement absent from history. This production-hardening slice is
allowed now because it both closes the concrete cross-tenant privacy gap for
one source and proves the existing EOM lead flow end to end.

Diff-budget override: the additive migration, repository boundary, real
acknowledgement writer, scoped aggregate read, concurrency guard, and
PostgreSQL reachability proof are one indivisible security repair. Splitting
the writer or proof from the read would ship either an empty feature or an
unverified privacy boundary.

### Problem-derived contract

- Root cause: `sent_emails` carries only recipient-address metadata and has no
  tenant key; its repository is read but never written by production sending
  code. Customer context therefore performs an address-only global read and
  later discards it when scoped. Returning that page would leak same-recipient
  foreign or unclassified rows, and filtering after `LIMIT` would let those
  rows starve the correct tenant. Address-only lookup also loses an
  acknowledgement sent to a corrected/submitted address when the intake
  resolves the contact by phone and intentionally leaves its stored identity
  unchanged. The aggregate validates the contact object fetched before child
  reads rather than re-fetching current ownership after the final await.
- Correct fix must touch/change: add a replay-safe nullable tenant key and
  exact pre-pagination repository predicates; persist the successful EOM
  website acknowledgement as a secondary best-effort write; pass effective
  tenant scope and the writer's contact metadata into the sent-history read;
  skip still-unaddressable inbox/B2B providers under scope; return scoped sent
  history with an accurate source-level omission signal; re-fetch current
  contact ownership after all child reads; avoid logging recipient/content
  PII from the newly activated writer; preserve the legacy global-history
  response shape; and prove the real FastAPI intake plus CRM MCP aggregate
  against PostgreSQL with corrected-address, foreign/NULL page-starvation, and
  ownership-race probes.
- Must not change: do not infer or backfill legacy row ownership; do not include
  NULL email rows under any scope; do not tenant-stamp global
  `b2b_churn_signals`; do not open the global mailbox under scope; do not alter
  acknowledgement copy or the public lead response; do not add email events to
  the authenticated HTTP timeline; do not centralize unrelated Atlas email
  senders; and preserve unscoped repository/customer-context behavior.

## Scope (this PR)

Ownership lane: eom-crm/email-tenancy
Slice phase: Production hardening

1. Make `sent_emails` exactly tenant-addressable while retaining legacy
   unscoped compatibility and leaving existing NULL rows unclassified.
2. Record successful EOM website acknowledgements and expose only those
   tenant-matched rows through scoped CRM customer context.
3. Prove the complete intake-to-context path and its fail-closed boundaries
   against isolated PostgreSQL.

### Review Contract

- Acceptance criteria:
  1. Migration replay is safe; existing rows remain NULL; non-NULL tenant IDs
     cannot be blank; no ownership is defaulted or inferred.
  2. Repository scoped reads use exact tenant equality before
     `ORDER BY`/`LIMIT`/`OFFSET`; NULL and foreign rows never appear or starve
     the requested page. Within that tenant, customer context can match either
     the current address or writer-recorded contact ID so corrected submitted
     addresses remain linked. Omitting scope preserves existing global reads.
  3. A freshly sent EOM website acknowledgement creates one
     `effingham_maids` history row only after transport success. Send refusal,
     duplicate/honeypot/cap/disabled paths create none.
  4. History persistence failure is logged but cannot fail an already-captured
     lead, flip `email_sent`, or trigger a duplicate send.
  5. Scoped CRM context returns the EOM sent row, excludes populated foreign
     and NULL rows, never queries inbox/B2B providers, retains the aggregate
     omission flags, and identifies `inbox_emails` in
     `email_sources_omitted_under_scope`.
  6. A contact reassigned foreign or explicitly to NULL after the sent-email
     query returns not-found with no email serialization; deployment-default
     tenant-plus-NULL contact compatibility remains unchanged.
  7. The activated writer does not log recipient/content PII. The lead HTTP
     response, acknowledgement copy, HTTP timeline, unscoped context, and
     global email-history tool response shape retain their existing behavior.
- Reachability proof: FastAPI `POST /api/v1/leads/intake` executes through the
  real route and database CRM/repository against isolated PostgreSQL; the test
  then calls the real CRM MCP `get_customer_context` and observes exactly one
  EOM-stamped acknowledgement despite newer foreign and NULL rows for the same
  recipient.
- Affected surfaces: EOM lead intake, sent-email schema/model/repository,
  customer-context aggregation, CRM MCP serialization, EOM PostgreSQL CI, and
  focused boundary tests.
- Risk areas: guessing legacy ownership, address collision across tenants,
  filtering after pagination, secondary-write failure after delivery,
  duplicate sends, provider access under a forbidden scope, stale-contact
  TOCTOU, default-versus-explicit NULL semantics, and additive response
  compatibility.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12,
  R13, R14.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/api/leads.py`
- `atlas_brain/mcp/crm_server.py`
- `atlas_brain/services/customer_context.py`
- `atlas_brain/storage/migrations/349_sent_emails_business_context.sql`
- `atlas_brain/storage/models.py`
- `atlas_brain/storage/repositories/email.py`
- `plans/INDEX.md`
- `plans/PR-EOM-Sent-Email-Tenant-Scope.md`
- `plans/archive/PR-EOM-API-Contacts-Auth.md`
- `tests/test_crm_read_scoping.py`
- `tests/test_eom_sent_email_tenant_scope.py`
- `tests/test_leads_intake.py`
- `tests/test_migrations_runner.py`

## Mechanism

Migration 349 adds a nullable, nonblank-when-present `business_context_id` plus
a partial recency index. NULL is retained solely as unclassified legacy state.
The email repository appends optional tenant/contact arguments to its
write/read methods, normalizes provided tenant IDs, stores the key, projects
it into `SentEmail`, and adds exact tenant equality before pagination.
Customer-context queries combine current-address and metadata contact-ID
identity inside that exact tenant predicate; unscoped callers retain the
address-only behavior. `SentEmail.to_dict()` intentionally keeps its legacy
shape. Customer-context serialization delegates to that projection and removes
the tenant key from fallback mappings so neither scoped nor unscoped public
context exposes internal ownership metadata.

The EOM intake route injects the history repository. After the provider reports
a successful acknowledgement, a separate guarded secondary write records the
rendered content, tenant, template, provider message ID, and source/contact
metadata. Transport and history failure remain separate states.

Customer context supplies its effective scope to the sent-email query. Scoped
gathers replace inbox and B2B coroutines with empty results so the global
providers are not opened. CRM serialization returns `ctx.sent_emails`, keeps
the existing aggregate omission flags, and adds
`email_sources_omitted_under_scope=["inbox_emails"]`. Immediately before
synchronous serialization, the MCP tool re-fetches the contact and applies the
existing explicit/default visibility predicate to that current row.

## Intentional

- Only the EOM website acknowledgement is enrolled as a writer. This is the
  smallest real operator-visible path; broad sender centralization is deferred.
- Existing rows remain NULL and scoped reads use exact equality only. Recipient
  address is evidence for lookup, not ownership.
- `business_context_id` remains optional at the repository boundary for
  compatibility, but the covered EOM writer always supplies it; blank supplied
  values fail before SQL.
- The existing `emails_omitted_under_scope` flag remains true while inbox is
  withheld. The additive source list prevents that aggregate flag from
  implying sent history is still wholly absent.
- No `contact_id` column is added. This slice preserves address matching inside
  an exact tenant boundary and uses the known contact ID stored in metadata as
  an alternate match for corrected/submitted addresses.
- The database migration precedes application rollout. Its additive nullable
  shape keeps old code and rollback safe.

## Deferred

- #2177 / #2171B: bind CRM business contexts to mailbox/provider credentials
  and refuse unmapped contexts before opening IMAP/Gmail.
- #2178 / #2171C: authorize B2B CRM enrichment through a
  CRM-context-to-SaaS-account mapping and the existing `tracked_vendors`
  model; do not alter global signal ownership.
- #2179 / #2171D: centralize history persistence for the remaining Atlas
  outbound email producers and reconcile the global `list_sent_history`
  claim.
- The authenticated HTTP contact timeline remains email-free until its own
  accepted product/output slice.

Parked hardening: none.

## Verification

- `ATLAS_MIGRATION_TEST_DATABASE_URL=postgresql://atlas_test:atlas_test@localhost:55433/atlas_migration_tests python -m pytest tests/test_crm_read_scoping.py tests/test_eom_complaints_integration.py tests/test_eom_contacts_api_tenant_scope.py tests/test_eom_lead_pipeline_integration.py tests/test_eom_recurring_appointments_integration.py tests/test_eom_sent_email_tenant_scope.py tests/test_leads_intake.py tests/test_migrations_runner.py -q`
  — 120 passed.
- `python scripts/maturity_sweep.py atlas_brain/api --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_api.json --min-score 8`
  — ratchet passed.
- `python scripts/maturity_sweep.py atlas_brain/mcp --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_mcp.json --min-score 8 --sensitive-glob '**/*'`
  — ratchet passed.
- `python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob 'atlas_brain/storage/**'`
  — ratchet passed.
- Focused Python compilation — passed:

      python -m py_compile atlas_brain/api/leads.py atlas_brain/mcp/crm_server.py atlas_brain/services/customer_context.py atlas_brain/storage/models.py atlas_brain/storage/repositories/email.py tests/test_eom_sent_email_tenant_scope.py
- `git diff --check` — passed.
- Managed local review — passed:

      ATLAS_SESSION_STATE_FILE=SESSION_STATE.codex-eom-email-tenancy.local.md ATLAS_CURRENT_PR_BODY_FILE=tmp/pr-eom-sent-email-tenant-scope-body.md bash scripts/local_pr_review.sh
- Independent judgment review round 1 found corrected/missing stored-address
  linkage, recipient-log PII, and global serializer compatibility gaps. All
  three were fixed and covered behaviorally; final review is against the
  publication head.
- GitHub Codex review round 2 found customer-context serialization bypassed the
  legacy `SentEmail.to_dict()` projection and leaked the new tenant member in
  unscoped MCP responses. The serializer now uses the legacy projection,
  strips the internal key from fallbacks, and the PostgreSQL MCP proof covers
  the unscoped response.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 11 |
| `atlas_brain/api/leads.py` | 35 |
| `atlas_brain/mcp/crm_server.py` | 27 |
| `atlas_brain/services/customer_context.py` | 65 |
| `atlas_brain/storage/migrations/349_sent_emails_business_context.sql` | 26 |
| `atlas_brain/storage/models.py` | 1 |
| `atlas_brain/storage/repositories/email.py` | 109 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Sent-Email-Tenant-Scope.md` | 225 |
| `plans/archive/PR-EOM-API-Contacts-Auth.md` | 0 |
| `tests/test_crm_read_scoping.py` | 57 |
| `tests/test_eom_sent_email_tenant_scope.py` | 541 |
| `tests/test_leads_intake.py` | 125 |
| `tests/test_migrations_runner.py` | 22 |
| **Total** | **1247** |
