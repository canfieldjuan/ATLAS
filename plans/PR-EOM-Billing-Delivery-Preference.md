# PR-EOM-Billing-Delivery-Preference

## Why this slice exists

The Billing & Payments coordinating issue [#2362](https://github.com/canfieldjuan/ATLAS/issues/2362)
requires an explicit billing-delivery preference before any approved candidate
can create an invoice, PDF, Gmail draft, or Square queue item.  The deployed
commercial preview deliberately returns `deliveryMethod: null` and always
blocks an otherwise-commercial candidate with
`missing_billing_delivery_preference`
(`atlas_brain/services/commercial_billing_candidates.py:669-737,944-961`).
That safe placeholder makes the durable review UI useful, but it also proves
that the next dependency is a canonical source for the preference—not an
approval writer that guesses Gmail from customer type or an existing email.

Deferred finding H-13 records the other prerequisite: services expose calendar
evidence and event locations but no canonical service/site model.  This slice
does not solve that separate evidence problem; it leaves approval unavailable
until it is resolved.

### Diff budget

This exceeds the 400-LOC soft target because the smallest safe vertical
behavior is one chain: additive profile schema -> canonical/actor-audited
upsert -> authenticated provider route -> pure candidate fingerprint/blocker
projection.  The larger share is the real-PostgreSQL admission/retry/concurrency
proof and real-ASGI credential proof.  Splitting the schema/provider from that
proof or from the candidate reader would either publish an unverified policy
writer or recreate a stored policy that no review/staleness boundary observes.

### Problem-derived contract

- Root cause: a candidate needs a durable, explicit delivery decision in its
  source evidence.  Today there is no canonical preference to read, so any
  later invoice writer would either infer one from customer type/email or lose
  the choice from its stale-source fingerprint.
- Correct fix must touch/change: add an additive canonical-customer profile
  table, actor-audited read/upsert provider methods, authenticated full-provider
  GET/PUT routes, and pure-candidate integration so the preference appears in
  `deliveryMethod`, the blocker contract, and the existing source fingerprint.
- Must not change: invoices, invoice balances, payments, receipts, deposits,
  billing-run state/snapshots, service markers, PDFs, Gmail, email transport,
  sent reconciliation, Square API calls, MCP, the legacy monthly scheduler,
  tracker storage, Website behavior, or a service/site model.  No default,
  backfill, or customer-type inference is permitted.

## Scope (this PR)

Ownership lane: eom/billing-delivery-preference
Slice phase: Vertical slice
Max files: 9

1. Add an explicit canonical EOM billing-delivery preference with exactly three
   persisted methods: `gmail_pdf`, `manual_square`, and
   `no_invoice_residential_receipt`.  The authenticated operator supplies the
   method; each row retains created/updated actor and time.  A same-method retry
   is a no-write replay and a changed method updates the actor/time atomically.
2. Add authenticated full-provider preference GET/PUT routes and make the
   existing pure commercial candidate reader consume the preference.  Existing
   customers with no preference remain visibly blocked; a Square preference
   does not require a Gmail recipient; a no-invoice/receipt preference is
   visibly ineligible for a commercial invoice candidate.  Focused ASGI,
   real-PostgreSQL, source-fingerprint, retry, malformed-input, and
   no-delivery-writer tests settle those claims locally.

### Review Contract

- Acceptance criteria:
  - `DatabaseCRMProvider` admits a preference only for the active canonical EOM
    customer named by its tenant-scoped SQL predicate.  The real-PostgreSQL
    test proves an eligible row records the selected method and both actors,
    while inactive, foreign-tenant, lead, and absent contacts write zero rows.
  - `delivery_method` is a closed public contract.  FastAPI/Pydantic rejects an
    unrecognized string before provider invocation; migration 371 rejects
    impossible stored values; the route test covers valid values, malformed
    values, missing actor, wrong token, and no-token calls.
  - Equal contact/method retries return `changed: false` without changing
    `updated_at` or `updated_by`; a different method returns `changed: true`
    and records its actor. The real-PostgreSQL test forces same-method and
    different-method contenders through the canonical row lock, then cancels a
    blocked profile update and proves the committed state still contains only
    the preceding complete change.
  - `CommercialBillingCandidateService` returns the exact stored method as
    `deliveryMethod`; an absent preference keeps the existing missing-preference
    blocker, `manual_square` does not add `missing_billing_email`, and
    `no_invoice_residential_receipt` adds the closed
    `no_invoice_delivery_preference` blocker for a commercial candidate.
    Candidate tests prove every case, one bounded preference query for all
    candidate customers, and that a preference change changes the SHA-256
    source fingerprint consumed by the existing run reconciliation. The
    current candidate contract is explicitly version 2; durable-run parsing
    continues to read persisted version-1 snapshots.
  - Static source assertions prove this slice imports no invoice/PDF/Gmail/
    email/notification/monthly-scheduler writer and its tests exercise only a
    disposable local PostgreSQL schema or ASGI transport; it cannot send real
    customer mail or create a live financial record.
  - The migration is additive and has no default/backfill/destructive DDL.
    Older code ignores the table and newer code treats its absence as a
    controlled 503; rollback is application rollback first while retaining the
    audited profile evidence.
- Reachability proof: the active full application mounts
  `atlas_brain.api.invoicing.receivables` through
  `atlas_brain.main.app -> api_router -> invoicing router` under `/api/v1`.
  The smoke test uses that real app object, observes an authenticated GET at
  the prefixed path, and proves the unprefixed path is 404. A separate
  router-ASGI test exercises PUT with a fake canonical provider. Production
  verification is unauthenticated-only (401); it will not create a real
  preference.
- Affected surfaces: migration 371, `DatabaseCRMProvider`, full receivables
  router, the pure candidate source protocol/projection, existing invoicing
  workflow, focused provider/route/candidate tests, and this plan.
- Risk areas: tenant/customer admission, actor provenance, retry/write
  idempotency, Postgres rollout, stale evidence, closed-vocabulary drift,
  recipient leakage/use for Square, and accidental financial or delivery side
  effects.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R13,
  and R14. R7 is N/A: no new async worker; R9 is N/A: no browser change.

### Boundary-change enumeration

**Seam 1 — preference write admission.** Authenticated
`PUT /receivables/commercial-billing-delivery-preferences/{contact_id}` ->
`DatabaseCRMProvider.set_eom_billing_delivery_preference` -> tenant-scoped,
active-customer row lock/upsert.

- Replaced-path behaviors: none. This is additive; current preview stays
  blocked until a preference is explicitly recorded.
- Guard-relevant fields: bearer token, `X-EOM-Actor`, UUID contact ID,
  `delivery_method`, `contacts.business_context_id`, `status`, and
  `contact_type`.
- Caller x input shape: receivables credential x {each known method, malformed
  method, absent body, inactive customer, lead, foreign tenant, unknown UUID};
  no/wrong credential x every shape; present/blank actor; same/different
  preference retry.  Invalid request/auth shapes fail before provider write.

**Seam 2 — pure candidate source resolution.**
`CommercialBillingCandidateService._build_preview` reads one tenant-scoped
delivery-preference map for all linked candidate contacts, then
`_load_customer_evidence` reads canonical customer and conditional recipient
evidence before constructing each candidate.

- Replaced-path behaviors: the existing hard-coded null/missing-preference
  projection.  No writer is introduced.
- Guard-relevant fields: customer type, exact closed delivery method, recipient
  eligibility, and candidate source fingerprint.
- Caller x input shape: canonical commercial customer x {no preference,
  gmail+recipient, gmail+no-email, manual Square+no-email,
  no-invoice/receipt}; unavailable/malformed provider evidence x every class.
  Unknown stored delivery values fail closed as source evidence invalid rather
  than claiming approval eligibility.

### Preference-write execution model and invariant

The admitted execution surface is one call to
`DatabaseCRMProvider.set_eom_billing_delivery_preference` for one contact.
Input vocabulary and actor validation happen before the transaction. Every
admitted provider write then uses one PostgreSQL transaction without an
application-level isolation override (the PostgreSQL default is `READ
COMMITTED`): it first selects the active, tenant-scoped canonical `contacts`
row `FOR UPDATE`, reads the one-to-one profile, conditionally inserts or
updates it, and commits or rolls back as one unit. PostgreSQL holds that
contact-row lock until commit/rollback; the profile's primary key supplies at
most one row per contact. A deployment-selected stronger isolation may abort a
contender, but cannot expose a partial profile. All application writes in this
slice use this method; direct administrative SQL outside this provider is not
an admitted caller.

For every schedule among those admitted calls, each successful different-method
write linearizes when it acquires the canonical row lock. The committed profile
is therefore either absent or exactly the method/actor/timestamp of the final
different-method call in that lock order; its `created_*` fields remain from
the insert, and an equal-method replay makes no state change. A failed
admission creates no profile. Calls for different contacts are intentionally
independent rather than globally ordered.

Cancellation, database error, or process failure before commit exits the
transaction and PostgreSQL rolls back its partial work before releasing locks;
there is no invoice, payment, Gmail, or other external side effect in this
transaction. If a caller loses its result after the server may have committed,
the result is intentionally ambiguous to that caller, but persistent state is
still either the prior complete profile or the complete committed profile—never
a partial row. Retrying the same observed method reads that current state and
returns the no-write replay result. The real-PostgreSQL concurrency test forces
same-method replay, different-method serialization, cancellation while the
profile `UPDATE` is blocked, and the subsequent retry; it asserts this
invariant before and after lock release.

### Deployed-config probing

- Deployed/default config values: no new setting, credential, flag, fallback,
  or browser-visible secret. The route reuses deployed receivables token/auth
  and primary CRM pool.
- Explicit value probe: ASGI valid `gmail_pdf`, `manual_square`, and
  `no_invoice_residential_receipt` requests prove the closed values arrive at
  the provider with the authenticated actor.
- Absent value probe: no row returns `deliveryMethod: null`; the candidate
  preserves `missing_billing_delivery_preference`. Missing token/actor and a
  missing preference table return 401/422/503 rather than guessing a method.
- Default-session/default-context probe: existing customers have no new row or
  default during migration. No customer type, email, service, or calendar value
  becomes an implicit delivery preference.
- Side-effect ordering: PUT starts one database transaction, locks/validates
  the canonical contact, then conditionally writes the preference. Candidate
  read happens separately and has no writer import or call. No financial or
  mail side effect precedes, joins, or follows either operation.

### Closure declaration

- Delivery method is **CLOSED** and code-owned by
  `EOMBillingDeliveryMethod`, with matching migration `CHECK` enforcement:
  `gmail_pdf`, `manual_square`, and `no_invoice_residential_receipt`. External
  unmatched values are rejected at the request boundary; an impossible stored
  value produces the safe `source_evidence_invalid` candidate blocker.
- Candidate blocker code is **CLOSED** and code-owned by
  `BILLING_CANDIDATE_BLOCKER_CODES`. This slice adds
  `no_invoice_delivery_preference`; it means a commercial candidate cannot
  proceed as an invoice candidate under a receipt-only policy. Unknown source
  evidence is not normalized into a new blocker token.
- Candidate JSON is **versioned**: this new delivery method/blocker semantics
  increments `_CANDIDATE_CONTRACT_VERSION` from 1 to 2. New previews and their
  durable snapshot fingerprints carry 2; the run reader intentionally accepts
  already-persisted positive versions, including version 1.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/api/invoicing/receivables.py`
- `atlas_brain/services/commercial_billing_candidates.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/storage/migrations/371_eom_billing_delivery_preferences.sql`
- `plans/PR-EOM-Billing-Delivery-Preference.md`
- `tests/test_commercial_billing_candidates.py`
- `tests/test_commercial_billing_runs.py`
- `tests/test_eom_billing_recipients.py`

## Mechanism

Migration 371 introduces one profile row per canonical contact.  The profile is
not a ledger and does not duplicate money or invoice state.  Its foreign key is
restrictive so the policy evidence cannot be silently erased through a contact
delete.

The provider locks the admitted active customer row in the primary canonical
CRM database and then conditionally upserts its preference.  SQL returns the
existing row unchanged for an equal-method replay.  A changed method updates
only the profile's delivery method, updater, and timestamp.  Both read and
write project the current preference plus actor/timestamp audit fields.

The candidate reader obtains one tenant-scoped batch of preferences for all
linked candidate contacts, then combines each map entry with its existing
customer and recipient evidence. It puts the exact method into version-2
canonical candidate JSON before hashing. Gmail still needs a recipient; manual
Square does not. Receipt-only is represented explicitly but remains blocked for
a commercial invoice candidate. Thus durable-run reconciliation sees a changed
preference as a changed source fingerprint, while no invoice/PDF/Gmail/Square
writer exists in this slice.

## Intentional

- No migration default or customer-type/email inference: historical customer
  records remain blocked until an operator selects a policy.
- A same-method retry does not overwrite `updated_by` merely because a later
  operator repeats it. This preserves the actor who made the last material
  policy change and avoids turning a transport retry into an audit mutation.
- `manual_square` is a routing decision only. It neither invokes Square nor
  marks an invoice sent; those audited actions belong after explicit approval.
- `no_invoice_residential_receipt` is retained as an explicit universal policy
  value but blocks a commercial invoice candidate, instead of silently turning
  a commercial candidate into a receipt workflow.

## Deferred

- #2362: tracker proxy and Website UI for preference editing; selected-candidate
  approval; idempotent invoice/PDF/Gmail-draft creation and recovery; sent-mail
  reconciliation; manual Square queue/reference; operating instructions.
- #2363 H-13: canonical service/site identity and evidence must be complete
  before an approval writer exists.
- #2363 H-07: payment-method reference hardening remains unrelated.

Parking predicate: broader delivery transport, generic profile/event modeling,
and legacy scheduler behavior are parked unless they violate this explicit
policy write/read and candidate-fingerprint contract. Parked hardening:
historical profile change-event journaling, a generic cross-product
delivery-preference model, Square API integration, and legacy scheduler changes
remain outside this canonical EOM profile slice.

## Verification

- Local workflow-equivalent evidence (2026-08-14): the exact three commands in
  `.github/workflows/atlas_invoicing_checks.yml` ran with CPython 3.11.15 against the isolated
  local PostgreSQL 16 container at `127.0.0.1:55441` (disposable schemas only):
  approval-blocker command **2 passed**; ledger/repository command **253
  passed**; MCP/OAuth command **43 passed**. The new real-PostgreSQL test also
  proved actor-audited set/read, no-write retry, check-constraint rejection,
  forced same- and different-method serialization, cancellation rollback, and
  post-cancellation retry (**81 focused tests passed**).
- CPython 3.11 `compileall`, focused `ruff check`, candidate/route/migration/
  no-writer/workflow-enrollment tests, `sync_pr_plan.py --check`, and `git diff
  --check` passed. The candidate test covers no preference, Gmail/no-email,
  manual Square/no-recipient, receipt-only, unknown stored value, one batched
  preference query, source change, error/recovery, and no side effects. The
  real `main.app` smoke test proves the `/api/v1` full-router mount; version-2
  new previews and version-1 durable snapshots are both covered.
- Before push: run the repository `push_pr.sh`/`open_pr.sh` local review path.
  Hosted Actions are diagnostic only under Juan's explicit local-check
  direction. Before merge, inspect GraphQL review threads on the published
  head; fix confirmed actionable threads, locally re-verify/publish, then
  resolve them.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/api/invoicing/receivables.py` | 78 |
| `atlas_brain/services/commercial_billing_candidates.py` | 147 |
| `atlas_brain/services/crm_provider.py` | 199 |
| `atlas_brain/storage/migrations/371_eom_billing_delivery_preferences.sql` | 42 |
| `plans/PR-EOM-Billing-Delivery-Preference.md` | 323 |
| `tests/test_commercial_billing_candidates.py` | 190 |
| `tests/test_commercial_billing_runs.py` | 16 |
| `tests/test_eom_billing_recipients.py` | 634 |
| **Total** | **1631** |
