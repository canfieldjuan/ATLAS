# PR-EOM-Public-Onboarding-Tracker-Context-Recovery

## Why this slice exists

The merged public-onboarding authority intentionally gives the Tracker a
browser-safe session projection: `/public-onboarding/session` returns immutable
prefill fields but strips Atlas `contact_id` and `handoff_id`. That is the right
contract for a browser bridge, but the next Tracker slice also needs durable
Atlas identifiers after it creates its local Customer/Site. If the local save
succeeds and bearer redemption fails or its response is lost, the generic
office handoff remains fenced by the still-issued token. Staff would otherwise
need the raw bearer or a non-atomic revoke-then-handoff repair.

This Atlas dependency adds the service-only context and recovery primitives the
Tracker needs before it can safely build the no-login flow. It closes a real
handoff durability risk, not a cosmetic or browser-shape concern. The Tracker
and Website remain separate, ordered slices; production configuration stays
disabled.

Diff-budget override: This intentionally exceeds the normal 400-line target
because the private context, one-transaction recovery operation, FastAPI
admission tests, and real-Postgres state/rollback/concurrency proofs are one
authorization-and-durability boundary. Splitting them would either ship a
recovery path without its atomicity proof or defer the only safe repair for a
persisted Tracker Customer/Site after raw-bearer finalization becomes ambiguous.

### Problem-derived contract

- Root cause: one public-token state is serving two deliberately distinct
  audiences without a private recovery bridge. The current safe browser
  projection omits token-bound Atlas IDs, and the only finalizer for an issued
  token requires its raw bearer. After Tracker persists a Customer/Site, a
  permanent or ambiguous Atlas-finalize outcome cannot be resumed from durable
  IDs alone because the active-token fence rejects ordinary office handoff.
  A two-request repair would add an avoidable state gap between revocation and
  finalization.
- Correct fix must touch/change:
  1. Add a distinct, service-authenticated tracker-context route and provider
     projection. It must authenticate the raw bearer through the existing
     canonical parser and return only to the Tracker the token, draft, and
     contact IDs plus the existing immutable prefill/completion fields.
  2. Preserve the existing browser-bridge session response exactly while
     sharing the same durable token-state decision underneath it; the new
     private projection must not make IDs visible through `/session`.
  3. Add one actor-audited recovery operation that accepts only stored token,
     contact, and Tracker Customer/Site IDs. In one PostgreSQL transaction it
     must take the established handoff and draft locks, revoke an `issued`
     token only immediately before reusing the existing handoff finalizer, and
     roll both changes back if finalization fails. A redeemed matching handoff
     and a previously recovered matching handoff must replay idempotently;
     mismatched Tracker IDs must reject.
  4. Add the private recovery route behind the existing service bearer and
     actor evidence, available even when public-link authority is disabled so
     a previously issued token cannot strand local work. The future Tracker
     endpoint will apply its existing configured Juan-only admin guard before
     sending this actor evidence; Atlas records the actor in the canonical
     handoff lifecycle rather than attempting to invent a second admin-login
     system.
  5. Prove the added routes through real FastAPI entrypoints and prove the
     issued/revoked/redeemed recovery transitions through the disposable
     PostgreSQL provider, including idempotence, mismatches, and rollback of
     the token status when finalization cannot proceed.
- Must not change:
  1. Do not change the request or response contract of
     `POST /eom-funnel/public-onboarding/session`; it must remain ID-free.
     Do not change the existing bearer grammar, HMAC rotation, public
     `finalize` route, issuance-pause semantics, or service bearer boundary.
  2. Do not store or return a raw bearer outside the existing in-memory
     parser/formatter path, expose a service credential to a browser, or add
     an unauthenticated Atlas endpoint.
  3. Do not add a migration, new config, a Tracker Customer/Site writer, a
     Tracker or Website route/UI, a unique `customers.atlas_contact_id` rule,
     or a second handoff table/finalizer. Do not change the Tracker's existing
     configured Juan approver policy in this Atlas dependency.
  4. Do not alter lead-stage vocabulary, booking/calendar behavior, email
     transport, payroll, QR/GPS, scheduling, payments, production values, or
     production data.

## Scope (this PR)

Ownership lane: eom-public-onboarding-tracker-bridge
Slice phase: Vertical slice
Max files: 7

1. Add the private Atlas context lookup required for the Tracker to retain
   token/contact/draft identity without widening the browser-safe session
   response.
2. Add the atomically recoverable handoff path for Tracker-local records when
   token redemption cannot be completed with the raw bearer.
3. Test the route authorization/projection boundary and real Postgres state
   transitions, then archive the merged token-authority plan and refresh the
   plan index as required housekeeping.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /eom-funnel/public-onboarding/session` remains service-only and
    returns no `contact_id`, `draft_id`, `token_id`, or `handoff_id`; settled
    by `tests/test_eom_public_onboarding.py::test_public_session_requires_service_auth_and_never_requires_an_actor`.
  - [ ] `POST /eom-funnel/public-onboarding/tracker-context` accepts a
    canonical bearer only after the existing service and public-authority
    dependencies pass, then returns the token-bound `token_id`, `draft_id`,
    and `contact_id` to the service caller; missing service auth yields 401,
    disabled authority yields 503, and malformed bearer causes no CRM lookup.
    Settled by focused ASGI route tests in
    `tests/test_eom_public_onboarding.py`.
  - [ ] `POST /eom-funnel/public-onboarding/recover` requires the service
    bearer plus actor headers, needs no raw bearer or enabled public-link
    authority, and exposes only completion evidence. Its actor is passed into
    the existing handoff lifecycle. Settled by focused ASGI route tests and
    the Tracker's later Juan-only route contract.
  - [ ] For every admitted recovery interleaving, the recovery transaction
    acquires the same sorted contact/Tracker-ID locks plus the token draft lock
    before its decisive row read; an issued token transitions to `revoked` and
    the one handoff commits together, or both roll back. A matching redeemed or
    prior-recovery record replays idempotently; different Tracker IDs reject.
    Settled by `tests/test_eom_lead_conversion_integration.py` against
    disposable Postgres.
  - [ ] No schema/config/production mutation is introduced and the merged
    token-authority plan is moved only to `plans/archive/`; settled by the cold
    diff reconstruction and `git diff --check`.
- Reachability proof: the FastAPI tests invoke both new routes through the
  registered router and observe response status/body; the Postgres test invokes
  `DatabaseCRMProvider.recover_eom_public_onboarding` and observes token,
  contact, lifecycle, and handoff rows.
- Affected surfaces: `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/services/crm_provider.py`, service authentication, the existing
  public-onboarding token state, EOM customer-handoff transaction, and their
  focused FastAPI/Postgres tests.
- Risk areas: service authorization, browser/private data separation,
  backward-compatible response shapes, idempotency, transaction rollback,
  recovery after authority disable, and duplicate/conflicting handoffs.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R14.

### Boundary-change enumeration

- Boundary path/seam: `POST /public-onboarding/tracker-context` is a new
  service-only projection seam. Existing `/session` remains the browser-bridge
  seam and is preserved. `POST /public-onboarding/recover` is a new
  service-and-actor recovery admission seam over stored IDs.
- Replaced-path behaviors: none. The context route is additive; the existing
  session and finalize routes retain their parser, authorization, status
  masking, and response shapes. Recovery replaces no public path; it gives the
  Tracker a safe alternative to a raw-bearer retry or manual two-request
  repair.
- Guard-relevant fields: raw `token` reaches only the canonical HMAC parser;
  `token_id` and `contact_id` are UUIDs; Tracker Customer/Site IDs are strict
  positive signed-bigint values; service authorization and actor headers must
  pass before a provider call.
- Caller x input shape:
  - Tracker service x valid bearer: context returns private token/draft/contact
    IDs and immutable prefill or completed evidence.
  - Browser/no bearer x any bearer: service dependency rejects before parser or
    provider.
  - Tracker service x malformed bearer: parser returns the established
    unavailable response before provider access.
  - Tracker recovery route x issued matching IDs: atomic revoke and office
    finalization under the recovery key.
  - Tracker recovery route x revoked/no handoff matching IDs: finalizes once
    under the recovery key; replay is idempotent.
  - Tracker recovery route x redeemed/existing matching IDs: returns completed
    idempotently; different IDs reject.

### Deployed-config probing

- Deployed/default config values: public onboarding authority is default
  disabled; the EOM service API itself is separately default disabled. No
  configuration value changes in this PR.
- Explicit value probe: enabled service + valid public authority authenticates
  the tracker-context bearer and reaches its provider projection.
- Absent value probe: no service bearer returns 401 before context/recovery
  provider calls; disabled public authority returns 503 before context
  bearer/provider use.
- Default-session/default-context probe: recovery uses only the configured
  service boundary plus actor evidence and remains reachable when public
  authority is disabled, matching the existing private revoke-link recovery
  policy.
- Side-effect ordering: recovery validates the service/actor/payload boundary,
  obtains the deterministic sorted lock set, and then performs its decisive
  token/contact read. The issued-to-revoked update and existing finalizer share
  one transaction, so a rejected finalizer cannot leave a new terminal token
  state behind.

### Files touched

- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/crm_provider.py`
- `plans/INDEX.md`
- `plans/PR-EOM-Public-Onboarding-Tracker-Context-Recovery.md`
- `plans/archive/PR-EOM-Public-Onboarding-Token-Authority.md`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_public_onboarding.py`

## Mechanism

The context route takes the same raw bearer as the existing session route but
is separately named and service-authenticated. Its provider projection is
derived from the same one durable token/draft/contact read, so it can return
the immutable prefill plus opaque `token_id`, `draft_id`, and `contact_id` to
the Tracker without making those IDs part of the browser-safe `/session`
contract.

Recovery receives only opaque values that the Tracker stored after its local
reservation: token ID, Atlas contact ID, and Tracker Customer/Site IDs. It
derives a deterministic recovery approval key from the token ID, takes the
existing sorted customer-handoff locks and the public-token draft lock, then
re-reads and locks the token/draft/contact rows. For an issued token it records
revocation and calls `finalize_eom_customer_handoff` on that same connection.
The finalizer sees no active fence and uses its existing immutable handoff,
lifecycle, and uniqueness rules. A failure rolls back the revocation as well as
the finalizer. A prior recovery reuses its deterministic key; redeemed or other
matching completed evidence returns idempotently, while different local IDs do
not get silently attached to the Atlas contact.

The recovery route remains intentionally usable after the public authority is
disabled: it never parses a bearer, mints a link, or reads its HMAC config. It
is still restricted to the Tracker service bearer and actor headers. The next
Tracker slice supplies the authenticated, configured-Juan admin gate that
chooses whether it may invoke this Atlas operation.

## Intentional

- `/public-onboarding/session` remains ID-free even though it is currently
  service-authenticated. Keeping the private context separate prevents a later
  browser bridge from accidentally inheriting a broader response.
- Recovery marks an issued token revoked and uses the normal `office`
  completion channel. It is a staff intervention after raw-bearer completion
  could not be trusted, not a second public redemption channel.
- Atlas does not add a Juan identity configuration. Tracker already owns the
  authenticated employee session and configured stable approver ID; Atlas
  receives only its service-authenticated actor evidence and records it in the
  canonical lifecycle event.
- No migration is needed: token IDs, draft IDs, contact IDs, status, revocation
  state, and the immutable handoff transaction already exist. A new table would
  create a parallel recovery source of truth.

## Deferred

- The immediately following Tracker slice will add the public session/complete
  proxy, durable local reservation, conflict states, admin list/revoke/recover
  controls, and the configured Juan-only gate that invokes this Atlas recovery
  route. It cannot start until this Atlas PR merges or Juan explicitly releases
  that ordering.
- The Website no-login EN/ES page, fragment cleanup, form fields, and customer
  success/retry rendering remain a later slice.
- Expiry/reissue policy, reminders, multi-site onboarding, metrics, and
  customer-visible copy are outside this dependency.

Parking predicate: park browser/UI design, extra onboarding fields,
observability polish, and recovery scenarios that do not create a duplicate
handoff or strand an issued token. Keep raw-bearer exposure, missing service or
actor authorization, changed `/session` visibility, or non-atomic
issued-token recovery in scope because each breaks this dependency's safety
contract.

Parked hardening: none.

## Verification

- Passed: `python -m pytest -q tests/test_eom_public_onboarding.py` — 39 passed.
- Passed: the exact EOM lead-pipeline workflow test list against a disposable
  PostgreSQL 16 instance — 1114 passed, 5 skipped, 1 environment warning in
  93.13s. This includes
  `tests/test_eom_lead_conversion_integration.py` and the new real-Postgres
  recovery-state, rollback, and cross-finalizer-concurrency proofs.
- Passed: `python -m compileall -q atlas_brain/eom_api/funnel.py atlas_brain/services/crm_provider.py tests/test_eom_public_onboarding.py tests/test_eom_lead_conversion_integration.py`.
- Passed: Ruff against the two changed Atlas modules and two focused test
  modules.
- Passed: `bash scripts/check_ascii_python.sh`.
- Passed: `python scripts/check_guard_class_closure.py --base origin/main --strict`
  (no guard-shaped change without a property test).
- Passed: `git diff --check` and `git diff --cached --check`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/funnel.py` | 92 |
| `atlas_brain/services/crm_provider.py` | 259 |
| `plans/INDEX.md` | 3 |
| `plans/PR-EOM-Public-Onboarding-Tracker-Context-Recovery.md` | 285 |
| `plans/archive/PR-EOM-Public-Onboarding-Token-Authority.md` | 0 |
| `tests/test_eom_lead_conversion_integration.py` | 292 |
| `tests/test_eom_public_onboarding.py` | 240 |
| **Total** | **1171** |
