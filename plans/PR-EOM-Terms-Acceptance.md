# PR-EOM-Terms-Acceptance

## Why this slice exists

Issue #2156 and the merged `PR-EOM-Terms-Authority` contract require every EOM
customer, residential or commercial, to accept the exact published Terms that
apply to that account. The authority slice can publish immutable bilingual
content, but it cannot initiate that obligation for a customer, authenticate a
customer response, prove what was accepted, deliver the executed copy, or tell
an operator whether a material revision requires reacceptance.

This is the second Atlas provider slice. It closes that provider path through
the real private/Tracker-proxied EOM API while leaving the Tracker bridge and
Website acceptance/admin UI to their subsequent consumer slices. The approved
issue shape includes Terms in the invitation email, a separately required
additional-work acknowledgement, an emailed executed copy, and manual
invitations for existing customers. Exact refined Terms prose remains an
operator-owned publication decision and is neither seeded nor changed here.

Diff-budget override: the invitation, acceptance, immutable evidence,
idempotent Resend delivery, guarded schema, and real API proof are one legal
evidence path. Splitting storage from the reachable acceptance route would land
either floating schema or a customer acceptance endpoint without protected
evidence.

### Problem-derived contract

- Root cause: the only Terms state on current `main` is document-version
  authority. There is no customer-bound, expiring bearer pinned to one
  immutable version; no acceptance record for the two independently required
  acknowledgements; no executed-copy delivery evidence; and no readiness rule
  relating an acceptance to later material versions. Consequently the system
  cannot safely invite a customer or prove that a specific customer accepted a
  specific published release. Reusing the profile-onboarding token row would
  also couple two obligations whose revocation and completion lifecycles are
  independent.
- Correct fix must touch/change: add a separately controlled DBA migration for
  guard-owned invitation, immutable acceptance, and content-stable delivery
  records; grant the runtime only the exact columns needed for issue, revoke,
  accept, claim, and confirm transitions; add a Terms-specific HMAC bearer
  domain while reusing the already validated public-onboarding key rotation and
  HTTPS base URL; add one service that derives audience/email from the canonical
  active EOM customer, pins the current published version, renders the selected
  published locale into invitation/executed-copy emails, serializes issue,
  acceptance, revocation, and readiness against the existing Terms publication
  lock, and drives the existing idempotent slim-profile Resend transport; add
  private issue/revoke/readiness/reconcile routes plus Tracker-proxied token
  session/accept routes; bind the service to the slim funnel pool; expose the
  controlled migration through `./ops`; and add focused route, boundary,
  concurrency, migration, and real-PostgreSQL proof enrolled in the EOM
  workflow.
- Must not change: do not change or seed Terms document content, version hashing
  or publication semantics; do not reuse or mutate existing
  `eom_public_onboarding_tokens`, onboarding drafts, first-clean receipts or
  candidates; do not change Customer/Site handoff, contact classification,
  Tracker, Website, calendar, employee timekeeping, billing, payment, Stripe,
  charge authorization, or card-vault behavior; do not add device
  fingerprinting or trust a browser-supplied customer/audience/email/IP; do not
  deploy, restart, apply either controlled migration, or send a production
  customer email in this PR.

### Contract revision

- New evidence: the invitation caller supplies only request key, contact, and
  locale; Atlas derives and pins the current version. Therefore a retry after a
  later publication cannot truthfully be compared to a caller-requested new
  version.
- Revised requirement: an existing request key replays its original pinned
  invitation across later publication changes when contact and locale match.
  Reusing that key for another contact or locale conflicts. A genuinely new
  invitation for the new current version requires a new request key.

## Scope (this PR)

Ownership lane: eom-onboarding-terms
Slice phase: Vertical slice
Max files: 18

1. Issue or idempotently replay a manual invitation for one active EOM customer,
   deriving residential/commercial and recipient email from the canonical
   contact and pinning the then-current published Terms version and selected
   `en`/`es` locale.
2. Deliver the invitation through the established slim-profile Resend sender
   with a persisted, content-stable payload and deterministic idempotency key;
   store only the token UUID/key fingerprint, never the raw bearer.
3. Resolve a valid token into the exact audience/locale document snapshot and
   accept only separate affirmative Terms and additional-work acknowledgements,
   a valid typed signer, and a trusted service-forwarded client IP.
4. Persist one append-only acceptance and one content-stable executed-copy
   delivery, attempt that delivery without rolling back a completed legal
   acceptance, and expose explicit reconciliation for a transport outcome that
   remained `sending`.
5. Report customer readiness against the current published version: later
   non-material releases preserve readiness, while any later material release
   requires a new invitation and acceptance.
6. Permit actor-audited revocation before acceptance; serialize revocation,
   acceptance, issue, and publication so every admitted ordering has one
   deterministic outcome.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_terms_acceptance.py` proves invitation issue reads the
    contact and current version inside the shared publication lock, rejects a
    non-customer, inactive/archived/unknown-type/no-email contact before any
    delivery claim, and never accepts audience or recipient from the caller.
  - The same test proves a request-key replay returns the original invitation
    and delivery without a second send even after publication advances, while
    the same key with another contact or locale conflicts.
  - Token tests prove the Terms bearer has a domain distinct from the existing
    profile token, authenticates only under the current or one previous key,
    binds that key fingerprint to the stored invitation, and places the bearer
    only in a URL fragment and transient transport body.
  - Route tests call the mounted EOM router for private invitation/revoke/
    readiness/reconcile and Tracker-proxied session/acceptance entrypoints; they
    prove service authentication, actor evidence where required, and a valid
    `X-EOM-Client-IP` are enforced before a state change.
  - Acceptance tests prove missing/false acknowledgements, invalid signer/IP,
    expired/revoked/unknown/wrong-key tokens, and a later material publication
    fail before acceptance; an unchanged replay returns one acceptance, and a
    changed signer conflicts.
  - The service execution model below makes publication/issue/accept/readiness
    linearize on one PostgreSQL advisory lock, revocation/acceptance linearize on
    the invitation row lock, and delivery claim linearize on
    `UPDATE ... WHERE status = 'pending' RETURNING`; focused concurrent tests
    settle the duplicate issue, double accept, accept-vs-revoke, and double-send
    invariants under that model.
  - Migration 397 tests prove the guard owner owns all three relations and guard
    functions, the direct `atlas` login has no table-wide write/delete/truncate
    or function-replacement authority, acceptance rows and delivery payloads
    cannot be rewritten, and only valid one-way revoke/delivery transitions are
    admitted.
  - Readiness tests prove acceptance of the current or an older release followed
    only by non-material releases is ready, and any later material release is
    `reacceptance_required`.
  - Invitation and executed-copy transport tests assert persisted body equality
    with the selected published audience/locale, separate acknowledgement
    evidence, deterministic Resend idempotency keys, and an acceptance that
    remains committed when receipt delivery needs reconciliation.
  - `./ops db controlled eom-terms-acceptance preflight|apply` dispatches only
    migration 397 after migration 396; capability/runbook evidence requires
    both migration receipts before deploying these routes and retains every
    Terms row on application rollback.
  - The diff changes no existing onboarding-token row, candidate, payment,
    Stripe, calendar, Tracker, Website, or Terms publication behavior; settled
    by cold diff reconstruction plus the adjacent focused suites named below.
- Reachability proof: an ASGI client calls the mounted slim EOM router. The
  observable results are a captured invitation email containing one fragment
  bearer, a token session returning the stored published snapshot, one
  immutable accepted row, one captured executed-copy email, and readiness that
  changes only after a material publication.
- Affected surfaces: controlled migration catalog/operation, EOM Terms token
  format, new acceptance service, EOM funnel models/dependencies/routes, slim
  pool binding, EOM pipeline CI enrollment, and focused tests.
- Risk areas: customer/account misbinding, legal-evidence mutability, raw-token
  persistence or logging, stale-version acceptance, key rotation, expiry,
  duplicate/uncertain email delivery, accept/revoke/publication races,
  actor/client-IP spoofing, rollback ordering, accidental production send.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R7, R8, R10, R11, R12,
  R14.

### Execution model

- PostgreSQL is the closed execution surface. Invitation issue, acceptance, and
  readiness acquire the same transaction-scoped advisory lock used by Terms
  publication. A publication and an issue/acceptance therefore have one lock
  order: issue pins the current row visible under its lock; acceptance either
  commits before a later material publication (and becomes not-ready after that
  publication) or observes that publication and rejects the stale invitation.
- Invitation creation linearizes on the unique request key. Revocation and
  acceptance each lock the invitation row; acceptance additionally linearizes
  on the unique invitation foreign key. Across every admitted interleaving,
  exactly one of revocation or first acceptance wins, and an unchanged accept
  replay returns the existing immutable row.
- Delivery payload creation commits with its invitation/acceptance. Delivery
  itself occurs after commit and claims exactly one pending row through a
  conditional update. The deterministic Resend idempotency key is a second
  boundary against transport replay. A failed/unknown transport leaves
  `sending` reconciliation evidence; it never rolls back or silently retries a
  legal acceptance. An authenticated actor may confirm `sending -> sent` only
  after external reconciliation.
- Transaction cancellation, connection loss, or process death before commit
  releases locks and rolls back the whole database transition. No lease, file
  lock, background scheduler, cross-database write, or caller-supplied clock
  participates in correctness.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: Terms-specific bearer formatting/authentication; private
  issue/revoke/readiness/reconcile routes; Tracker-proxied session/acceptance;
  invitation/acceptance/delivery database transitions.
- Replaced-path behaviors: none. Existing profile-onboarding tokens and Terms
  publication remain separate and unchanged.
- Guard-relevant fields: requestKey, contactId, locale, invitation UUID,
  delivery UUID, token grammar/MAC/key fingerprint, signerName,
  termsAccepted, additionalWorkAccepted, trusted client-IP header, actor
  id/name, database-derived audience/email/current version/timestamps, delivery
  status.
- Caller x input shape: authenticated office/Tracker bearer plus actor headers x
  `{requestKey, contactId, locale}` for issue; bearer+actor x invitation for
  revoke; bearer x contact for readiness; bearer x raw Terms token for session;
  bearer+raw Terms token+two strict `true` acknowledgements+typed signer+trusted
  IP header for acceptance; bearer+actor x `sending` delivery for reconciliation.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: reuse the existing, disabled-by-default
  `ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_*` HTTPS URL/current+previous HMAC key
  boundary and issuance override. No new environment fallback is added. The
  currently running Brain predates migration 396, so deployment of this slice
  requires controlled 396 then 397 receipts before route rollout.
- Explicit value probe: current-key issue/session/accept and previous-key
  redemption tests with issuance enabled.
- Absent value probe: disabled issuance, missing/unsafe public-onboarding
  configuration, missing service bearer/actor/client IP, and unavailable schema
  all fail before the relevant mutation or send.
- Default-session/default-context probe: every invitation derives and enforces
  `effingham_maids`; no caller-supplied tenant, audience, recipient, timestamp,
  or Terms version is accepted.
- Side-effect ordering: issue preflights config and email transport before the
  invitation transaction; acceptance commits immutable evidence and a pending
  executed-copy delivery before any email call; both email calls claim their
  persisted payload before transport and confirm only after transport
  acceptance.

### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/database.md`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_terms_acceptance.py`
- `atlas_brain/services/eom_terms_authority.py`
- `atlas_brain/storage/migrations/397_eom_terms_acceptance.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `ops`
- `plans/PR-EOM-Terms-Acceptance.md`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `scripts/apply_eom_terms_acceptance_schema.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_eom_terms_acceptance.py`
- `tests/test_migrations_runner.py`

## Mechanism

Migration 397 adds separately guard-owned invitation, acceptance, and delivery
relations. Invitation identity is a UUID signed in a Terms-specific HMAC domain;
the database stores only that UUID and the admitting key fingerprint. The issue
transaction locks Terms publication, derives the account audience/email from
`contacts`, pins the current immutable version, persists the exact selected
email payload, and replays only an identical request key. A single conditional
delivery claim calls the established slim-profile Resend sender with a stable
idempotency key.

Session and acceptance authenticate the bearer before database access and bind
the verified key fingerprint to the invitation. Acceptance uses database time,
the invitation row lock, and the publication lock to validate expiry,
revocation, customer/version identity, and intervening material releases. It
then inserts one immutable acceptance and the exact executed-copy email payload
in the same transaction. Readiness searches accepted versions and treats only a
later published `material_change = true` release as requiring reacceptance.
Routes reproject closed response models and the slim app injects the same EOM
funnel pool used by the authority.

## Intentional

- The existing public-onboarding HTTPS base URL and current/previous HMAC key
  rotation are reused, but the Terms token grammar and durable rows are
  separate. This shares provisioned cryptographic/configuration rails without
  coupling profile-token completion or revocation state.
- Invitation and executed-copy emails contain the selected operator-published
  Terms, Services We Cannot Provide, and additional-work acknowledgement as
  text. No placeholder or source-PDF prose is seeded, refined, translated, or
  published here.
- Customer IP is accepted only from a required service-forwarded header on the
  authenticated Tracker boundary. The Atlas socket peer is the Tracker, and a
  body field would be browser-controlled; no device fingerprint is collected.
- A newly issued link expires after 30 days using database time. This bounds
  bearer lifetime without adding caller-controlled expiry or another deployment
  setting; a fresh actor-audited invitation is required after expiry.
- Delivery uncertainty remains explicit `sending` evidence and requires
  operator reconciliation. There is no automatic retry after an ambiguous
  external side effect.
- This provider can send only when an authenticated private caller explicitly
  issues an invitation. It adds no scheduler, bulk send, or production data
  mutation.

## Deferred

- Tracker proxy/capability contract for Terms invitation, session, acceptance,
  readiness, revocation, and delivery-reconciliation routes.
- Website Onboarding tab, public bilingual acceptance page, admin readiness and
  reconciliation UI, and operator-approved refined bilingual Terms publication.
- Structured one-time/recurring service plan, first-clean payment allocation,
  and residential Stripe SetupIntent/card-vault onboarding remain later slices
  in #2156.

Parking predicate: UI/presentation/copy polish, bulk invitation tooling,
automatic delivery retry, delivery analytics, and payment/card behavior are
parked unless they prove this provider can misbind a customer, mutate legal
evidence, leak a bearer, send twice, or report false readiness.

Parked hardening: none.

## Verification

- Passed locally: the canonical EOM workflow test list against disposable
  PostgreSQL (`1681 passed, 5 skipped`); the isolated Terms suite with live
  migration 396/397, runtime ACL, immutability, boundary, and concurrency proof
  (`27 passed`); controlled-runner/operations/migration tests
  (`245 passed, 1 skipped`); targeted Ruff, format-check, compile, and diff
  check.
- Pending: synchronized-plan check and mechanical pre-push review.
- Hosted: EOM pipeline disposable-PostgreSQL proof and GitHub-only Unit Gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 11 |
| `.agent/runbooks/database.md` | 50 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 9 |
| `atlas_brain/eom_api/funnel.py` | 369 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/eom_terms_acceptance.py` | 1814 |
| `atlas_brain/services/eom_terms_authority.py` | 4 |
| `atlas_brain/storage/migrations/397_eom_terms_acceptance.sql` | 875 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 5 |
| `plans/PR-EOM-Terms-Acceptance.md` | 346 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 24 |
| `scripts/apply_eom_terms_acceptance_schema.py` | 42 |
| `tests/test_agent_operations_contract.py` | 35 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 95 |
| `tests/test_eom_terms_acceptance.py` | 1240 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **4924** |
