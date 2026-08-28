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

### Review-evidence contract revision

- New evidence: a delivery claim previously serialized only the delivery row,
  so acceptance or revocation could commit after the invitation usability read
  and before transport. Material-release comparisons also used wall-clock
  timestamps, which do not define a strict release order, and the delivery
  evidence guard rejected Resend's provider-confirmed idempotent replay shape
  when that replay has no message id. Further review proved that keeping claim,
  provider call, and confirmation in one transaction erased `sending` when the
  provider accepted but database confirmation failed; contact identity was not
  revalidated or locked through transport; readiness still selected among
  acceptances by wall-clock time; and receipt identity fields admitted embedded
  line controls. A later review proved that the shared contact comparison was
  still enforced only for invitation delivery, allowing an executed-copy
  receipt to target a stale address after acceptance.
- Revised requirement: delivery first commits a durable `pending -> sending`
  claim before crossing the provider boundary. It then reacquires the
  publication, invitation, delivery, and canonical contact locks, revalidates
  material currency and contact identity for both invitation and executed-copy
  delivery, and retains those locks through the bounded provider outcome and
  database confirmation. Any uncertain provider or confirmation result remains
  `sending` and is never automatically retried; acceptance and revocation reject
  that reconciliation state. Migration 397
  adds a guard-owned monotonic publication order assigned under the existing
  publication lock, every later-material decision and readiness selection uses
  that order, and a sent transition admits a missing provider message id only
  when the established sender explicitly reports
  `idempotent_replay = true`. Shared signer/customer/actor text rejects every
  non-printable line-control character before it can enter receipt evidence.
- Preserved boundary: this internal serialization and release-order evidence
  does not change Terms prose, version hashes, the operator publish/current
  selection API, route payloads, customer classification, payment/onboarding
  state, or production delivery authority.

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
    the invitation row lock, and delivery commit its `pending -> sending` claim
    before transport, then hold publication, invitation, delivery, and contact
    locks through the provider outcome; focused concurrent tests settle
    duplicate issue, double accept, accept-vs-revoke, send-vs-revoke,
    contact-update-vs-send, confirmation-failure, and double-send invariants
    under that model.
  - Migration 397 tests prove the guard owner owns all three relations and guard
    functions, the direct `atlas` login has no table-wide write/delete/truncate
    or function-replacement authority, acceptance rows and delivery payloads
    cannot be rewritten, and only valid one-way revoke/delivery transitions are
    admitted.
  - Readiness tests prove acceptance of the current or an older release followed
    only by non-material releases is ready, and any later material release is
    `reacceptance_required`, including when release or acceptance wall-clock
    timestamps contradict the database-owned publication order.
  - Invitation and executed-copy transport tests assert persisted body equality
    with the selected published audience/locale, separate acknowledgement
    evidence, deterministic Resend idempotency keys, rejection after canonical
    contact drift for either delivery kind, and an acceptance that remains
    committed when receipt delivery needs reconciliation.
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
  acquires the publication, invitation, delivery, and contact locks, validates
  canonical recipient for both delivery kinds and current material release for
  invitations, and commits exactly one `pending -> sending` claim before
  transport. It then reacquires the same locks, revalidates every condition,
  and retains them through the bounded provider result and `sent` confirmation.
  A failed/unknown sender call or a database failure after provider acceptance
  leaves durable `sending` reconciliation evidence; later invocations do not
  cross the provider boundary again. No background retry is introduced, and
  actor confirmation remains limited to a persisted `sending` state after
  external reconciliation. Acceptance and revocation cannot race through that
  state.
- Published Terms versions receive a database-assigned positive
  `publication_order` under the existing publication advisory lock. Historical
  releases are backfilled deterministically with the selected current release
  last. Later-material acceptance and readiness decisions compare this order,
  never the non-monotonic wall clock.
- Transaction cancellation, connection loss, or process death before commit
  releases locks and rolls back the whole database transition. No lease, file
  lock, background scheduler, cross-database write, or caller-supplied clock
  participates in correctness.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: _terms_authority_dependency
- Replaced-path behaviors: none; this adds the slim-app Terms authority
  dependency without replacing another dependency resolver.
- Guard-relevant fields: initialized EOM funnel pool and migration-396 schema
  readiness.
- Caller x input shape: mounted slim EOM app x injected authority service.

- Boundary path/seam: _authenticated_terms_token
- Replaced-path behaviors: none; the existing profile-onboarding token remains
  a separate bearer domain.
- Guard-relevant fields: raw Terms token, current/previous HMAC secret, token
  grammar, MAC, invitation UUID, and key fingerprint.
- Caller x input shape: authenticated Tracker proxy x raw Terms token for
  session or acceptance.

- Boundary path/seam: EOMTermsAcceptanceValidationError
- Replaced-path behaviors: none; this adds a typed failure for the new Terms
  boundary.
- Guard-relevant fields: request key, locale, UUIDs, acknowledgements, signer,
  actor, trusted client IP, email, and receipt identity text.
- Caller x input shape: malformed private/public Terms input x closed validation
  error before database mutation.

- Boundary path/seam: AuthenticatedEOMTermsToken
- Replaced-path behaviors: none; no existing token value object is reused or
  changed.
- Guard-relevant fields: authenticated invitation UUID and signing-key
  fingerprint only; the raw bearer is excluded.
- Caller x input shape: successful Terms token authentication x closed internal
  identity value.

- Boundary path/seam: authenticate_eom_terms_token
- Replaced-path behaviors: none; profile-onboarding authentication stays
  separate and unchanged.
- Guard-relevant fields: token prefix/grammar, UUID, MAC, current key, previous
  key, and derived fingerprint.
- Caller x input shape: raw Terms bearer x current/previous configured key.

- Boundary path/seam: _require_authenticated_token
- Replaced-path behaviors: none; this narrows service entry to the authenticated
  Terms token type.
- Guard-relevant fields: runtime token type, invitation UUID, and key
  fingerprint.
- Caller x input shape: service session/accept call x authenticated internal
  token value.

- Boundary path/seam: eom_terms_authority_schema_ready
- Replaced-path behaviors: none; migration-396 readiness gains exact
  compatibility with migration 397 when installed.
- Guard-relevant fields: Terms authority relations, triggers, guard functions,
  ownership, runtime privileges, and optional publication-order guard.
- Caller x input shape: EOM authority service x current database schema.

- Boundary path/seam: eom_terms_authority_schema_ready#2
- Replaced-path behaviors: none; this is another changed hunk of the same schema
  readiness boundary.
- Guard-relevant fields: Terms authority relations, triggers, functions,
  ownership, and runtime privileges.
- Caller x input shape: EOM authority service x incomplete schema.

- Boundary path/seam: eom_terms_authority_schema_ready#3
- Replaced-path behaviors: none; this is another changed hunk of the same schema
  readiness boundary.
- Guard-relevant fields: publication-order column, assignment trigger, and guard
  ownership when migration 397 exists.
- Caller x input shape: EOM authority service x migration-396-only or
  migration-396-plus-397 schema.

- Boundary path/seam: eom_terms_authority_schema_ready#4
- Replaced-path behaviors: none; this is another changed hunk of the same schema
  readiness boundary.
- Guard-relevant fields: owner-role membership and guard-function execution
  privileges.
- Caller x input shape: Atlas runtime role x guarded Terms authority schema.

- Boundary path/seam: eom_terms_authority_schema_ready#5
- Replaced-path behaviors: none; this is another changed hunk of the same schema
  readiness boundary.
- Guard-relevant fields: forbidden table-wide mutation and function replacement
  privileges.
- Caller x input shape: Atlas runtime role x authority object privilege matrix.

- Boundary path/seam: eom_terms_authority_schema_ready#6
- Replaced-path behaviors: none; this is another changed hunk of the same schema
  readiness boundary.
- Guard-relevant fields: exact column grants for draft creation and publication
  transitions.
- Caller x input shape: Atlas runtime role x admitted authority writes.

- Boundary path/seam: EOMTermsAuthority
- Replaced-path behaviors: none; this adds the Terms publication authority and
  leaves customer acceptance in its separate service.
- Guard-relevant fields: immutable bilingual documents, version labels, material
  flag, content hash, publication order, actor evidence, and current selection.
- Caller x input shape: authenticated operator draft/publish request x guarded
  migration-396 authority state.

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
  invitation transaction; acceptance commits immutable evidence and a newly
  created executed-copy delivery before any email call; both email calls
  durably commit their persisted payload as `sending` before transport,
  revalidate under the canonical locks, and confirm only after transport
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
email payload, and replays only an identical request key. A durable conditional
delivery claim commits before the provider boundary; a second transaction
revalidates the canonical contact for invitation and executed-copy delivery and
holds publication/invitation/delivery/contact locks while it calls the
established slim-profile Resend sender with a stable idempotency key and
persists either the provider message id or its explicit idempotent-replay result.

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
- Delivery uncertainty from the sender or confirmation database remains
  explicit `sending` evidence and requires operator reconciliation. There is no
  autonomous or later-call provider retry after an ambiguous external side
  effect.
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

- Passed locally after review fixes: the focused acceptance, authority, and
  controlled migration-runner suite against disposable PostgreSQL
  (`206 passed, 1 skipped`), targeted Ruff and Ruff format checks, Python
  compilation, and `git diff --check`.
- Passed: synchronized-plan, boundary-enumeration, deployed-config, and
  PR-side consistency checks.
- Hosted: EOM pipeline disposable-PostgreSQL proof and GitHub-only Unit Gate.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 11 |
| `.agent/runbooks/database.md` | 50 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 9 |
| `atlas_brain/eom_api/funnel.py` | 369 |
| `atlas_brain/main_eom.py` | 1 |
| `atlas_brain/services/eom_terms_acceptance.py` | 1930 |
| `atlas_brain/services/eom_terms_authority.py` | 108 |
| `atlas_brain/storage/migrations/397_eom_terms_acceptance.sql` | 1045 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 5 |
| `plans/PR-EOM-Terms-Acceptance.md` | 479 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 24 |
| `scripts/apply_eom_terms_acceptance_schema.py` | 42 |
| `tests/test_agent_operations_contract.py` | 35 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 95 |
| `tests/test_eom_terms_acceptance.py` | 1801 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **6008** |
