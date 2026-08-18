# PR-EOM-Public-Onboarding-Token-Authority

## Why this slice exists

Juan authorized the next EOM funnel vertical slice: a customer who has received
an approved onboarding email should be able to complete the operations-only
onboarding data without a staff login. Atlas issues the approved email today,
but its current draft lifecycle is text-only; its only customer finalizer is the
separate, actor-authenticated office handoff. The current source therefore has
no authority path from an approved email to a safe public completion.

This is the Atlas A4 foundation for Atlas #2188/#2275. Tracker and Website
companions will use this private authority in later PRs; this PR remains
dormant until those deployed callers and the explicit Atlas configuration are
available.

This foundation intentionally exceeds the normal diff target because durable
token state, issuance, service-only redemption, the office-handoff fence,
revocation, and their transaction proofs form one safety boundary. Splitting
them would leave either a bearer without durable revocation/fencing, a durable
row without its only safe redemption path, or a caller-facing route without the
proof that its public and office completion paths cannot race.

### Problem-derived contract

- Root cause: The approved onboarding draft is only an email snapshot and the
  existing `customer-handoffs` endpoint is only an office command. Neither
  creates a revocable bearer capability nor coordinates that capability with
  the one immutable Atlas Customer/Site handoff. Exposing the generic customer
  writer would bypass the CRM ownership boundary; letting the public and office
  finalizers race would permit an orphaned tracker record or competing customer
  transition.
- Correct fix must touch/change:
  1. Add typed, disabled-by-default EOM funnel configuration for an HTTPS public
     onboarding base URL and an Atlas-only HMAC secret; incomplete or unsafe
     explicit configuration fails closed.
  2. Add an additive Atlas migration and startup-readiness check for durable,
     revocable, one-use onboarding-token state. The database stores an opaque
     token id, lifecycle state, approved-office actor, and completed handoff
     reference -- never the bearer token itself.
  3. Add one canonical HMAC token formatter/parser and private service routes
     for tracker-only session lookup/finalization and staff-authorized link
     revocation. The route responses whitelist only tracker reconciliation and
     browser-prefill fields, never Atlas contact or handoff ids. Browser callers
     never receive Atlas credentials or call Atlas directly. Revocation remains
     available to a private office actor after public issuance is disabled, so
     an issued link cannot strand office handoff behind its own safety fence.
  4. Extend the existing approved-draft claim only while the feature is enabled:
     atomically mint the token with `pending -> sending`, build a fragment-based
     link only for the email transport, and keep the token out of the persisted
     draft/history body. The added transport-only invitation carries concise
     English and Spanish copy because the current draft model has no customer
     language preference. The enabled claim takes the shared contact handoff
     lock before its lead predicate, so an office handoff cannot win between
     that predicate and token insertion; disabled configuration retains the
     current email body and state machine exactly.
  5. Reuse the existing immutable Customer/Site handoff transaction for public
     completion, under the employee who approved the email and with explicit
     `completion_channel` lifecycle metadata. The shared contact lock fences an
     active public token from the office finalizer; the public finalizer and
     token redemption commit as one PostgreSQL transaction.
  6. Add route, token-grammar, configuration, migration/readiness, normal
     approval, revocation, replay, and cross-finalizer tests that prove the
     above from real FastAPI and disposable-Postgres entrypoints.
- Must not change:
  1. Do not add or revive the stale unique `customers.atlas_contact_id` index;
     current tracker source intentionally permits several operational Customers
     to share an Atlas contact and this slice does not alter that topology.
  2. Do not expose the generic Atlas operator-contact/customer writer, add an
     unauthenticated Atlas route, expose a service bearer, log a raw onboarding
     token, or reuse the tracker QR/JWT signing secret.
  3. Do not alter the existing office handoff request/response contract,
     generic onboarding email behavior while public onboarding is disabled,
     lead-stage vocabulary, booking/calendar logic, payment/receivables,
     payroll, QR/GPS, scheduling, or customer-type semantics.
  4. Do not create tracker Customers/Sites, add a tracker public route, add a
     Website page, change staff portal copy, enable production configuration,
     send test email, or mutate production data in this PR.

### Current-head review remediation contract

- Root cause: The first implementation made the issued bearer and the ready
  projection depend on mutable runtime state: one configured HMAC secret and
  the contact's current PII. It also normalized an operator URL before rejecting
  control characters, left the safe-disable/revoke-before-rollback sequence
  undocumented, and did not enroll the new authority proof in the explicit EOM
  pull-request test list.
- Correct remediation must:
  1. Keep the existing opaque bearer grammar, mint with the primary HMAC secret,
     and persist only a non-secret signing-key fingerprint with each token.
     Accept at most one explicitly configured, validated previous verifier;
     require the returned verifier fingerprint to match the token row so an old
     key cannot authenticate a bearer for a newly issued row. This preserves
     outstanding links across one controlled secret rotation without storing a
     raw bearer or raw secret.
  2. Persist the exact ready-state prefill projection with the token inside the
     issuance transaction, and return that immutable snapshot while the token is
     issued. Contact lifecycle predicates still use the live contact row, but a
     later office correction cannot disclose new PII through an old link.
  3. Reject every ASCII control character in the raw public onboarding URL
     before URL parsing or transport-link construction, and extend the
     independent configuration oracle to cover that input family and the
     bounded previous-verifier rules.
  4. Document the ordered rollback: disable issuance, use the existing private
     `revoke-link` recovery command to drain every issued token, verify no
     `issued` row remains, retain the audit table, then roll back application
     code. The current release must remain deployed until that drain succeeds.
  5. Enroll the new public-onboarding unit/route suite and authority source/
     migration paths in the explicit EOM pull-request workflow.
- Must not change: token text/version, service-only route authentication,
  browser/Tracker scope, office-handoff request or response shapes, generic
  CRM writers, tracker Customer/Site creation, production configuration, or any
  payroll, QR/GPS, scheduling, payment, and existing disabled-issuance email
  behavior.

### Issuance-pause review remediation contract

- Root cause: `public_onboarding_enabled` currently carries two incompatible
  decisions. It tells `approve-send` whether to mint a new bearer, but the same
  value also gates the configured HMAC authority used by session/finalize.
  Turning it off to pause issuance therefore returns 503 before an already
  issued bearer reaches HMAC or durable-token validation, even though that row
  still fences the office handoff. This change fixes that upstream control-model
  error, not the downstream 503 symptom.
- Correct remediation must:
  1. Retain `public_onboarding_enabled` as the existing, migration-gated
     authority switch for the private service routes so absent authority remains
     fail-closed and existing deployment configuration remains compatible.
  2. Add a typed optional issuance override whose unset value derives the
     legacy enabled behavior. With authority enabled and that override false,
     `approve-send` must retain its ordinary draft claim/send/confirm flow
     without minting or transporting a new bearer, while session/finalize can
     still authenticate a valid outstanding bearer through the existing private
     service boundary.
  3. Reject an explicit request to issue while the authority is disabled. Do not
     loosen HMAC, URL, API-bearer, token-status, revocation, key-fingerprint,
     migration-readiness, or office-handoff checks.
  4. Prove all three states through the real ASGI routes: fully disabled
     authority returns 503; enabled authority with the pause override permits a
     valid session and finalization; and the same pause sends no new link.
     Update the rollback instructions so an operator pauses issuance without
     disabling redemption before issued rows are drained.
- Must not change: the meaning of a fully disabled authority, the legacy
  configured-and-enabled issuance default, token format/rotation, service
  authentication, persistence/migrations, revocation, office-fence semantics,
  tracker/Website scope, production configuration, or any unrelated EOM
  lifecycle, payment, payroll, QR/GPS, or scheduling behavior.

### URL-whitespace review remediation contract

- Root cause: URL validation rejects C0/DEL control characters but accepts
  printable and Unicode whitespace because `urlsplit` can still return a
  truthy hostname. The original accepted value is later used to construct the
  email link, so a whitespace-bearing URL is structurally admitted at startup
  yet cannot be delivered as one usable link. This is an input-admission defect,
  not an email-rendering symptom.
- Correct remediation must:
  1. Reject every raw URL value containing whitespace before URL parsing, while
     retaining the existing C0/DEL rejection and HTTPS/credential/query/
     fragment/secret validation.
  2. Extend the independent configuration oracle and its generated input
     families to cover ordinary and Unicode whitespace, so an acceptance change
     cannot silently re-admit a link-breaking value.
  3. Reject rather than canonicalize whitespace: valid configured base URLs
     retain their exact behavior and no surprising address mutation reaches an
     approval email.
- Must not change: issuance-pause behavior, the configured redemption authority,
  token grammar/rotation, service authentication, persistence/migrations,
  revocation/fence behavior, valid URL semantics, Tracker/Website scope,
  production configuration, or any unrelated EOM lifecycle, payroll, QR/GPS,
  scheduling, payment, or email-copy behavior.

### Current-head CI recovery contract

- Root cause: This unmerged PR assigned numeric migration prefix `382` to the
  EOM public-onboarding schema while `origin/main` already contains
  `382_commercial_billing_candidate_overrides.sql`. The migration runner treats
  the prefix as its primary version and the repository test deliberately admits
  only historic collisions, so the newly introduced collision fails both the
  dedicated migration-runner check and the EOM pipeline that includes it. In a
  separate policy defect, the PR body describes why this over-cap safety
  boundary is indivisible but omits the exact visible `Diff-budget override:`
  decision marker required by the diff-budget gate.
- Correct remediation must:
  1. Rename the unmerged EOM migration to the next free positive prefix after
     the current `origin/main` maximum (`383`) rather than add this new collision
     to the historic-exception allowlist.
  2. Update every direct filename and canonical migration-name reference in the
     EOM workflow enrollment, integration migration setup, focused migration
     proof, plan, and PR body so the migration is discovered, applied, and
     recorded consistently as `383_eom_public_onboarding_tokens`.
  3. Add one visible, substantive `Diff-budget override:` line to the PR body
     using the already-established indivisibility rationale, so the gate records
     the deliberate overage instead of inferring one from unrelated prose.
  4. Prove the repository-wide duplicate-prefix policy and exact EOM workflow
     test list pass locally; the existing collision test is the regression proof
     that the new migration no longer becomes a silently accepted duplicate.
- Must not change: the migration SQL/schema and its transactional semantics,
  migration-runner collision recovery or historic allowlist, public-onboarding
  token/configuration/authorization behavior, production migration state, EOM
  test selection other than renamed path enrollment, or the diff-budget
  threshold/gate policy. Do not split or broaden the product slice solely to
  silence this metadata and naming repair.

### Current-head recipient-snapshot and dormant-recovery contract

- Root cause: The enabled draft claim intentionally resolves a repeat
  intake's newer approved recipient into `eom_onboarding_email_drafts`, but
  token issuance then copies mutable `contacts.email` into `prefill_email`.
  A newly approved recipient can therefore receive a valid link whose immutable
  prefill discloses a different, stale address. Separately, startup treats the
  public-token table as entirely irrelevant while authority is disabled, even
  when the table already exists. That leaves the normal office-handoff fence
  (`SELECT ... FOR UPDATE`) and the private `revoke-link` recovery command
  able to fail later on missing columns or runtime privileges instead of
  failing closed at startup.
- Correct remediation must:
  1. Bind `prefill_email` in the token insert to the successfully claimed
     draft's non-null `recipient_email`, not to `contacts.email`. Preserve the
     existing token-time snapshot of the other contact fields and the existing
     draft claim/recipient predicate.
  2. Split datastore readiness into (a) dormant-safe recovery checks that run
     whenever `eom_public_onboarding_tokens` exists and verify exactly the
     relation columns used by the unconditional office fence/revocation paths
     (`id`, `draft_id`, `contact_id`, `status`, `revoked_at`) plus runtime
     `SELECT` and `UPDATE`, and (b) enabled-only issuance checks for the full
     durable token shape, constraints/index, and `INSERT` privilege. A missing
     token relation remains compatible only while authority is disabled.
  3. Prove both seams through disposable Postgres: a latest-intake recipient
     becomes the draft and issued-token prefill email; an existing token table
     missing a recovery column or `SELECT`/`UPDATE` access rejects disabled
     startup, while a missing table and an issuance-only missing column retain
     the intentionally dormant behavior.
- Must not change: the migration schema or its SQL, raw-bearer handling,
  HMAC/token grammar, feature/route authorization, issuance-pause semantics,
  office-handoff request/response or locking semantics, draft email transport,
  generic CRM writers, Tracker/Website scope, production configuration, or
  any payroll, QR/GPS, scheduling, payment, and unrelated EOM behavior. The
  only disabled-state behavior change is fail-closed readiness when an already
  present token relation cannot support its still-reachable recovery paths.

### Current-head browser-normalized separator contract

- Root cause: the URL validator delegates raw reverse-solidus (`\`) handling
  to Python's `urlsplit`. For
  `https://example.com\evil.com/onboarding`, that parser reports a truthy
  HTTPS hostname and the current structural predicate accepts the value. The
  browser URL parser instead treats the reverse solidus as a path separator and
  opens `https://example.com/evil.com/onboarding`. Because the original raw
  value is used later to compose the approval link, the two parsers disagree
  about the address that will be delivered. This is a configuration-admission
  defect, not a link-rendering or token-authorization defect.
- Correct remediation must:
  1. Reject every raw public-onboarding URL containing a reverse solidus before
     `urlsplit` or email-link construction, alongside the existing C0/DEL and
     whitespace rejection. Reject rather than canonicalize or repair the
     operator value.
  2. Extend the independent configuration oracle and generated URL families
     with a reverse-solidus authority family, and directly prove both the
     authority and path forms reject. Valid HTTPS URLs retain their existing
     acceptance and delivery behavior.
  3. Preserve all existing HTTPS/host/credential/port/query/fragment and
     paired-secret predicates; this repair must not make either Python or a
     browser URL parser the authority for normalizing malformed input.
- Must not change: valid URL semantics, issuance-pause behavior, configured
  redemption authority, token grammar/rotation, service authentication,
  persistence/migrations, revocation/fence behavior, email-copy/transport,
  Tracker/Website scope, production configuration, or any unrelated EOM
  lifecycle, payroll, QR/GPS, scheduling, payment, or CRM behavior.

## Scope (this PR)

Ownership lane: eom-public-onboarding-token-authority
Slice phase: vertical slice
Max files: 13

1. Establish dormant Atlas token issuance, validation, explicit revocation, and
   one-time finalization authority for one active `won` EOM lead.
2. Preserve one durable `eom_customer_handoffs` record for either completion
   channel and make the public-token channel wait/fail safely rather than race
   the existing office channel.
3. Publish tracker-only service endpoints, not browser endpoints, with a
   narrow prefill response and no generic CRM mutation surface.
4. Prove the new surface through the router and a disposable PostgreSQL
   transaction suite before a tracker or Website caller is introduced.
5. Repair the current-head issuance-pause control seam without changing the
   durable authority, datastore-readiness, or handoff state machine.
6. Reject link-breaking whitespace at public-onboarding URL admission without
   canonicalizing operator input or changing the enabled delivery flow.
7. Repair the unmerged migration's numeric-prefix collision and record the
   existing indivisibility rationale in the canonical PR-body diff-budget form.
8. Bind the immutable public prefill email to the approved draft recipient and
   fail disabled startup only when an already-present token relation cannot
   serve its still-reachable office-fence/revocation recovery paths.
9. Reject browser-normalized reverse solidi at public-onboarding URL admission
   without canonicalizing an operator value or changing valid URL delivery.

### Review Contract

- Acceptance criteria:
  1. With public onboarding disabled, `POST .../approve-send` follows its
     current claim/send/confirm path and submits the stored draft body unchanged;
     the focused HTTP approval regression test settles this behavior. The open
     configuration tuple is checked over a generated URL/secret/flag grammar
     against an independent contract oracle, so unsafe or incomplete values
     cannot accidentally take the enabled branch, including an embedded control
     character or an invalid bounded previous verifier.
  2. With valid explicit public configuration, one approved, active `won` lead
     gets exactly one opaque HMAC token during the same transaction that claims
     its draft. The email transport sees an HTTPS URL whose token is in the
     fragment, while the persisted draft and email-history payload contain no
     bearer. The token row records an immutable, token-time prefill projection
     and non-secret signing-key fingerprint; focused service tests settle this
     behavior.
  3. The token parser is the single admission choke point for the closed
     `eomob1.<UUID>.<base64url-HMAC>` grammar. Malformed, unknown-version, or
     MAC-mismatched strings make no CRM lookup or mutation; generated mutation
     tests settle its out-of-grammar rejection. Its grammar-product test derives
     the expected verdict from the documented HMAC construction across version,
     UUID, signature, framing, and direct input-shape families rather than from
     the parser or formatter under test.
  4. The tracker-only session and finalize routes require the existing private
     EOM funnel bearer but no fabricated employee actor. Route tests exercise
     the real FastAPI entrypoints and prove that a browser-shaped unauthenticated
     request cannot obtain prefill or finalize a token; their whitelisted
     responses omit Atlas contact and handoff identifiers. A one-generation
     HMAC rotation accepts the configured prior verifier only for a row minted
     under that verifier, and post-issuance contact edits cannot alter prefill.
  5. An issued token can resolve only its active EOM `lead` at `won`; a revoked
     token resolves nothing; a redeemed token replays only the tracker
     Customer/Site ids already committed. Disposable-Postgres tests settle the
     status and contact predicates.
  6. The execution model is one Postgres transaction per enabled issuance or
     finalization. The enabled claim plus both public and office paths obtain
     the shared per-contact handoff advisory lock before their decisive lead/
     handoff/token check; the invariant is at most one handoff for the contact
     and only the matching token can redeem it. Integration tests exercise
     issuance locking, duplicate, and competing-channel calls; uniqueness and
     the immutable handoff table remain the database backstops.
  7. Staff link revocation changes only an issued token to revoked, makes its
     session/finalization unavailable, and releases the office handoff fence;
     it never rewrites a sent email or a redeemed handoff. It stays behind the
     existing private bearer and office-actor checks even if public issuance is
     disabled, solely to recover an already-issued link. Route plus database
     tests settle this behavior.
  8. Existing lead conversion, draft approval, capability-manifest, and
     migration-readiness tests remain green; the public-onboarding route/token/
     configuration suite is enrolled in the explicit EOM pull-request workflow.
     Together these prove the staff paths and previously advertised API shapes
     retain their contracts.
  9. `public_onboarding_enabled` remains the authority/readiness switch. Its
     optional issuance override defaults to the legacy effective value, rejects
     `true` when authority is disabled, and may be explicitly false to stop
     future tokenized emails while real ASGI session/finalize requests for an
     outstanding valid bearer remain service-authenticated and reachable. A
     fully disabled authority still returns 503, and the paused approval route
     sends no fragment link; focused route/configuration tests settle each
     observable state.
  10. A public-onboarding base URL containing ordinary or Unicode whitespace is
      rejected before parsing or email construction; its independent grammar
      oracle rejects the same families, while a valid configured URL retains the
      existing issuance behavior.
  11. The public-onboarding migration has a unique positive prefix against the
      current repository migration set; its EOM workflow trigger and integration
      proof reference that same canonical name. The PR body carries a visible,
      substantive diff-budget override rather than changing the soft-cap policy.
  12. The issued token's immutable `prefill_email` equals the approved draft
      recipient even when repeat intake intentionally leaves `contacts.email`
      stale. With authority disabled, a missing token table remains compatible,
      but an existing table must expose the fence/revocation columns and runtime
      `SELECT`/`UPDATE` privileges before startup succeeds; enabled issuance
      retains the stricter full-shape and `INSERT` readiness requirement.
  13. A raw reverse solidus in either public-onboarding URL authority or path is
      rejected before Python URL parsing or email construction. The independent
      configuration oracle's generated authority family rejects the same input,
      while an ordinary valid HTTPS URL retains its existing behavior.
- Reachability proof: `tests/test_eom_public_onboarding.py` uses an ASGI
  transport against the registered `/eom-funnel/public-onboarding/*` routes;
  `tests/test_eom_lead_conversion_integration.py` uses a disposable Postgres
  schema to observe the token row, contact transition, and immutable handoff.
- Affected surfaces: Atlas EOM typed configuration; draft-approval sender;
  private FastAPI funnel router/auth; CRM provider handoff transaction; EOM
  funnel readiness; additive migration; focused tests. The future tracker and
  Website consumers are documented dependencies but are not touched here.
- Risk areas: bearer secrecy, authorization, PII disclosure, token replay,
  competing office/public conversion, crash/retry consistency, feature-flag
  rollout, migration/privilege safety, and backward-compatible email delivery.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: canonical public-token parser/validator.
  - Replaced-path behaviors: no prior public-token parser exists; every raw
    browser token currently has no Atlas route.
  - Guard-relevant fields: version, canonical UUID token id, signature encoding,
    constant-time HMAC comparison, durable token status, draft status, and
    contact business-context/type/stage/status.
  - Caller x input shape: tracker session/finalize client x arbitrary JSON
    token string; malformed/unknown/invalid inputs reject before data access;
    valid input is then constrained by the durable record predicates.
- Boundary path/seam: enabled draft claim and email transport composition.
  - Replaced-path behaviors: current pending draft claims/sends only its stored
    body. Disabled configuration preserves that branch; enabled configuration
    adds a transient fragment link after the atomic claim succeeds.
  - Guard-relevant fields: config enablement, safe HTTPS base URL, HMAC secret,
    draft `pending`/blocker/recipient predicates, approved draft recipient
    snapshot, active EOM `lead`/`won` contact, approving employee identity.
  - Caller x input shape: authenticated office approval x pending/sending/sent/
    revoked drafts; only the current pending claim is intentionally changed. A
    repeat intake's newer approved recipient becomes the immutable prefill email
    even if `contacts.email` intentionally remains stale.
- Boundary path/seam: public-onboarding authority versus issuance pause.
  - Replaced-path behaviors: one authority flag previously both made routes
    reachable and chose whether approval minted a bearer. The new optional
    issuance override preserves the former authority decision and isolates the
    latter mint/no-mint decision.
  - Guard-relevant fields: authority flag, optional issuance override, valid
    URL/HMAC tuple, private funnel API flag and bearer, and a durable bearer
    whose token-status/key-binding predicates are still enforced downstream.
  - Caller x input shape: office approval x authority enabled/disabled and
    override unset/false; tracker session/finalize x valid bearer with authority
    enabled and override false; authority-disabled session/finalize x any
    bearer. The disabled-authority case remains rejected before CRM access.
- Boundary path/seam: raw public-onboarding URL admission.
  - Replaced-path behaviors: the original validator rejected C0/DEL characters
    but delegated ordinary whitespace and raw reverse solidi to `urlsplit`,
    whose truthy hostname could admit a value that cannot be emitted as one
    email link. The raw admission boundary now rejects all whitespace and
    reverse solidi before parsing and retains every later structural URL
    predicate.
  - Guard-relevant fields: raw URL characters, normalized empty/nonempty state,
    HTTPS scheme, authority/host, credentials, port, query/fragment, and the
    paired HMAC secret.
  - Caller x input shape: operator configuration x valid URL, blank disabled
    default, ordinary-space host/path, Unicode-whitespace host, reverse-solidus
    authority/path, ASCII control, and malformed URL families. Those raw
    character families reject before draft claim or transport construction.
- Boundary path/seam: public token finalizer and office-handoff fence.
  - Replaced-path behaviors: office handoff currently accepts an active lead in
    `new`, `estimate_booked`, or `won`; it now rejects only while a durable
    issued public token for that contact exists. The matching public token is
    intentionally admitted as the alternate finalizer.
  - Guard-relevant fields: token id/status, contact id, tracker ids,
    deterministic completion key, stored approving actor, and existing handoff
    uniqueness evidence.
  - Caller x input shape: office actor x existing handoff payload; tracker
    service x token plus positive tracker ids; replay with matching vs different
    ids; private office actor x issued/redeemed/missing token revocation even
    after public issuance has been disabled.
- Boundary path/seam: dormant public-token relation readiness.
  - Replaced-path behaviors: a disabled authority previously ignored a present
    token table entirely. A missing table remains a compatible pre-migration
    state, but a present table now must admit the unconditional office fence and
    private recovery command before startup succeeds.
  - Guard-relevant fields: relation existence; `id`, `draft_id`, `contact_id`,
    `status`, and `revoked_at`; current runtime `SELECT` and `UPDATE`
    privileges; authority-enable decision; enabled-only full projection,
    constraints/index, and `INSERT` privilege.
  - Caller x input shape: disabled authority x missing relation, complete
    relation, issuance-only missing column, recovery-column omission, no token
    `SELECT`, SELECT without UPDATE, and SELECT+UPDATE without INSERT; enabled
    authority x incomplete full projection. The missing relation remains
    accepted only for the disabled authority; any present recovery deficiency
    rejects at startup before a later office command reaches database DML.

Closure declaration:

- Raw bearer input is **OPEN**, while the admitted token language is **CLOSED**:
  one version prefix, canonical UUID, and one fixed-length base64url HMAC.
  Membership is **DERIVED** from the documented `eomob1.<UUID>` HMAC message and
  canonical UUID grammar at the new parser choke point; all unrecognized text,
  modified parts, and non-string container shapes reject before a database read
  because bearer admission is asymmetric-security-sensitive. The generated test
  uses an independent HMAC oracle over version/UUID/signature/framing/input-shape
  families, so it proves the grammar rather than a fixture list.
- The public-onboarding configuration tuple is **OPEN** because URL and secret
  values are arbitrary operator-provided text. Its safe membership is
  **DERIVED** at model validation time from HTTPS URL structure, the paired
  URL/secret requirement, absence of C0/DEL, all whitespace, and raw reverse
  solidi, minimum secret byte length, and enabled/API flag relationship; no host
  or URL list is copied into the policy. Every partial, malformed,
  credential-bearing, query/fragment-bearing, whitespace-bearing,
  reverse-solidus-bearing, or disabled-without-safe-pair tuple rejects before
  issuance, which is the safer and cheaper outcome because it preserves the
  existing disabled email path. A generated URL/secret/flag product checks this
  result against an independent configuration contract oracle.
- The issuance override is **CLOSED**: its typed source is `bool | None` on
  `EOMFunnelConfig`. `None` derives the legacy authority-enabled issuance
  behavior; `false` pauses only new issuance; `true` is admitted only when the
  authority is enabled. Model validation rejects the false-authority/
  true-issuance combination before an approval claim. The effective decision is
  derived once in configuration, and every rejected or unconfigured authority
  path remains on the safe no-issue/no-redemption side.
- Token statuses are **CLOSED** and authored by the new migration CHECK plus
  the canonical Python status vocabulary. Out-of-set database values are
  rejected by the schema; unknown application values fail closed rather than
  being treated as issued.
- The existing staff capability manifest remains **UNCHANGED**. Route
  registration alone does not establish that this config-gated private
  authority is safe to call, so this dormant tracker-only surface is not
  advertised as a staff capability. Its future tracker bridge must treat the
  Atlas `503` configuration boundary as unavailable rather than infer
  readiness from an application route existing.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: public-onboarding authority is disabled and
  the optional issuance override is unset; existing approved-draft email
  behavior is the default session behavior.
- Explicit value probe: a valid HTTPS page URL plus high-entropy Atlas-only
  HMAC secret and enabled authority yields the tokenized email and private
  routes when the issuance override is unset or true. Keeping that authority
  enabled while explicitly setting the override false yields private redemption
  routes but no new tokenized email.
- Absent value probe: disabled/blank configuration never mints a token or sends
  a link and rejects public session/finalize access; a partially present,
  insecure, whitespace-bearing, or malformed explicit configuration fails
  startup/route admission before a draft claim.
- Default-session/default-context probe: no actor header is accepted as a
  substitute for service authentication; no service bearer is emitted to a
  response, email, log, draft, or Website source.
- Side-effect ordering: validate transport/config before claim; atomically
  claim draft plus mint durable token; send outside the transaction; atomically
  validate/redeem token plus write the existing handoff/contact transition.
- Dormant-recovery probe: a disabled deployment without migration 383 remains
  ready. If that relation is present, startup requires the exact columns and
  runtime `SELECT`/`UPDATE` used by the private fence/revocation recovery paths;
  an issuance-only field or `INSERT` remains enabled-only. This makes a partial
  migration or separate-DBA privilege omission fail before an office handoff.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_auth.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_onboarding_drafts.py`
- `atlas_brain/services/eom_public_onboarding_tokens.py`
- `atlas_brain/storage/migrations/383_eom_public_onboarding_tokens.sql`
- `plans/PR-EOM-Public-Onboarding-Token-Authority.md`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_public_onboarding.py`
- `tests/test_eom_render_profile.py`

## Mechanism

The new `eom_public_onboarding_tokens` table is the authoritative token state:
one row belongs to one draft and one Atlas contact, starts `issued`, may be
explicitly `revoked`, and records its one immutable Atlas handoff when
`redeemed`. A randomly generated UUID makes the bearer unguessable; the token
formatter signs `eomob1.<id>` with the primary Atlas-only HMAC secret. The row
stores a non-secret fingerprint of that signing key and the immutable prefill
projection admitted at issuance; it stores neither a raw bearer nor a raw
secret. The raw token is regenerated only in memory to build
`https://.../onboarding#token=...` for the approved email transport, then
discarded.

When disabled, the existing `claim_eom_onboarding_draft` query and sender are
unchanged. When enabled, its transaction requires the contact to still be an
active EOM `lead` at `won`, changes the draft to `sending`, records the
approving employee in the token row, and returns the transient link to the
sender. Its immutable `prefill_email` is the claimed draft's approved recipient
rather than a possibly stale contact column; the remaining prefill fields retain
their token-time contact snapshot. The sender appends a fixed onboarding
invitation in memory, confirms the ordinary draft delivery state as today, and
writes redacted history using the stored draft body rather than the
bearer-bearing transport body.

The tracker-only session/finalize routes authenticate with
`require_eom_funnel_api` and the configured public-onboarding authority,
never `require_eom_funnel_actor`. The authority flag continues to control
migration readiness and complete route availability. A separate optional
issuance override is consulted only by `approve-send`: its default retains the
legacy enabled behavior, while `false` stops new bearer minting without
interrupting redemption or private revocation. The session route returns only
the prefill that the tracker needs to construct the public form. The finalizer
takes only the bearer plus tracker Customer/Site identifiers; it derives the
Atlas contact and approval actor from durable token state. Its transaction
reuses the existing sorted advisory locks and handoff uniqueness constraints,
writes the normal contact-to-customer evidence plus
`completion_channel=public_onboarding`, inserts the single handoff, and marks
the token redeemed. The staff-only `revoke-link` command deliberately stays
available if issuance is later paused; it is the recovery path that releases
the existing office fence.

The configuration validator admits the raw public-onboarding base URL before
any URL normalization or email-link construction. It rejects C0/DEL and every
Unicode whitespace character plus every raw reverse solidus, then applies the
existing HTTPS/host/credential/port/query/fragment and paired-secret checks.
The validator rejects rather than rewrites an operator value, so a valid base
URL retains its established delivery semantics and a malformed one cannot be
transported as a fragment link.

The readiness query distinguishes no table from a present table. A missing token
table is valid only when authority is disabled, preserving deploy-before-migrate
compatibility. Once present, it must provide the fields and runtime DML used by
the office fence and private revocation recovery regardless of issuance state.
Only enabled issuance additionally requires the entire token projection,
constraints/index, and `INSERT`; an operator can therefore pause authority only
when its already-deployed recovery surface remains sound.

## Intentional

- The token is opaque, HMAC-signed, one-use, and explicitly revocable; it is
  not a JWT and it does not reuse the tracker QR/JWT secret. Atlas owns the
  authority because it owns lead lifecycle.
- This version does not invent an expiration period. An issued link stays valid
  until completion or an office revokes it, including across one controlled HMAC
  rotation when the prior verifier remains configured. Before a second rotation,
  the office must revoke/reissue or drain all tokens signed by the old prior key.
  Expiry/resend policy is a later product decision.
- The public completion is recorded through the existing office-approved
  handoff evidence using the approving employee and explicit channel metadata.
  It does not add a second contact-to-tracker mapping table or change the
  intentional many-to-one tracker-to-Atlas topology.
- The Atlas PR is deliberately feature-gated and not a live end-to-end rollout:
  tracker and Website callers must land before an operator enables it.
- `public_onboarding_enabled=false` remains a full authority disable, not an
  issuance-pause command. To drain existing issued rows, retain the configured
  authority and set only the optional issuance override false; leaving the
  override unset preserves the behavior of already-configured deployments.

### Rollback safety

The migration is audit-bearing and is never dropped as an application rollback
shortcut. If public onboarding has ever been enabled, the release manager must:

1. While the current Atlas release is still serving, retain
   `ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_ENABLED=true` and set
   `ATLAS_EOM_FUNNEL_PUBLIC_ONBOARDING_ISSUANCE_ENABLED=false`. This pauses new
   token minting while session/finalize and the existing private `revoke-link`
   route remain available to drain already-issued rows.
2. Enumerate every `issued` row through the authorized Atlas data path, invoke
   `revoke-link` for its draft, and retain the revoked row as audit evidence.
3. Verify `SELECT count(*) FROM eom_public_onboarding_tokens WHERE status =
   'issued'` is zero. Do not roll back while any issued row remains.
4. Only then deploy the earlier application revision; retain migration 383 and
   its terminal audit rows. A later re-enable is a new rollout decision, not a
   rollback side effect.

The private-route proof covers recovery while issuance is disabled, and the
disposable-Postgres proof covers revocation releasing the office handoff fence.

## Deferred

- Tracker companion: token-only public session/complete bridge, narrow local
  reservation/retry record, no staff credential exposure, and admin proxy for
  link revocation.
- Website companion: no-login mobile EN/ES page, fragment extraction/cleanup,
  identity read-only display, public data fields, success/invalid/retry states,
  and staff revocation affordance in the existing queue.
- An explicit expiry/resend policy, reminders, multi-site onboarding, anonymous
  edits to CRM identity fields, rate/frequency/scheduling capture, and metrics
  are outside this approved v1.
- Production configuration, synthetic end-to-end lead, real email, and data
  cleanup require a separate operator-approved live rehearsal after all three
  deployable components are available.

Parked hardening: cosmetic copy, extra onboarding fields, automated expiry/
reminder workflows, and observability polish are parked by default. Any finding
that can expose a bearer, bypass service authentication, create a second
handoff, or leave an unrecoverable token/lead state remains in scope because it
blocks the safety of this vertical path.

## Verification

- Passed locally:
  - `pytest -q tests/test_eom_public_onboarding.py` -- 32 passed; its
    independent grammar includes a reverse-solidus authority family, and direct
    authority/path examples reject before URL parsing.
  - `pytest -q tests/test_migrations_runner.py` -- 30 passed, 1 skipped;
    its repository-wide duplicate-prefix policy admits only the established
    historic collisions and rejects the new `382` collision.
  - The exact `Run EOM lead pipeline checks` file list in
    `.github/workflows/atlas_eom_lead_pipeline_checks.yml`, run locally with
    `ATLAS_MIGRATION_TEST_DATABASE_URL` pointed at a fresh disposable
    `postgres:16-alpine` instance -- 1104 passed, 5 skipped.
  - Focused disposable-Postgres recipient/recovery proof -- 4 passed; the full
    `tests/test_eom_lead_conversion_integration.py` suite -- 87 passed. These
    cover the latest-intake recipient snapshot, an issuance-only missing field
    under disabled authority, a missing recovery field, and a non-owner runtime
    that gains disabled readiness only after `SELECT` plus `UPDATE` (not
    `INSERT`) on the present token relation.
  - `python -m compileall -q` over the changed Python modules/test,
    `ruff check` over the same paths, `git diff --check`, and
    `python scripts/check_diff_budget.py --additions 3767 --body-file <PR body>`
    -- passed.
  - `python scripts/check_guard_class_closure.py --base origin/main --strict`
    -- advisory lint passed with no guard-shaped change lacking property proof.
- The disposable `postgres:16-alpine` container was local-only and removed after
  the integration suite. No configured repository formatter/type-check command
  applies to this slice; the existing broad Black rewrite was not run because
  it would create unrelated formatting churn.
- Cold reconstruction is recorded in the PR body with file-and-line citations
  before push. It must report no open contract, scope, or forbidden-touch gap.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 7 |
| `atlas_brain/eom_api/config.py` | 145 |
| `atlas_brain/eom_api/funnel.py` | 218 |
| `atlas_brain/eom_api/funnel_auth.py` | 62 |
| `atlas_brain/eom_api/funnel_store.py` | 116 |
| `atlas_brain/services/crm_provider.py` | 651 |
| `atlas_brain/services/eom_onboarding_drafts.py` | 47 |
| `atlas_brain/services/eom_public_onboarding_tokens.py` | 169 |
| `atlas_brain/storage/migrations/383_eom_public_onboarding_tokens.sql` | 72 |
| `plans/PR-EOM-Public-Onboarding-Token-Authority.md` | 723 |
| `tests/test_eom_lead_conversion_integration.py` | 708 |
| `tests/test_eom_public_onboarding.py` | 908 |
| `tests/test_eom_render_profile.py` | 14 |
| **Total** | **3840** |
