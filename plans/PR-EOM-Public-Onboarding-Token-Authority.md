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

## Scope (this PR)

Ownership lane: eom-public-onboarding-token-authority
Slice phase: vertical slice

1. Establish dormant Atlas token issuance, validation, explicit revocation, and
   one-time finalization authority for one active `won` EOM lead.
2. Preserve one durable `eom_customer_handoffs` record for either completion
   channel and make the public-token channel wait/fail safely rather than race
   the existing office channel.
3. Publish tracker-only service endpoints, not browser endpoints, with a
   narrow prefill response and no generic CRM mutation surface.
4. Prove the new surface through the router and a disposable PostgreSQL
   transaction suite before a tracker or Website caller is introduced.

### Review Contract

- Acceptance criteria:
  1. With public onboarding disabled, `POST .../approve-send` follows its
     current claim/send/confirm path and submits the stored draft body unchanged;
     the focused HTTP approval regression test settles this behavior. The open
     configuration tuple is checked over a generated URL/secret/flag grammar
     against an independent contract oracle, so unsafe or incomplete values
     cannot accidentally take the enabled branch.
  2. With valid explicit public configuration, one approved, active `won` lead
     gets exactly one opaque HMAC token during the same transaction that claims
     its draft. The email transport sees an HTTPS URL whose token is in the
     fragment, while the persisted draft and email-history payload contain no
     bearer; focused service tests settle this behavior.
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
     responses omit Atlas contact and handoff identifiers.
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
     migration-readiness tests remain green; this proves the staff paths and
     previously advertised API shapes retain their contracts.
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
    draft `pending`/blocker/recipient predicates, active EOM `lead`/`won`
    contact, approving employee identity.
  - Caller x input shape: authenticated office approval x pending/sending/sent/
    revoked drafts; only the current pending claim is intentionally changed.
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
  URL/secret requirement, minimum secret byte length, and enabled/API flag
  relationship; no host or URL list is copied into the policy. Every partial,
  malformed, credential-bearing, query/fragment-bearing, or disabled-without-
  safe-pair tuple rejects before issuance, which is the safer and cheaper
  outcome because it preserves the existing disabled email path. A generated
  URL/secret/flag product checks this result against an independent configuration
  contract oracle.
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

- Deployed/default config values: public onboarding is disabled by default;
  existing approved-draft email behavior is the default session behavior.
- Explicit value probe: a valid HTTPS page URL plus high-entropy Atlas-only
  HMAC secret and enabled flag yields the tokenized email and private routes.
- Absent value probe: disabled/blank configuration never mints a token or sends
  a link; a partially present, insecure, or malformed explicit configuration
  fails startup/route admission before a draft claim.
- Default-session/default-context probe: no actor header is accepted as a
  substitute for service authentication; no service bearer is emitted to a
  response, email, log, draft, or Website source.
- Side-effect ordering: validate transport/config before claim; atomically
  claim draft plus mint durable token; send outside the transaction; atomically
  validate/redeem token plus write the existing handoff/contact transition.

### Files touched

- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_auth.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/services/crm_provider.py`
- `atlas_brain/services/eom_onboarding_drafts.py`
- `atlas_brain/services/eom_public_onboarding_tokens.py`
- `atlas_brain/storage/migrations/382_eom_public_onboarding_tokens.sql`
- `plans/PR-EOM-Public-Onboarding-Token-Authority.md`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_eom_public_onboarding.py`
- `tests/test_eom_render_profile.py`

## Mechanism

The new `eom_public_onboarding_tokens` table is the authoritative token state:
one row belongs to one draft and one Atlas contact, starts `issued`, may be
explicitly `revoked`, and records its one immutable Atlas handoff when
`redeemed`. A randomly generated UUID makes the bearer unguessable; the token
formatter signs `eomob1.<id>` with the Atlas-only HMAC secret. The raw token is
regenerated only in memory to build `https://.../onboarding#token=...` for the
approved email transport, then discarded.

When disabled, the existing `claim_eom_onboarding_draft` query and sender are
unchanged. When enabled, its transaction requires the contact to still be an
active EOM `lead` at `won`, changes the draft to `sending`, records the
approving employee in the token row, and returns the transient link to the
sender. The sender appends a fixed onboarding invitation in memory, confirms
the ordinary draft delivery state as today, and writes redacted history using
the stored draft body rather than the bearer-bearing transport body.

The tracker-only session/finalize routes authenticate with
`require_eom_funnel_api` and a separate public-onboarding-enabled dependency,
never `require_eom_funnel_actor`. The session route returns only the prefill
that the tracker needs to construct the public form. The finalizer takes only
the bearer plus tracker Customer/Site identifiers; it derives the Atlas contact
and approval actor from durable token state. Its transaction reuses the existing
sorted advisory locks and handoff uniqueness constraints, writes the normal
contact-to-customer evidence plus `completion_channel=public_onboarding`,
inserts the single handoff, and marks the token redeemed. The staff-only
`revoke-link` command deliberately stays available if issuance is later
disabled; it is the recovery path that releases the existing office fence.

## Intentional

- The token is opaque, HMAC-signed, one-use, and explicitly revocable; it is
  not a JWT and it does not reuse the tracker QR/JWT secret. Atlas owns the
  authority because it owns lead lifecycle.
- This version does not invent an expiration period. An issued link stays valid
  until completion or an office revokes it, which avoids an unannounced customer
  deadline. Expiry/resend policy is a later product decision.
- The public completion is recorded through the existing office-approved
  handoff evidence using the approving employee and explicit channel metadata.
  It does not add a second contact-to-tracker mapping table or change the
  intentional many-to-one tracker-to-Atlas topology.
- The Atlas PR is deliberately feature-gated and not a live end-to-end rollout:
  tracker and Website callers must land before an operator enables it.

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
  - `pytest -q tests/test_eom_public_onboarding.py tests/test_eom_lead_conversion.py tests/test_eom_funnel_capability_manifest.py tests/test_eom_render_profile.py tests/test_eom_link_verification.py tests/test_eom_billing_recipients.py tests/test_eom_payment_receipts.py`
  - `ATLAS_MIGRATION_TEST_DATABASE_URL=<disposable local PostgreSQL> pytest -q tests/test_eom_lead_conversion_integration.py -rs`
  - `python -m compileall -q` over every changed Python module/test,
    `ruff check` over the same paths, and `git diff --check`.
- The disposable `postgres:16-alpine` container was local-only and removed after
  the integration suite. No configured repository formatter/type-check command
  applies to this slice; the existing broad Black rewrite was not run because
  it would create unrelated formatting churn.
- Cold reconstruction is recorded in the PR body with file-and-line citations
  before push. It must report no open contract, scope, or forbidden-touch gap.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/config.py` | 80 |
| `atlas_brain/eom_api/funnel.py` | 208 |
| `atlas_brain/eom_api/funnel_auth.py` | 47 |
| `atlas_brain/eom_api/funnel_store.py` | 80 |
| `atlas_brain/services/crm_provider.py` | 627 |
| `atlas_brain/services/eom_onboarding_drafts.py` | 47 |
| `atlas_brain/services/eom_public_onboarding_tokens.py` | 119 |
| `atlas_brain/storage/migrations/382_eom_public_onboarding_tokens.sql` | 62 |
| `plans/PR-EOM-Public-Onboarding-Token-Authority.md` | 345 |
| `tests/test_eom_lead_conversion_integration.py` | 468 |
| `tests/test_eom_public_onboarding.py` | 630 |
| `tests/test_eom_render_profile.py` | 9 |
| **Total** | **2722** |
