# PR-EOM-Card-Vault-Authority

## Why this slice exists

Issue #2156 requires residential post-clean onboarding to distinguish a current
Terms acceptance from a completed card-on-file step. Atlas currently owns the
former but has no durable, provider-confirmed card-vault state. This slice adds
that provider authority before any Tracker or Website consumer is allowed to
depend on it. The diff exceeds the normal budget because the schema, signed
provider confirmation, authenticated API, real entrypoint/deployment wiring,
controlled migration, and negative boundary proof are one authority boundary;
splitting them would merge either unreachable state or an unsafe partial path.

### Problem-derived contract

- Root cause: Atlas stops residential post-clean onboarding at Terms acceptance.
  It stores neither a card-vault enrollment nor provider-confirmed payment-method
  state, so it cannot authoritatively answer whether the required card-on-file
  step is ready. A redirect alone would not fix this because browser return
  URLs are not proof that Stripe saved a payment method.
- Correct fix must touch/change: add an EOM-owned durable enrollment schema;
  typed, fail-closed Stripe configuration; a token-gated residential session
  creator using Stripe-hosted Checkout in setup mode; a separately signed EOM
  webhook that confirms the SetupIntent and payment method before advancing
  state; a read-only readiness projection; real ASGI entrypoint wiring;
  controlled migration plumbing; and focused tests for eligibility,
  idempotency, signature rejection, replay, configuration boundaries, and
  entrypoint reachability.
- Must not change: Terms text, Terms acceptance semantics or documents;
  first-clean receipt/candidate creation; contact/customer classification;
  invoices, receivables, charges, prices, payroll, or accounting; Website or
  timetracker code; commercial onboarding behavior; customer-visible copy or
  UI; existing Stripe SaaS billing routes/webhooks; and the trusted-customer
  exception policy.

## Scope (this PR)

Ownership lane: eom-onboarding-card-vault
Slice phase: Vertical slice
Max files: 27

1. Add Atlas authority to create or reuse a hosted setup session only for an
   eligible residential post-clean candidate with current accepted Terms, and
   mark the enrollment ready only from a verified Stripe event.
2. Expose read-only card readiness and prove configuration, eligibility,
   provider-event, replay, concurrency, migration, and real-entrypoint
   boundaries with focused tests.
3. Keep capability discovery compatible with both eagerly flattened and lazy
   FastAPI router includes under the dependency version used by CI.

### Review Contract

- Acceptance criteria:
  - The database allows one durable enrollment per candidate/contact, preserves
    the initiating acceptance as audit evidence, records the current accepted
    Terms on each hosted-session attempt, and enforces coherent pending/ready
    provider identifiers; settled by the migration contract tests.
  - Session creation rejects disabled/missing configuration, invalid or expired
    public tokens, non-residential/inactive contacts, missing current Terms,
    blocked or absent candidates, and any request after the persisted subject
    relationships drift; settled by service/API negative tests.
  - Repeated or concurrent creation returns one logical enrollment and does not
    intentionally create multiple Stripe customers or Checkout sessions;
    settled by idempotency keys, row locking, and replay/concurrency tests.
    The admitted execution model is any number of requests for the same current
    candidate/acceptance, including cancellation or process loss at every await.
    PostgreSQL serializes reservation with the invitation row lock, unique
    candidate/contact enrollment constraints, and a locked latest-session read;
    it commits stable enrollment/session UUIDs before releasing those locks and
    starting provider I/O. Multiple callers may consequently issue the same
    Stripe request, but each uses identical parameters and an idempotency key
    derived only from the committed UUID. Stripe must return the same Customer
    or Checkout Session for that key (and reject parameter drift), while the
    compare-and-store database writes accept only that same provider identity.
    A provider success followed by cancellation or database failure therefore
    leaves a reusable `creating` reservation; a retry repeats the same key and
    converges on the same logical provider object. This is logical exactly-once
    state, not a claim that only one outbound HTTP attempt occurs. The concurrent
    request, provider-failure, and post-provider database-failure tests exercise
    those interleavings and pin the stable keys and monotonic stored identity.
  - Checkout uses `mode=setup` and never creates a PaymentIntent or charge;
    settled by the Stripe-call contract assertion.
  - The webhook reads the raw request, rejects missing/invalid/oversized
    signatures, ignores unrelated events, validates Checkout/SetupIntent
    ownership, and only then stores the provider payment-method identifier;
    settled by webhook boundary tests.
  - Duplicate successful events are idempotent and cannot regress ready state;
    settled by replay tests and monotonic SQL update conditions.
  - Readiness reports card required only for residential contacts and ready only
    from provider-confirmed state; settled by projection tests.
  - The slim EOM and full Atlas production ASGI entrypoints mount the expected
    session/readiness and root webhook routes and explicitly bind their canonical
    pool factories; settled by entrypoint reachability tests that drive
    provider-confirmed state through each composition.
  - The controlled migration has apply/verify tooling and is admitted to the EOM
    CI contract; settled by focused migration-runner tests.
- Reachability proof: `atlas_brain.main_eom:app` routes an authenticated Tracker
  relay carrying a public Terms token to hosted session creation and
  `/webhooks/eom-card-vault` to signed provider confirmation; the observable
  effect is a persisted ready enrollment
    returned by the private readiness route. The test replaces Stripe's outer
    transport and the database adapter while exercising the real ASGI
    application, router, and card-vault service; a separate PostgreSQL test
    executes the guarded migration and attestation boundary.
- Affected surfaces: EOM funnel configuration and router, new EOM card-vault
  service/schema, EOM and full Atlas ASGI composition, migration operations,
  capability manifest, EOM focused CI selection, and focused tests.
- Risk areas: auth/token substitution, stale acceptance/candidate joins,
  provider/DB retry gaps, duplicate events, concurrent creation, signature
  verification, secret exposure, config defaults, webhook misrouting, and
  accidental coupling to the existing SaaS Stripe webhook.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R8, R10, R11, R12, R13, R14.

### Fix-loop disposition preflight

- Root decision: Specify the concurrency execution model
- Source trace: `plans/PR-EOM-Card-Vault-Authority.md:60` claimed one logical
  provider operation -> `atlas_brain/services/eom_card_vault.py:719` releases
  database locks before provider I/O -> the plan must state the admitted
  interleavings and Stripe/database convergence invariant.
- Upstream files: `plans/PR-EOM-Card-Vault-Authority.md`.
- Fix strategy: upstream-root
- Blocking predicate: claimed-mechanism
- Disposition: fixed-in
- Allowed files: `plans/PR-EOM-Card-Vault-Authority.md`.
- Max files: 27
- Parked hardening: none

- Root decision: Log card-vault confirmation failures
- Source trace: `atlas_brain/services/eom_card_vault.py:1051` translated database
  exceptions without operational evidence -> the service boundary must emit
  structured operation and non-secret subject identifiers before translation.
- Upstream files: `atlas_brain/services/eom_card_vault.py`,
  `tests/test_eom_card_vault.py`.
- Fix strategy: upstream-root
- Blocking predicate: claimed-mechanism
- Disposition: fixed-in
- Allowed files: `atlas_brain/services/eom_card_vault.py`, `tests/test_eom_card_vault.py`, `plans/PR-EOM-Card-Vault-Authority.md`.
- Max files: 27
- Parked hardening: none

- Root decision: Validate the created Stripe customer subject
- Source trace: `atlas_brain/services/eom_card_vault.py:743` creates a subject-bound
  Customer -> `atlas_brain/services/eom_card_vault.py:754` previously accepted
  only its identifier -> returned source/enrollment/contact/candidate metadata
  must match before persistence.
- Upstream files: `atlas_brain/services/eom_card_vault.py`,
  `tests/test_eom_card_vault.py`.
- Fix strategy: upstream-root
- Blocking predicate: security
- Disposition: fixed-in
- Allowed files: `atlas_brain/services/eom_card_vault.py`, `tests/test_eom_card_vault.py`, `plans/PR-EOM-Card-Vault-Authority.md`.
- Max files: 27
- Parked hardening: none

- Root decision: Make card-vault preflight attest the deployed schema
- Source trace: `scripts/apply_eom_card_vault_schema.py:37` delegates to the shared
  controlled runner -> `scripts/apply_eom_first_clean_completion_schema.py:433`
  previously checked only migration bookkeeping -> migration 398 must also run
  the runtime-role card-vault schema attestation and reject either drift direction.
- Upstream files: `scripts/apply_eom_first_clean_completion_schema.py`,
  `tests/test_eom_first_clean_completion_dba_runner.py`.
- Fix strategy: upstream-root
- Blocking predicate: claimed-mechanism
- Disposition: fixed-in
- Allowed files: `.agent/capabilities.yaml`, `.agent/runbooks/database.md`, `atlas_brain/services/eom_card_vault.py`, `atlas_brain/storage/migrations/398_eom_card_vault.sql`, `atlas_brain/storage/migrations/__init__.py`, `ops`, `plans/PR-EOM-Card-Vault-Authority.md`, `scripts/apply_eom_card_vault_schema.py`, `scripts/apply_eom_first_clean_completion_schema.py`, `tests/test_agent_operations_contract.py`, `tests/test_eom_card_vault.py`, `tests/test_eom_first_clean_completion_dba_runner.py`, `tests/test_eom_terms_acceptance.py`, `tests/test_migrations_runner.py`.
- Max files: 27
- Parked hardening: none

- Root decision: Exercise the full Atlas webhook entrypoint
- Source trace: `atlas_brain/main.py:1224` mounts the root webhook ->
  `tests/test_eom_card_vault.py:1053` previously exercised only `main_eom.app` ->
  `atlas_brain/main.py:1188` now binds the canonical pool factory and a signed
  request traverses `main.app` through that explicit composition seam.
- Upstream files: `atlas_brain/main.py`, `tests/test_eom_card_vault.py`.
- Fix strategy: upstream-root
- Blocking predicate: claimed-mechanism
- Disposition: fixed-in
- Allowed files: `.github/workflows/atlas_eom_lead_pipeline_checks.yml`, `atlas_brain/eom_api/card_vault.py`, `atlas_brain/eom_api/config.py`, `atlas_brain/eom_api/funnel.py`, `atlas_brain/eom_api/funnel_auth.py`, `atlas_brain/main.py`, `atlas_brain/main_eom.py`, `plans/PR-EOM-Card-Vault-Authority.md`, `render.eom.yaml`, `requirements.eom.txt`, `tests/test_eom_card_vault.py`, `tests/test_eom_first_clean_completion.py`, `tests/test_eom_funnel_capability_manifest.py`, `tests/test_eom_missed_call_recovery.py`, `tests/test_eom_render_profile.py`.
- Max files: 27
- Parked hardening: none

### Guard class-closure declaration

- Stripe credential and provider-object inputs are OPEN because environment
  text and Stripe response/event containers are producer supplied. Membership
  is DERIVED at each use from the documented key/identifier grammar, exact
  provider source and subject metadata, and the expected scalar/mapping/object
  representation. The semantic-oracle tests generate identifier families x
  token classes x container shapes. Any malformed, unknown, or mismatched value
  fails closed before a provider or readiness effect; a validly signed but
  unrelated event is acknowledged without mutation, which is the safe side.
- Eligibility and readiness fields are CLOSED and DERIVED from the named SQL
  projections over the canonical contacts, candidate, invitation, acceptance,
  and Terms-version schema. A missing or unrecognized status/audience/identity
  takes the not-found or unavailable path before Stripe is called.
- Runtime privileges and guard triggers are CLOSED and ENUMERATED from the exact
  columns and writes used by this service and migration 398. Missing required
  membership or any extra INSERT/UPDATE/DELETE/TRUNCATE authority makes schema
  attestation fail, which is safer than serving against a drifted boundary.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: typed environment -> enabled provider dependency; public
  token -> candidate/contact/current acceptance; provider event -> enrollment
  transition; private service bearer -> readiness projection.
- Replaced-path behaviors: none. These are additive routes and state. Existing
  Terms and billing paths remain authoritative for their own contracts.
- Guard-relevant fields: enabled flag, Stripe secret/webhook secret, public
  onboarding URL, token invitation/nonce/contact, candidate presence/status,
  contact status/customer type/business context, acceptance version/material
  fingerprint, Checkout mode/customer/session metadata, SetupIntent status,
  customer/payment method, Stripe event identifier and signature.
- Caller x input shape: authenticated Tracker relay x
  valid/invalid/expired/substituted token; Stripe x
  valid/invalid/duplicate/unrelated/mismatched event; service caller x
  residential/commercial/absent contact or candidate.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: card vault is disabled by default; no Stripe
  key or webhook secret is inferred from the SaaS billing configuration.
- Explicit value probe: enabled plus valid dedicated Stripe values and public
  onboarding URL constructs the provider dependency and routes successfully.
- Absent value probe: enabled with either secret or public onboarding disabled
  fails configuration validation before serving card-vault behavior.
- Default-session/default-context probe: default/unset configuration leaves all
  existing applications bootable while the card-vault routes fail closed and
  produce no provider or database side effect.
- Side-effect ordering: eligibility and current Terms are locked and validated,
  and stable operation identifiers are committed before provider creation;
  provider confirmation is validated before the monotonic ready update;
  unrelated/invalid events write nothing.

### Files touched

- `.agent/capabilities.yaml`
- `.agent/runbooks/database.md`
- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `atlas_brain/eom_api/card_vault.py`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_auth.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/eom_card_vault.py`
- `atlas_brain/storage/migrations/398_eom_card_vault.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `ops`
- `plans/PR-EOM-Card-Vault-Authority.md`
- `render.eom.yaml`
- `requirements.eom.txt`
- `scripts/apply_eom_card_vault_schema.py`
- `scripts/apply_eom_first_clean_completion_schema.py`
- `tests/test_agent_operations_contract.py`
- `tests/test_eom_card_vault.py`
- `tests/test_eom_first_clean_completion.py`
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_missed_call_recovery.py`
- `tests/test_eom_render_profile.py`
- `tests/test_eom_terms_acceptance.py`
- `tests/test_migrations_runner.py`

## Mechanism

Atlas persists one enrollment keyed to the post-clean candidate and contact; its
initial acceptance is immutable audit evidence rather than the lifetime of the
saved card. Each hosted-session attempt records the then-current Terms
acceptance. The public route authenticates the existing Terms token, revalidates
all persisted relationships under a transaction, and commits the enrollment and
stable provider-operation identifiers before making network calls. Those
identifiers become Stripe idempotency keys for a hosted Checkout Session in
setup mode. The browser receives only the hosted URL; the return URL never marks
readiness. A dedicated EOM webhook verifies Stripe's signature over the raw body,
validates the completed Checkout Session and retrieved SetupIntent against the
enrollment, records the event, and advances the row monotonically. A private read
model derives whether a card is required and provider-confirmed independently of
later Terms reacceptance.

## Intentional

- Use Stripe-hosted Checkout rather than collecting card fields in EOM systems.
- Use a dedicated EOM webhook secret and route rather than overloading the SaaS
  billing webhook, because the EOM slim deployment has a distinct database and
  entrypoint.
- Treat the database row as durable operation authority and Stripe idempotency
  as retry protection, not as permanent deduplication.
- Do not treat the success redirect as proof; only a signed event plus retrieved
  SetupIntent can make readiness true.

## Deferred

- The audited trusted-customer/no-card exception is a subsequent policy slice.
- Tracker proxy and Website/customer UI consumers follow after this provider
  contract is merged and deployed.
- Customer communication copy remains a later product slice and English-only.

Parked hardening: none.

## Verification

- `uv run --with stripe==15.3.0 python -m pytest` over the seven directly
  touched test modules: 412 passed with the real card-vault migration and
  migration-runner concurrency probes enabled against disposable PostgreSQL.
- `uv run --isolated --with-requirements requirements.eom.txt --with pytest
  --with pytest-asyncio python -m pytest tests/test_eom_card_vault.py -q`: 44
  passed under the exact slim EOM dependency set.
- `uv run --isolated --python 3.11 --with-requirements requirements.txt --with
  pytest --with pytest-asyncio python -m pytest -q` over the card-vault,
  migration-preflight, capability-manifest, and three affected legacy route
  tests: 99 passed under the dependency profile that exposed the CI failure.
- `python -m ruff check` over all eight changed Python files: passed.
- The full-Atlas signed-webhook entrypoint test passes through the explicit
  production pool-factory composition seam; the exact storage maturity-sweep
  ratchet command passes without changing its baseline.
- Ruff import/undefined-name checks passed for the new files; fatal Python
  checks passed across every touched Python file; all four new Python files are
  formatter-clean; `py_compile` passed across every touched Python file.
- Cold diff audit: every production, schema, deployment, operations, and test
  change traces to the Problem-derived contract; no required item is absent and
  no declared non-scope surface changed.
- Pending in the publish wrapper: synchronized plan/body, diff-budget override,
  whitespace, session-drift, and repository mechanical audits.

## Estimated diff size

| File | LOC |
|---|---:|
| `.agent/capabilities.yaml` | 11 |
| `.agent/runbooks/database.md` | 32 |
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 15 |
| `atlas_brain/eom_api/card_vault.py` | 241 |
| `atlas_brain/eom_api/config.py` | 71 |
| `atlas_brain/eom_api/funnel.py` | 54 |
| `atlas_brain/eom_api/funnel_auth.py` | 35 |
| `atlas_brain/main.py` | 5 |
| `atlas_brain/main_eom.py` | 3 |
| `atlas_brain/services/eom_card_vault.py` | 1211 |
| `atlas_brain/storage/migrations/398_eom_card_vault.sql` | 439 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 5 |
| `plans/PR-EOM-Card-Vault-Authority.md` | 364 |
| `render.eom.yaml` | 21 |
| `requirements.eom.txt` | 1 |
| `scripts/apply_eom_card_vault_schema.py` | 37 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 31 |
| `tests/test_agent_operations_contract.py` | 27 |
| `tests/test_eom_card_vault.py` | 1214 |
| `tests/test_eom_first_clean_completion.py` | 4 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 190 |
| `tests/test_eom_funnel_capability_manifest.py` | 44 |
| `tests/test_eom_missed_call_recovery.py` | 2 |
| `tests/test_eom_render_profile.py` | 27 |
| `tests/test_eom_terms_acceptance.py` | 90 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **4178** |
