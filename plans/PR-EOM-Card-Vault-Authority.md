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

1. Add Atlas authority to create or reuse a hosted setup session only for an
   eligible residential post-clean candidate with current accepted Terms, and
   mark the enrollment ready only from a verified Stripe event.
2. Expose read-only card readiness and prove configuration, eligibility,
   provider-event, replay, concurrency, migration, and real-entrypoint
   boundaries with focused tests.

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
  - The EOM production ASGI entrypoint mounts both the session/readiness API and
    the root webhook route; settled by the entrypoint reachability test.
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
- `tests/test_eom_first_clean_completion_dba_runner.py`
- `tests/test_eom_funnel_capability_manifest.py`
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
  touched test modules: 410 passed with the real card-vault migration and
  migration-runner concurrency probes enabled against disposable PostgreSQL.
- `uv run --isolated --with-requirements requirements.eom.txt --with pytest
  --with pytest-asyncio python -m pytest tests/test_eom_card_vault.py -q`: 42
  passed under the exact slim EOM dependency set.
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
| `atlas_brain/eom_api/funnel.py` | 15 |
| `atlas_brain/eom_api/funnel_auth.py` | 35 |
| `atlas_brain/main.py` | 4 |
| `atlas_brain/main_eom.py` | 3 |
| `atlas_brain/services/eom_card_vault.py` | 1160 |
| `atlas_brain/storage/migrations/398_eom_card_vault.sql` | 439 |
| `atlas_brain/storage/migrations/__init__.py` | 1 |
| `ops` | 5 |
| `plans/PR-EOM-Card-Vault-Authority.md` | 243 |
| `render.eom.yaml` | 21 |
| `requirements.eom.txt` | 1 |
| `scripts/apply_eom_card_vault_schema.py` | 37 |
| `scripts/apply_eom_first_clean_completion_schema.py` | 12 |
| `tests/test_agent_operations_contract.py` | 27 |
| `tests/test_eom_card_vault.py` | 961 |
| `tests/test_eom_first_clean_completion_dba_runner.py` | 91 |
| `tests/test_eom_funnel_capability_manifest.py` | 17 |
| `tests/test_eom_render_profile.py` | 27 |
| `tests/test_eom_terms_acceptance.py` | 90 |
| `tests/test_migrations_runner.py` | 3 |
| **Total** | **3562** |
