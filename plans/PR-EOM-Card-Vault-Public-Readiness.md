# PR-EOM-Card-Vault-Public-Readiness

## Why this slice exists

The public onboarding page can prove that a Terms invitation was accepted, but
that response does not carry the operator-owned one-time/recurring decision.
Atlas's existing public card-session route is also the wrong read boundary: it
requires enabled Stripe authority and admits only a currently eligible recurring
customer. Consequently a downstream UI cannot distinguish "card setup is not
required" from "card setup is required but blocked/unavailable" without either
guessing policy or attempting a provider mutation.

This slice exceeds the 400-LOC target because the public boundary is not safe to
land independently of its closure proof. The token/subject guard, shared policy
projection, exact capability advertisement, and negative, provider-disabled,
and real-database tests must ship together: splitting the route from those
checks would advertise an incompletely proven customer-data boundary, while
splitting the tests alone would not deliver a reachable behavior. Most of the
overage is the plan and boundary-test evidence; the production change remains
one read-only route, one shared projector, and one capability entry.

### Problem-derived contract

- Root cause: Atlas owns the accepted Terms evidence, first-clean candidate,
  service-commitment decision, and provider-confirmed enrollment, but exposes no
  token-bound, side-effect-free public projection joining those authorities.
  The only token-bound card endpoint begins Stripe session issuance, while the
  existing read-only readiness route is keyed by an internal contact UUID.
- Correct fix must touch/change: add one service-authenticated, Terms-token-bound
  Atlas read route; authenticate the existing opaque bearer before one
  side-effect-free database projection; derive its result from the same
  card-readiness decision used by the internal contact route; return only the
  closed customer-safe `cardRequired`, `cardReady`, and `reason` fields; advertise
  the exact route in the derived capability manifest; and add focused service,
  route, negative-boundary, no-provider, no-write, and capability-contract proof.
- Must not change: database schema or migrations; the one-time/recurring policy;
  Terms content, invitation issuance, acceptance, receipt, or delivery; Stripe
  customer/session creation, webhook verification, enrollment state, return URLs,
  or provider configuration; internal readiness response fields; first-clean,
  calendar, CRM, onboarding-email, payroll, billing, or payment behavior; public
  copy; dependencies; or any non-EOM module.

## Scope (this PR)

Ownership lane: eom/card-vault-public-readiness
Slice phase: Vertical slice
Max files: 7

1. Expose a minimal public card-readiness projection behind the existing EOM
   service token and opaque Terms bearer, with no Stripe/config/write dependency.
2. Pin the policy states, token/subject failure boundary, response privacy, exact
   capability signature, and unchanged session/internal-readiness behavior.

### Review Contract

- Acceptance criteria:
  - [ ] `POST /api/v1/eom-funnel/card-vault/public/readiness` accepts the existing
    service bearer plus a correctly signed Terms token and returns exactly
    `cardRequired`, `cardReady`, and a canonical `reason`; settled by the real
    ASGI-entrypoint test in `tests/test_eom_card_vault.py`.
  - [ ] Commercial and one-time residential subjects return
    `false/true/not_required`; recurring subjects return the existing
    `terms_not_ready`, `first_clean_not_confirmed`, `not_started`, `pending`, or
    `ready` state; an undecided residential subject returns
    `true/false/service_commitment_required`; settled by the parametrized service
    projection test.
  - [ ] A malformed bearer, mismatched signing-key fingerprint, revoked/expired
    invitation, or drifted contact/invitation subject returns the existing
    not-found boundary before any card-readiness data is returned; settled by
    route authentication and service subject-boundary tests.
  - [ ] The public readiness route still succeeds when card issuance is disabled
    and Stripe secrets are absent, and its dependency path performs only schema
    attestation plus one `SELECT`; settled by the disabled-provider ASGI test and
    the read-only pool spy.
  - [ ] The internal contact readiness response and the public session/webhook
    contracts are unchanged; settled by the pre-existing focused card-vault test
    file and hosted unit gate.
  - [ ] `card_vault.public.readiness` is advertised only when the exact POST route
    is registered; settled by `tests/test_eom_funnel_capability_manifest.py`.
- Reachability proof: the real `main_eom.app` ASGI entrypoint receives the
  authenticated POST and its JSON response is asserted with card issuance
  disabled and no Stripe credentials.
- Affected surfaces: EOM card-vault API/service, the EOM funnel capability map,
  and their focused contract tests.
- Risk areas: token authorization and subject isolation, customer-visible state
  semantics, provider/config coupling, response-data exposure, accidental writes,
  API compatibility, and capability over-advertising.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R12, R14.

### Closure Declaration

- Public token input space: **OPEN** (all JSON values). Membership is **DERIVED**
  at every call from the existing `authenticate_eom_terms_token` grammar and
  HMAC authority, followed by a database subject match. Every unrecognized,
  malformed, wrong-key, revoked, expired, or subject-drifted input takes the
  safer existing 404 boundary before projection. The existing grammar test plus
  new route and subject-boundary tests provide the semantic and representation
  probes; this PR does not add a token denylist.
- Readiness reasons: **CLOSED**. Membership is **DERIVED** by both response models
  from one canonical Atlas `Literal` alias whose members are the existing
  internal card-only readiness decision. Any new or malformed service reason is
  rejected by response validation rather than silently exposed.
- Public response fields: **CLOSED** and authored here as the minimal customer
  contract. Model validation forbids unknown input fields and the service
  constructs only the three named fields; internal IDs, audience, timestamps,
  and provider identifiers stay outside the projection.
- Capability membership: **CLOSED** over registered method/path signatures and
  **DERIVED** by `served_capabilities()` from the canonical route map plus the
  live router. An unregistered signature is omitted, which is the safe
  under-advertising direction.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: new service-authenticated public readiness POST -> existing
  Terms bearer authentication -> invitation/contact subject guard -> shared
  read-only readiness projection.
- Replaced-path behaviors: none; the contact-ID readiness GET and provider-writing
  public session POST remain separate and unchanged.
- Guard-relevant fields: token grammar/signature/key fingerprint; invitation
  revocation/expiry/audience/name/email; contact business context/type/status/
  customer type/name/email; Terms acceptance/audience/material-version state;
  pending first-clean candidate; service commitment; enrollment status.
- Caller x input shape: authenticated Tracker + valid recurring/one-time/
  commercial/undecided token objects are admitted to their exact projection;
  missing/wrong service bearer is rejected by the router; malformed/forged raw
  token values are rejected by the existing token parser; revoked/expired or
  drifted stored subjects are rejected by the new subject choke point.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: card issuance defaults disabled. Current live
  inspection before this slice found public onboarding authority configured but
  the card-vault enable flag and dedicated Stripe credentials unset.
- Explicit value probe: focused ASGI proof uses enabled public onboarding and a
  valid service bearer while card issuance and Stripe authority are absent.
- Absent value probe: missing/invalid service or Terms authority remains rejected
  by existing dependencies/authentication; card/provider authority is deliberately
  absent and does not gate this read.
- Default-session/default-context probe: the real `main_eom.app` route is used;
  no direct function-only proof substitutes for reachability.
- Side-effect ordering: service-token and Terms-token authentication precede the
  one projection; the route has no provider dependency, and the pool spy records
  any transaction attempt so the no-write assertion can fail.

### Files touched

- `atlas_brain/eom_api/card_vault.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/services/eom_card_vault.py`
- `plans/PR-EOM-Card-Vault-Public-Readiness.md`
- `tests/test_eom_card_vault.py`
- `tests/test_eom_funnel_capability_manifest.py`
- `tests/test_eom_terms_acceptance.py`

## Mechanism

The API authenticates the existing opaque Terms bearer with public-onboarding
authority and passes only the authenticated token object to a read-only service.
The service fetches the invitation-bound contact and all readiness inputs in one
snapshot query, rejects a revoked, expired, or drifted bearer subject, and sends that row
through the same pure readiness projector as the internal contact-ID route. A
separate response model strips internal/provider evidence and pins the public
three-field contract. The capability remains derived from the registered router.

## Intentional

- Keep readiness separate from session creation so a status read cannot create a
  Stripe customer, enrollment, or hosted session.
- Reuse the existing Terms bearer instead of minting another public credential.
- Fail closed on undecided residential service rather than guessing recurring;
  treat one-time and commercial as card-not-required exactly as current policy.
- Preserve the internal response instead of weakening it to the public shape.

## Deferred

- Tracker relay/capability proof and Website customer/staff UI are the next serial
  PRs after this provider contract is merged and deployed.
- Real Stripe redirect/webhook proof remains blocked until the operator provisions
  the dedicated card-vault enable flag, Stripe secret, and webhook secret.

Parking predicate: this provider slice parks findings that require the Tracker
relay, Website customer/staff UI or copy, live Stripe session/webhook execution,
provider credential provisioning, or unrelated onboarding surfaces unless they
show that Atlas's readiness read itself is unauthenticated, misclassified,
writable, overexposed, or unreachable.

Parked hardening: none against that predicate.

## Verification

- `./ops test focused tests/test_eom_card_vault.py tests/test_eom_funnel_capability_manifest.py`
  -> `97 passed in 4.30s`.
- `ATLAS_EOM_FIRST_CLEAN_TEST_DATABASE_URL=<redacted-disposable-runtime-url>
  ATLAS_EOM_FIRST_CLEAN_DBA_DATABASE_URL=<redacted-disposable-dba-url> python -m
  pytest -q
  tests/test_eom_terms_acceptance.py::test_public_card_readiness_executes_against_the_guarded_schema`
  -> `1 passed in 1.01s`. The credentials are intentionally redacted; the
  disposable PostgreSQL 16 instance used separate runtime and DBA roles,
  migrations 395 through 399, and was removed after the test.
- `ruff check --target-version py312 atlas_brain/eom_api/card_vault.py
  atlas_brain/eom_api/funnel.py atlas_brain/services/eom_card_vault.py
  tests/test_eom_card_vault.py tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_terms_acceptance.py` -> exit 0.
- `python -m compileall -q atlas_brain/eom_api/card_vault.py
  atlas_brain/eom_api/funnel.py atlas_brain/services/eom_card_vault.py
  tests/test_eom_card_vault.py tests/test_eom_funnel_capability_manifest.py
  tests/test_eom_terms_acceptance.py` -> exit 0.
- `python scripts/check_guard_class_closure.py --strict --base origin/main` ->
  `OK: no guard-shaped change without a property test`.
- `python scripts/sync_pr_plan.py --check
  plans/PR-EOM-Card-Vault-Public-Readiness.md origin/main` -> plan already in
  sync.
- `python scripts/audit_plan_doc.py
  plans/PR-EOM-Card-Vault-Public-Readiness.md` -> every required section OK.
- `python scripts/audit_plan_doc_files_touched.py
  plans/PR-EOM-Card-Vault-Public-Readiness.md origin/main` -> claimed and actual
  files both 7; OK.
- `python scripts/audit_plan_doc_diff_size.py
  plans/PR-EOM-Card-Vault-Public-Readiness.md origin/main` -> estimate and actual
  LOC matched with 0.0% drift; OK.
- `python scripts/audit_plan_code_consistency.py --base-ref origin/main
  plans/PR-EOM-Card-Vault-Public-Readiness.md` -> all 7 path claims resolve; OK.
- `git diff --check` -> exit 0.
- `bash scripts/push_pr.sh /tmp/atlas-eom-card-public-readiness-pr-body.md` ->
  managed local PR review passed and the reviewed head was pushed.
- boundary-probe: a correctly signed recurring bearer is admitted; commercial and
  one-time bearers take the no-card branch; an undecided residential bearer takes
  the blocked branch; malformed JSON token shapes, wrong-key fingerprints,
  missing/revoked/expired invitations, and every stored subject field drift are
  rejected before projection; provider-disabled/no-secret configuration remains
  admitted for the read.
- effect-trace: availability is controlled by the route's
  `_read_service_dependency`, not `_service_dependency` or
  `require_eom_card_vault_config`; the real-ASGI disabled-provider test reaches the
  new route and receives the exact minimal response while its transaction counter
  remains zero.
- Completed before publication: plan sync/audits, mechanical local review, guard
  boundary proof, database runtime-role proof, and cold diff reconstruction.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/card_vault.py` | 70 |
| `atlas_brain/eom_api/funnel.py` | 4 |
| `atlas_brain/services/eom_card_vault.py` | 261 |
| `plans/PR-EOM-Card-Vault-Public-Readiness.md` | 258 |
| `tests/test_eom_card_vault.py` | 243 |
| `tests/test_eom_funnel_capability_manifest.py` | 4 |
| `tests/test_eom_terms_acceptance.py` | 74 |
| **Total** | **914** |
