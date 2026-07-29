# PR-EOM-Render-Receivables-Auth

## Why this slice exists

The operator asked to continue the EOM Render hosting arc after #2220 merged.
#2217 created the slim EOM API profile, and #2220 wired the candidate Render
Postgres connection string. The remaining blocker before the already-Render-
hosted EOM time tracker can call Atlas is service-to-service auth: the slim API
needs a digest-only trust anchor that can be enabled safely after the receivables
database schema is provisioned.

This slice narrows the old "Slice C" into the Atlas-side trust anchor first:
allow the slim API to start only with digest-only bearer-token configuration,
keep raw bearer material out of the Atlas API service, and prove the real EOM
route accepts generated-token callers while rejecting non-generated bearers. The
candidate Blueprint deliberately keeps the API disabled because the fresh Render
database still has migrations disabled and this slice does not provision the
receivables tables.

Diff-budget rationale: this PR exceeds the 400-added-line soft cap because the
reviewer-blocker repairs are indivisible from this trust-anchor slice. The
production code, Render candidate posture, plan contract, and held-out
regressions must move together to avoid either a red PR or a deployable candidate
that is knowingly unsafe.

### Problem-derived contract

- Root cause: The slim EOM API currently has a generated-token/digest helper,
  but the runtime settings object exposes only the enable flag and startup
  validation rejects every operator-supplied enabled config. The first fix
  added a digest setting, but review showed two remaining bugs: the settings
  model ignores the forbidden raw-token env var before validation can see it,
  and digest shape alone cannot prove that the bearer preimage came from the
  generated-token helper. Follow-up review found the raw-token guard still
  missed env-file settings sources, and request validation still admitted
  generated-prefix bearers with the wrong payload length. Latest review also
  showed that the raw-token settings-source guard was case-sensitive and gated
  by the API enable flag, leaving lowercase aliases and disabled raw-token config
  admitted before projection. The newest review showed the evidence still proved
  sampled strings rather than the raw-key and bearer grammars, and the real-app
  reachability proof bypassed env projection by overriding the auth dependency.
  As a result the Render candidate cannot safely carry the receivables API
  settings without fail-closed checks and class-level evidence for both
  settings-source admission and request admission.
- Correct fix must touch/change: Add a typed digest-only runtime setting for
  the EOM receivables service token; reject the legacy/raw bearer-token key
  across process env and env-file settings sources case-insensitively and
  regardless of enable state before model projection can hide it; update request
  validation to require the presented bearer token to
  match the exact generated `eomrx_v1_...` payload length before hashing and
  comparing it; update the Render candidate to prompt for the digest while
  keeping the receivables API disabled until schema provisioning is wired; and
  add route/startup tests that prove valid runtime digest config authorizes the
  real EOM route through actual environment projection while missing, malformed,
  placeholder, raw-token-bearing, non-generated matching-digest, max+1
  generated-format, and grammar-generated invalid paths fail closed.
- Must not change: Do not store raw bearer tokens on the Atlas API service,
  modify the full Atlas app, enable migrations, change database schema, touch
  the EOM time tracker repository/deployment, change customer/onboarding or
  Stripe behavior, or touch unrelated open PR lanes.

## Scope (this PR)

Ownership lane: eom/render-receivables-auth
Slice phase: vertical slice

1. Allow the slim EOM API profile to run the receivables routes with a
   digest-only service-token runtime config.
2. Update the reviewable Render candidate so the SHA-256 digest secret is
   prompted but the API remains disabled until the receivables schema is
   provisioned.
3. Prove the real EOM app route accepts the generated token and rejects bad or
   missing auth through the runtime config object.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_and_receivables_auth`
    proves `render.eom.yaml` keeps the receivables API disabled, prompts for
    `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` with `sync: false`,
    keeps migrations disabled, preserves the Render Postgres connection-string
    mapping, and does not define any raw receivables token env var.
  - `tests/test_eom_render_profile.py::test_eom_receivables_runtime_config_accepts_generated_digest_only`
    proves `EOMInvoicingConfig` can enable the API with only a generated token
    digest, and that the config model exposes no raw-token field.
  - `tests/test_eom_render_profile.py::test_eom_receivables_raw_token_source_admission_matches_casefold_oracle`
    proves raw-token settings-source admission matches an independent
    casefolding oracle across process-env and dotenv sources, canonical/mixed
    case raw-token aliases, unrelated keys, and blank/nonblank values.
  - `tests/test_eom_render_profile.py::test_eom_receivables_runtime_config_rejects_raw_token_env_before_projection`
    proves a real environment-projected `EOMInvoicingConfig` rejects uppercase
    and lowercase `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` aliases whether
    the receivables API flag is enabled or disabled, even when a valid digest is
    also configured.
  - `tests/test_eom_render_profile.py::test_eom_profile_rejects_raw_receivables_token_from_dotenv_before_projection`
    proves the real EOM entrypoint rejects uppercase and lowercase forbidden raw
    token aliases supplied from admitted dotenv settings sources whether the API
    flag is enabled or disabled, before `extra="ignore"` can hide them.
  - `tests/test_eom_render_profile.py::test_eom_receivables_startup_rejects_unsafe_enabled_runtime_config`
    proves enabled config fails closed for missing digest, malformed digest,
    placeholder-derived digest, and any object carrying raw bearer-token
    material.
  - `tests/test_eom_render_profile.py::test_eom_receivables_request_auth_does_not_rescan_settings_sources`
    proves request-time bearer validation uses the in-memory runtime config and
    does not synchronously re-read dotenv settings sources on the async request
    path.
  - `tests/test_eom_render_profile.py::test_eom_receivables_bearer_admission_matches_generated_token_grammar`
    proves the free-form `Authorization` bearer token admission matches an
    independent prefix × payload grammar oracle across homogeneous payloads,
    mixed allowed payloads, and invalid characters inserted at first, middle,
    and last payload positions, with matching digests supplied for every
    generated ASCII token candidate so bad grammar cannot be hidden as a digest
    mismatch.
  - `tests/test_eom_render_profile.py::test_eom_receivables_ready_route_is_fail_closed`
    proves `require_receivables_api()` has one bearer-admission choke point:
    `_validate_generated_token()` must accept only the bounded
    `eomrx_v1_` prefix, exact payload length, and allowed-character recognizer
    before digest comparison. Every missing, unrecognized, malformed,
    non-ASCII, non-generated matching-digest, and max+1 generated-prefix bearer
    defaults to 401; the configured generated bearer returns 200; and disabled
    runtime config returns 503.
  - `tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app`
    proves a fresh process imports the real `atlas_brain.main_eom:app`, projects
    `ATLAS_INVOICING_*` env vars into the runtime singleton, uses no auth
    dependency override, and reaches `/api/v1/receivables/ready` with only the
    database/service seam mocked.
- Reachability proof: `tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app`
  launches a fresh Python process, drives the real EOM FastAPI app, and asserts
  the observable `{"status": "ready"}` HTTP JSON result with runtime digest auth
  enabled from `ATLAS_INVOICING_*`, no auth dependency override, and DB/service
  access mocked only at the service seam.
- Affected surfaces: EOM invoicing config, EOM receivables auth validation,
  candidate Render Blueprint, and focused EOM Render-profile tests.
- Risk areas: raw secret placement, auth fail-open behavior, startup failure
  behavior, Render env drift, accidental migration/API coupling, and EOM route
  reachability.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R8, R10, R11, R12, R13, R14.

### Files touched

- `atlas_brain/eom_api/auth.py`
- `atlas_brain/eom_api/config.py`
- `plans/PR-EOM-Render-Receivables-Auth.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`

## Mechanism

`EOMInvoicingConfig` gains a digest-only
`receivables_service_token_sha256` setting loaded through the existing
`ATLAS_INVOICING_` prefix. Its settings model rejects the legacy/raw
`ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` key from process env and admitted
dotenv settings sources with case-insensitive key matching and regardless of
the API enable flag, so `extra="ignore"` cannot hide forbidden bearer material
before validation. `validate_receivables_api_config()`
still returns early when the API is disabled. When enabled, it rejects any config
object that carries raw bearer-token material, then validates the digest shape
and rejects placeholder-derived digests. `require_receivables_api()` validates the presented bearer token
against the exact generated-token prefix and payload length before hashing it
and comparing digests with `hmac.compare_digest()`.

`render.eom.yaml` keeps `ATLAS_INVOICING_RECEIVABLES_API_ENABLED=false` and adds
`ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` as `sync: false`, so the
candidate can collect the digest without exposing the API before schema
provisioning. The raw generated token remains caller-side material for the later
EOM time tracker slice.

## Intentional

- This PR does not put the raw bearer token in `render.eom.yaml` or
  `EOMInvoicingConfig`; only the SHA-256 digest belongs on the Atlas API side.
- This PR keeps the candidate file named `render.eom.yaml`, not render dot yaml,
  so it remains reviewable infrastructure shape instead of auto-applied live
  Render configuration.
- This PR keeps the receivables API disabled in the Render candidate because
  database migrations remain disabled.
- The in-process trusted config helper remains available for narrow tests, but
  the real app reachability proof uses the `EOMInvoicingConfig` singleton
  projected from actual environment variables.

## Deferred

- Next slice: wire the EOM time tracker caller to the Atlas private service URL,
  store the raw generated bearer token only on that caller side, and prove
  private-service `/api/v1/receivables/ready` connectivity.
- Later slices: decide EOM migration ownership/order for production schema,
  connect EOM funnel/customer lifecycle events, and plan deprecation of duplicate
  or old onboarding/receivables code once the new path proves itself.

Parked hardening: none.

## Verification

- Passed locally:
  - Command: python -m pytest tests/test_eom_render_profile.py -- 31 passed, 1 warning
  - Command: python -m py_compile atlas_brain/eom_api/auth.py atlas_brain/eom_api/config.py atlas_brain/main_eom.py tests/test_eom_render_profile.py -- passed
  - Command: git diff --check -- passed
  - Command: python scripts/sync_pr_plan.py plans/PR-EOM-Render-Receivables-Auth.md -- passed
  - Command: python scripts/audit_plan_code_consistency.py plans/PR-EOM-Render-Receivables-Auth.md -- passed

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/auth.py` | 28 |
| `atlas_brain/eom_api/config.py` | 57 |
| `plans/PR-EOM-Render-Receivables-Auth.md` | 210 |
| `render.eom.yaml` | 2 |
| `tests/test_eom_render_profile.py` | 578 |
| **Total** | **875** |
