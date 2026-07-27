# PR-EOM-Render-Receivables-Auth

## Why this slice exists

The operator asked to continue the EOM Render hosting arc after #2220 merged.
#2217 created the slim EOM API profile, and #2220 wired the candidate Render
Postgres connection string. The remaining blocker before the already-Render-
hosted EOM time tracker can call Atlas is service-to-service auth: the API
still refuses to start when `ATLAS_INVOICING_RECEIVABLES_API_ENABLED=true`.

This slice narrows the old "Slice C" into the Atlas-side trust anchor first:
allow the slim API to start only with digest-only bearer-token configuration,
keep raw bearer material out of the Atlas API service, and prove the real EOM
route accepts generated-token callers while rejecting non-generated bearers.

### Problem-derived contract

- Root cause: The slim EOM API currently has a generated-token/digest helper,
  but the runtime settings object exposes only the enable flag and startup
  validation rejects every operator-supplied enabled config. The first fix
  added a digest setting, but review showed two remaining bugs: the settings
  model ignores the forbidden raw-token env var before validation can see it,
  and digest shape alone cannot prove that the bearer preimage came from the
  generated-token helper. As a result the Render candidate cannot safely enable
  the receivables API without fail-closed checks on both config projection and
  request admission.
- Correct fix must touch/change: Add a typed digest-only runtime setting for
  the EOM receivables service token; reject the legacy/raw bearer-token env var
  before model projection can hide it; update request validation to require the
  presented bearer token to match the generated `eomrx_v1_...` format before
  hashing and comparing it; update the Render candidate to prompt for the
  digest and enable the receivables API; and add route/startup tests that prove
  valid runtime digest config authorizes the real EOM route while missing,
  malformed, placeholder, raw-token-bearing, and non-generated matching-digest
  paths fail closed.
- Must not change: Do not store raw bearer tokens on the Atlas API service,
  modify the full Atlas app, enable migrations, change database schema, touch
  the EOM time tracker repository/deployment, change customer/onboarding or
  Stripe behavior, or touch unrelated open PR lanes.

## Scope (this PR)

Ownership lane: eom/render-receivables-auth
Slice phase: vertical slice

1. Allow the slim EOM API profile to run the receivables routes with a
   digest-only service-token runtime config.
2. Update the reviewable Render candidate so the API is enabled only with a
   prompted SHA-256 digest secret and no raw token value.
3. Prove the real EOM app route accepts the generated token and rejects bad or
   missing auth through the runtime config object.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_and_receivables_auth`
    proves `render.eom.yaml` enables the receivables API, prompts for
    `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` with `sync: false`,
    keeps migrations disabled, preserves the Render Postgres connection-string
    mapping, and does not define any raw receivables token env var.
  - `tests/test_eom_render_profile.py::test_eom_receivables_runtime_config_accepts_generated_digest_only`
    proves `EOMInvoicingConfig` can enable the API with only a generated token
    digest, and that the config model exposes no raw-token field.
  - `tests/test_eom_render_profile.py::test_eom_receivables_runtime_config_rejects_raw_token_env_before_projection`
    proves a real environment-projected `EOMInvoicingConfig` rejects
    `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` even when a valid digest is
    also configured.
  - `tests/test_eom_render_profile.py::test_eom_receivables_startup_rejects_unsafe_enabled_runtime_config`
    proves enabled config fails closed for missing digest, malformed digest,
    placeholder-derived digest, and any object carrying raw bearer-token
    material.
  - `tests/test_eom_render_profile.py::test_eom_receivables_ready_route_is_fail_closed`
    proves the route boundary returns 401 for missing/invalid bearer tokens,
    401 for malformed non-ASCII bearer bytes, 401 for a non-generated bearer
    whose digest matches config, 200 for the configured generated bearer token,
    and 503 when the API is disabled.
  - `tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app`
    proves the real `atlas_brain.main_eom:app` route transport reaches
    `/api/v1/receivables/ready` using the runtime digest config, not the
    in-process trusted test config.
- Reachability proof: `tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app`
  drives the real EOM FastAPI app and asserts the observable
  `{"status": "ready"}` HTTP JSON result with runtime digest auth enabled and
  DB access disabled/mocked at the service seam.
- Affected surfaces: EOM invoicing config, EOM receivables auth validation,
  candidate Render Blueprint, and focused EOM Render-profile tests.
- Risk areas: raw secret placement, auth fail-open behavior, startup failure
  behavior, Render env drift, accidental migration/API coupling, and EOM route
  reachability.
- Reviewer rules triggered: R1, R2, R3, R5, R6, R10, R11, R12, R14.

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
`ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN` env var when the API is enabled, so
`extra="ignore"` cannot hide forbidden bearer material before validation.
`validate_receivables_api_config()` still returns early when the API is
disabled. When enabled, it repeats the raw-env guard, rejects any config object
that carries raw bearer-token material, then validates the digest shape and
rejects placeholder-derived digests. `require_receivables_api()` validates the
presented bearer token against the generated-token format before hashing it and
comparing digests with `hmac.compare_digest()`.

`render.eom.yaml` turns on `ATLAS_INVOICING_RECEIVABLES_API_ENABLED` and adds
`ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` as `sync: false`, so Render
prompts the operator for the digest without hardcoding it in the candidate
Blueprint. The raw generated token remains caller-side material for the later
EOM time tracker slice.

## Intentional

- This PR does not put the raw bearer token in `render.eom.yaml` or
  `EOMInvoicingConfig`; only the SHA-256 digest belongs on the Atlas API side.
- This PR keeps the candidate file named `render.eom.yaml`, not render dot yaml,
  so it remains reviewable infrastructure shape instead of auto-applied live
  Render configuration.
- This PR enables only the auth gate for the receivables API candidate; database
  migrations remain disabled.
- The in-process trusted config helper remains available for narrow tests, but
  the real app reachability proof uses `EOMInvoicingConfig`.

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
  - Command: python -m pytest tests/test_eom_render_profile.py
  - Command: python -m py_compile atlas_brain/eom_api/auth.py atlas_brain/eom_api/config.py atlas_brain/main_eom.py tests/test_eom_render_profile.py
  - Command: git diff --check
  - Command: python scripts/sync_pr_plan.py plans/PR-EOM-Render-Receivables-Auth.md
  - Command: python scripts/audit_plan_code_consistency.py plans/PR-EOM-Render-Receivables-Auth.md

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/auth.py` | 38 |
| `atlas_brain/eom_api/config.py` | 34 |
| `plans/PR-EOM-Render-Receivables-Auth.md` | 161 |
| `render.eom.yaml` | 4 |
| `tests/test_eom_render_profile.py` | 149 |
| **Total** | **386** |
