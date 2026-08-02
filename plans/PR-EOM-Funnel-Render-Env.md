# PR-EOM-Funnel-Render-Env

## Why this slice exists

Effingham Office Maids issue #59 is blocked at the live enable-and-wire step:
the tracker already proxies the Leads review flow to Atlas, and Atlas already
has the private funnel endpoints, but the EOM Render profile does not declare
the funnel enable flag, service-token digest slot, and canonical CRM database
confirmation bit, and the slim EOM entrypoint does not mount the funnel router
or enforce the full-app funnel datastore preflight.

### Problem-derived contract

- Root cause: `render.eom.yaml` declares receivables auth envs for the slim EOM
  API but omits the already-implemented `ATLAS_EOM_FUNNEL_` runtime gate and
  digest envs, while `atlas_brain.main_eom` mounts only the receivables router.
  A blueprint-created EOM service therefore has no checked-in config slot and
  would still 404 `/api/v1/eom-funnel/*` after the flag is enabled. Mounting the
  router alone is incomplete because the slim app would serve funnel routes
  without first proving the lifecycle, handoff, protected-owner, and NocoDB
  privilege invariants that the full app already requires.
- Correct fix must touch/change: add the funnel envs to the EOM Render blueprint,
  keep that blueprint fail-closed against the non-canonical attached database,
  mount the funnel router in the slim EOM app, validate its startup config, run
  its curated schema prerequisites when the funnel API is enabled, run the same
  datastore preflight before serving funnel routes, enroll the slim funnel
  dependency set in the relevant domain workflow filters, and update EOM render
  profile tests to prove the env contract, real-app route reachability, startup
  ordering, and rejection paths.
- Must not change: no Atlas endpoint behavior, CRM behavior, receivables config,
  migrations, public routes, or user-facing product shape.

## Scope (this PR)

Ownership lane: eom-funnel
Slice phase: production hardening

1. Add disabled-by-default Atlas EOM funnel env placeholders to `render.eom.yaml`.
2. Mount the existing funnel router in `atlas_brain.main_eom` under `/api/v1`.
3. Validate enabled funnel config at startup and include the funnel schema
   prerequisites in the EOM startup migration set only when the funnel API is
   enabled.
4. Extend the EOM render profile tests to assert the env contract and real-app
   funnel route.
5. Move the existing full-app funnel datastore preflight into a shared EOM module
   and call it from the slim app after database initialization and migrations.
6. Keep the slim EOM funnel path fail-closed unless the operator explicitly
   confirms `ATLAS_DB_CONNECTION_STRING` points at the canonical Atlas CRM store.
7. Add the slim funnel config/router/auth/readiness modules and readiness
   migrations to the relevant invoicing and EOM lead-pipeline workflow path
   filters.

### Review Contract

- Acceptance criteria:
  - [ ] `render.eom.yaml` contains `ATLAS_EOM_FUNNEL_API_ENABLED` with
    `value: "false"` so the blueprint remains fail-closed until runtime config
    deliberately enables it.
  - [ ] `render.eom.yaml` contains `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` with
    `sync: false` so the digest is supplied as a Render-managed secret value.
  - [ ] `render.eom.yaml` contains
    `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED` with `value: "false"` so the
    checked-in attached EOM database cannot serve funnel traffic by only flipping
    the API flag.
  - [ ] `tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth`
    asserts both funnel env entries and that no raw funnel token env is declared.
  - [ ] `atlas_brain.main_eom.app` exposes `/api/v1/eom-funnel/leads` and
    `/api/v1/eom-funnel/customer-handoffs` without importing the full API package.
  - [ ] `tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app`
    proves an enabled generated funnel token reaches the real slim app route.
  - [ ] `tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness`
    asserts the EOM startup migration set contains the receivables and funnel
    SQL prerequisites.
  - [ ] `tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled`
    proves the privileged funnel migration set is not run while the funnel API
    remains disabled by default.
  - [ ] `tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database`
    and `tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema`
    prove the enabled slim app fails closed without the authoritative database
    and handoff privilege schema.
  - [ ] `tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation`
    proves the enabled slim app fails before inspecting the pool unless canonical
    CRM database confirmation is set.
  - [ ] `tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation`
    proves enabled funnel migrations are not selected without canonical CRM
    confirmation.
  - [ ] `tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init`
    proves enabled funnel startup with missing canonical confirmation rejects
    before database initialization or migration execution.
  - [ ] `tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations`
    proves the slim startup preflight runs after migrations and before serving.
- Reachability proof: `pytest tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app`
  calls the real `atlas_brain.main_eom.app` at `/api/v1/eom-funnel/leads`.
- Affected surfaces: Render EOM blueprint config, shared EOM funnel readiness,
  full-app compatibility wrapper, slim EOM app startup/router wiring, domain CI
  path filters, and EOM render profile tests.
- Risk areas: deployment config, service-auth setup, live enablement drift,
  curated startup migration coverage, datastore privilege invariants.
- Reviewer rules triggered: R1, R2, R3, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM funnel startup admission for `atlas_brain.main_eom`
  now calls the shared datastore preflight before serving mounted funnel routes.
- Replaced-path behaviors: the full app keeps its `_require_eom_funnel_data_store`
  name as a compatibility wrapper over the shared preflight.
- Guard-relevant fields: `ATLAS_EOM_FUNNEL_API_ENABLED`,
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256`, `ATLAS_DB_ENABLED`, and
  `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`, and
  `ATLAS_EOM_RUN_MIGRATIONS`.
- Caller x input shape: enabled funnel config plus disabled DB, uninitialized DB
  pool, missing canonical CRM confirmation, or failed schema preflight raises
  before route serving.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `ATLAS_EOM_FUNNEL_API_ENABLED` remains
  `"false"` and `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED` remains `"false"`
  in `render.eom.yaml` until live runtime config deliberately overrides them
  after pointing the service at the canonical CRM store.
- Explicit value probe: `test_eom_profile_reaches_funnel_leads_through_real_app`
  sets the enabled flag and generated-token digest, then calls the real route.
- Absent value probe: `test_eom_startup_migration_set_expands_only_when_funnel_enabled`
  proves the default disabled funnel does not run the privileged funnel migration
  set, and
  `test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation` proves
  an enabled funnel still fails before DB-pool inspection unless canonical CRM
  confirmation is set.
- Default-session/default-context probe: N/A.
- Side-effect ordering:
  `test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init` proves the
  canonical CRM confirmation guard runs before DB initialization and migrations,
  and `test_eom_lifespan_runs_funnel_preflight_after_migrations` proves the slim
  app initializes the DB, runs migrations, then runs the funnel datastore
  preflight before yielding to requests after confirmation is present.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel_readiness.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `plans/PR-EOM-Funnel-Render-Env.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`

## Mechanism

The blueprint now declares the two Atlas funnel runtime knobs that the existing
`EOMFunnelConfig` reads through the `ATLAS_EOM_FUNNEL_` prefix. The API flag is
checked in as disabled, the token digest is declared as `sync: false` so a live
operator can set it without committing the value, and the canonical CRM
confirmation bit is checked in as false so the slim profile cannot serve funnel
traffic against the attached EOM candidate database by only flipping the API
flag.

The full-app datastore preflight now lives in `atlas_brain.eom_api.funnel_readiness`
so the slim profile can use it without importing the full API aggregate. The
slim EOM app imports the existing funnel router, validates the funnel config
during startup, mounts the router under `/api/v1`, runs the combined EOM API
readiness migration set only when the funnel API is enabled, and then runs the
datastore preflight before serving. The render-profile tests load the real YAML,
call the real slim app route with a generated service token, and prove the
enabled startup rejection paths.

The slim funnel dependency set is also enrolled in the invoicing and EOM
lead-pipeline workflow path filters so later guard-, config-, router-, auth-, or
readiness-migration-only changes run the same domain suites that prove startup
and privilege behavior.

## Intentional

- This PR does not enable the live service; it gives the Render profile the
  required env slots and makes the slim EOM entrypoint able to serve the already
  implemented funnel routes. Live enablement still requires setting the runtime
  values.
- The raw tracker bearer token is not declared in Atlas. Atlas stores only the
  SHA-256 digest.
- The privileged handoff migration is part of the enabled funnel readiness set,
  but the default disabled funnel path keeps the existing receivables-only
  startup migrations so the fail-closed blueprint can still boot before operator
  role provisioning.
- The checked-in slim Render blueprint still uses the attached EOM database for
  the current receivables candidate, so enabled funnel startup additionally
  requires `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true` after
  `ATLAS_DB_CONNECTION_STRING` is pointed at the canonical Atlas CRM database.

## Deferred

- Set live Atlas and tracker runtime env values, restart/redeploy both services,
  and smoke the portal Leads tab.
- The original issue's pipeline board and booking actions remain a separate
  product-shape follow-up from this config slice.

Parked hardening: none.

## Verification

- Passed: `python -m py_compile atlas_brain/main.py atlas_brain/main_eom.py atlas_brain/eom_api/funnel_readiness.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_migration_helper_uses_curated_set tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_lifespan_closes_database_when_migration_startup_fails tests/test_eom_render_profile.py::test_eom_lifespan_initializes_database_without_running_migrations tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app`.
- Passed: `python -m py_compile atlas_brain/eom_api/config.py atlas_brain/main_eom.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema -q`.
- Passed: `python -m py_compile atlas_brain/main_eom.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations -q`.
- Passed: `python -m py_compile tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app -q`.
- Could not run locally: `pytest tests/test_eom_lead_conversion.py::test_enabled_full_atlas_funnel_requires_authoritative_data_store` because this minimal EOM venv lacks optional full-app dependency `numpy`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 4 |
| `.github/workflows/atlas_invoicing_checks.yml` | 18 |
| `atlas_brain/eom_api/config.py` | 7 |
| `atlas_brain/eom_api/funnel_readiness.py` | 157 |
| `atlas_brain/main.py` | 145 |
| `atlas_brain/main_eom.py` | 73 |
| `plans/PR-EOM-Funnel-Render-Env.md` | 231 |
| `render.eom.yaml` | 6 |
| `tests/test_eom_render_profile.py` | 374 |
| **Total** | **1015** |
