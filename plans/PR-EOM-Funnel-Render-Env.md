# PR-EOM-Funnel-Render-Env

## Why this slice exists

Effingham Office Maids issue #59 is blocked at the live enable-and-wire step:
the tracker already proxies the Leads review flow to Atlas, and Atlas already
has the private funnel endpoints, but the EOM Render profile does not declare
the funnel enable flag, service-token digest slot, dedicated canonical CRM DSN,
and canonical CRM database confirmation bit, and the slim EOM entrypoint does
not mount the funnel router or enforce the full-app funnel datastore preflight.

This slice is oversized because the reviewed failure modes are coupled at the
deployment boundary: declaring the Render env slots without mounting the slim
router leaves the enabled service at 404, mounting the router without the shared
preflight serves privileged funnel routes against an unproven schema, sharing the
receivables pool would make funnel traffic follow the attached EOM database
instead of the canonical CRM store, and bootstrapping migration 354 inside normal
serving startup can leave the service stuck after the temporary admin privilege
is revoked. The workflow path-filter updates and route/startup tests are part of
the same indivisible safety boundary because they keep later edits to the new
guard, config, router, auth, database, and migration prerequisites inside the
domain CI that proves this startup contract.

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
  give the slim funnel routes a separate canonical CRM pool instead of reusing
  the receivables pool, mount the funnel router in the slim EOM app, validate its
  startup config, run only the serving-safe curated schema prerequisites when the
  funnel API is enabled, leave privileged migrations 354 and 356 as out-of-band
  bootstrap prerequisites, run the same datastore preflight before serving
  funnel routes, enroll the slim funnel dependency set in the relevant domain
  workflow filters, and update EOM render profile tests to prove the env
  contract, real-app route reachability, startup ordering, and rejection paths.
- Must not change: no Atlas endpoint behavior, CRM behavior, receivables config,
  public routes, or user-facing product shape.

## Scope (this PR)

Ownership lane: eom-funnel
Slice phase: production hardening

1. Add disabled-by-default Atlas EOM funnel env placeholders to `render.eom.yaml`.
2. Mount the existing funnel router in `atlas_brain.main_eom` under `/api/v1`.
3. Validate enabled funnel config at startup and include the funnel schema
   prerequisites in the EOM startup migration set only when the funnel API is
   enabled, excluding migrations 354 and 356 from serving startup.
4. Extend the EOM render profile tests to assert the env contract and real-app
   funnel route.
5. Move the existing full-app funnel datastore preflight into a shared EOM module
   and call it from the slim app after database initialization and migrations.
6. Keep the slim EOM funnel path fail-closed unless the operator explicitly
   confirms `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` points at the canonical
   Atlas CRM store.
7. Add the slim funnel config/router/auth/readiness modules and readiness
   migrations to the relevant invoicing and EOM lead-pipeline workflow path
   filters.
8. Defer the legacy LLM registry import in `get_pipeline_llm()` until the legacy
   fallback path uses it, so full-suite CI can exercise mocked workload routing
   without requiring optional provider dependencies such as `torch`.
9. Keep `BaseModelService` importable without `torch` so HTTP/cloud LLM providers
   can be constructed in minimal CI environments, prove the torch-absent cloud
   provider metrics path in a subprocess, sync the extracted LLM bridge comment,
   and shrink the unit-gate baseline entries that now pass.
10. Require migration 356 to name the serving login through the
    `atlas.eom_funnel_runtime_role` session setting before it can grant runtime
    handoff table privileges or record success, and reject NOLOGIN/protected
    owner, superuser, database-owner, `CREATEROLE`, `CREATEDB`, and `BYPASSRLS`
    roles as grant targets.
11. Mark migration 356 as an out-of-band bootstrap file so the full app's
    unfiltered startup migration runner skips it unless a DBA runner explicitly
    selects it by name.
12. Close the receivables pool even if the dedicated funnel pool raises while
    closing during slim EOM shutdown.

### Review Contract

- Acceptance criteria:
  - [ ] `render.eom.yaml` contains `ATLAS_EOM_FUNNEL_API_ENABLED` with
    `value: "false"` so the blueprint remains fail-closed until runtime config
    deliberately enables it.
  - [ ] `render.eom.yaml` contains `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` with
    `sync: false` so the digest is supplied as a Render-managed secret value.
  - [ ] `render.eom.yaml` contains `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` with
    `sync: false` so funnel traffic can use the canonical CRM database without
    moving receivables off the attached EOM database.
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
    asserts the EOM startup migration set contains the receivables prerequisites,
    the serving-safe funnel SQL prerequisites, and lists migrations 354 and 356
    only as out-of-band bootstrap migrations.
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
  - [ ] `tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init`
    proves the slim-only dedicated DSN requirement rejects before database
    initialization without changing the shared full-app funnel auth validator.
  - [ ] `tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract`
    proves the full Atlas app can still use enabled funnel auth with only the
    generated service-token digest and its primary Atlas pool.
  - [ ] `tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations`
    proves the slim startup preflight runs after migrations and before serving.
  - [ ] `tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout`
    proves the slim funnel request pool keeps its normal timeout while startup
    DDL uses a temporary migration pool with `command_timeout=None`.
  - [ ] `tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema`
    proves readiness SQL checks the connected `current_user` for the SELECT,
    INSERT, and UPDATE privileges needed by the lead-review and customer-handoff
    routes.
  - [ ] Migration safety disposition: enabled slim startup selects funnel
    readiness migrations 346/351/353/355 only after canonical CRM confirmation
    and a dedicated funnel CRM pool is initialized. Migrations 354 and 356 are
    deliberately excluded from serving startup and must be applied out of band
    during the bootstrap window. Migration 356 requires the DBA session to set
    `atlas.eom_funnel_runtime_role` to the serving LOGIN role before recording;
    the migration rejects `atlas_eom_handoff_owner` and other NOLOGIN roles as
    runtime targets. Then the temporary admin membership must be revoked before
    restarting the serving app. Before migration 354 is applied, rollback is
    leaving `ATLAS_EOM_FUNNEL_API_ENABLED=false` or
    `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=false`, which keeps the slim
    profile on the receivables-only serving path. After migrations 354 and 356
    are applied to the canonical CRM database, the ownership transfers, NocoDB
    grant narrowing, schema privilege changes, and runtime handoff grants are
    intentional hardening for the handoff tables. A database rollback requires
    restoring the canonical CRM database from a pre-migration backup or a
    DBA-owned audited reverse privilege script built from a pre-migration
    privilege snapshot.
- Reachability proof: `pytest tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app`
  calls the real `atlas_brain.main_eom.app` at `/api/v1/eom-funnel/leads`.
- Affected surfaces: Render EOM blueprint config, shared EOM funnel readiness,
  slim EOM funnel CRM pool, full-app compatibility wrapper, slim EOM app
  startup/router wiring, domain CI path filters, EOM render profile tests, EOM
  lead conversion startup-auth tests, and an LLM routing import-order fix plus
  torch-optional base-service import fix exposed by the full-suite unit gate.
- Risk areas: deployment config, service-auth setup, live enablement drift,
  curated startup migration coverage, datastore privilege invariants.
- Reviewer rules triggered: R1, R2, R3, R4, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM funnel startup admission for `atlas_brain.main_eom`
  now calls the shared datastore preflight before serving mounted funnel routes.
- Replaced-path behaviors: the full app keeps its `_require_eom_funnel_data_store`
  name as a compatibility wrapper over the shared preflight.
- Guard-relevant fields: `ATLAS_EOM_FUNNEL_API_ENABLED`,
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256`,
  `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`, `ATLAS_DB_ENABLED`,
  `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`, and
  `ATLAS_EOM_RUN_MIGRATIONS`.
- Caller x input shape: enabled funnel config plus disabled DB, uninitialized DB
  pool, missing canonical CRM confirmation, or failed schema preflight raises
  before route serving.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `ATLAS_EOM_FUNNEL_API_ENABLED` remains
  `"false"`, `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` is a `sync: false` secret,
  and `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED` remains `"false"` in
  `render.eom.yaml` until live runtime config deliberately sets the funnel CRM
  DSN to the canonical CRM store and confirms it.
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
- `atlas_brain/eom_api/funnel_database.py`
- `atlas_brain/eom_api/funnel_readiness.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/pipelines/llm.py`
- `atlas_brain/services/base.py`
- `atlas_brain/storage/migrations/356_eom_customer_handoff_runtime_grants.sql`
- `atlas_brain/storage/migrations/__init__.py`
- `extracted_llm_infrastructure/pipelines/llm.py`
- `extracted_llm_infrastructure/services/base.py`
- `plans/PR-EOM-Funnel-Render-Env.md`
- `render.eom.yaml`
- `tests/test_eom_lead_conversion.py`
- `tests/test_eom_lead_conversion_integration.py`
- `tests/test_base_model_service_torch_optional.py`
- `tests/test_eom_render_profile.py`
- `tests/test_migrations_runner.py`
- `tests/unit_gate_baseline.txt`

## Mechanism

The blueprint now declares the Atlas funnel runtime knobs that the existing
`EOMFunnelConfig` reads through the `ATLAS_EOM_FUNNEL_` prefix. The API flag is
checked in as disabled, the token digest and dedicated funnel CRM DSN are
declared as `sync: false` so a live operator can set them without committing the
values, and the canonical CRM confirmation bit is checked in as false so the
slim profile cannot serve funnel traffic against the attached EOM candidate
database by only flipping the API flag.

The slim EOM profile now keeps receivables on the existing
`ATLAS_DB_CONNECTION_STRING` pool and initializes a separate funnel CRM pool from
`ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` only when the funnel API is enabled. The
mounted funnel router overrides only its CRM dependency to use that dedicated
pool, so funnel lead traffic can be pointed at canonical Atlas CRM without
moving receivables off `atlas-eom-postgres`. The dedicated DSN requirement is
enforced by the slim `main_eom` startup admission, not by the shared funnel auth
validator, so the full Atlas app keeps its existing digest-only funnel
configuration contract and primary-pool behavior.

The full-app datastore preflight now lives in `atlas_brain.eom_api.funnel_readiness`
so the slim profile can use it without importing the full API aggregate. The
slim EOM app imports the existing funnel router, validates the funnel config
during startup, mounts the router under `/api/v1`, runs the receivables readiness
migrations on the receivables pool, runs only the serving-safe funnel readiness
migrations 346/351/353/355 through a temporary unbounded-command-timeout funnel
migration pool when the funnel API is enabled, and then runs the datastore
preflight before serving. The normal request pool still has a 30-second command
timeout. Migrations 354 and 356 remain out-of-band bootstrap migrations because
they need temporary admin membership that must be revoked before normal serving
startup. Migration 356 is a separate repair migration rather than an edit to
already-recorded migration 354: it runs after the protected handoff ownership
transfer and grants the runtime login the `SELECT`/`INSERT`/`UPDATE` privileges
required by the handoff transaction. The shared datastore preflight now also requires the connected
`current_user` to have the SELECT, INSERT, and UPDATE privileges used by the two
funnel routes, so a read-only or NocoDB login cannot pass startup and then fail
at request time. The render-profile tests load the real YAML, call the real slim
app route with a generated service token, and prove the enabled startup
rejection paths.

The slim funnel dependency set is also enrolled in the invoicing and EOM
lead-pipeline workflow path filters so later guard-, config-, router-, auth-, or
readiness-migration-only changes run the same domain suites that prove startup
and privilege behavior.

The unit-gate full-suite selection also exposed that `get_pipeline_llm()` loaded
the legacy LLM registry before workload routing could return through mocked or
explicit non-legacy paths. The registry import now stays on the legacy fallback
path where `llm_registry.get_active()` is used, avoiding optional provider
imports for workload routes that do not need them.

The same unit-gate reproduction showed the shared LLM base class required
`torch` at import time even for HTTP/cloud providers. `BaseModelService` now uses
a torch-unavailable CUDA shim when `torch` is absent, preserving CUDA metrics
when torch is installed while allowing provider imports in minimal CI. The
reasoning-routing unit-gate baseline entries made stale by that fix were
removed.

## Intentional

- This PR does not enable the live service; it gives the Render profile the
  required env slots and makes the slim EOM entrypoint able to serve the already
  implemented funnel routes. Live enablement still requires setting the runtime
  values.
- The raw tracker bearer token is not declared in Atlas. Atlas stores only the
  SHA-256 digest.
- Receivables keep using the existing `ATLAS_DB_CONNECTION_STRING` pool. Funnel
  traffic uses `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`, which is required only
  when `ATLAS_EOM_FUNNEL_API_ENABLED=true` in the slim EOM profile.
- The slim funnel request pool keeps a 30-second command timeout, but startup
  migrations use a short-lived migration pool with no command timeout so
  production DDL is not cancelled by the request-query cap.
- Migrations 354 and 356 are not part of serving startup. They are out-of-band
  bootstrap prerequisites that must be applied with temporary admin membership
  and followed by an explicit revoke/restart before the serving app can pass
  readiness.
- Migration 356 repairs databases that already recorded migration 354 by
  granting the runtime login `SELECT`/`INSERT`/`UPDATE` on
  `eom_customer_handoffs` after ownership has moved to
  `atlas_eom_handoff_owner`, so those privileges survive revoking the temporary
  guard-role membership.
- The negative runtime role-capability policy is slim-only. The full Atlas app
  keeps its shared preflight compatibility wrapper on the primary pool, while
  `atlas_brain.main_eom` opts the dedicated funnel DSN into the extra superuser,
  database-owner, `CREATEROLE`, `CREATEDB`, and `BYPASSRLS` rejection.
- Integration schema setup provisions a non-privileged runtime login before
  applying migration 356, so the CI administrator remains only the migration
  executor and is no longer recorded as the serving role target.
- The checked-in slim Render blueprint still uses the attached EOM database for
  the current receivables candidate, so enabled funnel startup additionally
  requires `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true` after
  `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` is pointed at the canonical Atlas CRM
  database.
- Migrations 354 and 356 privilege changes are intentionally not undone by
  disabling the feature flag after they have been applied to the canonical CRM database.
  They are database hardening, not a reversible feature toggle; reversing them
  requires restoring a pre-migration database backup or executing a DBA-owned
  audited reverse privilege script from a pre-migration privilege snapshot.
- The LLM routing import-order change is limited to when the existing legacy
  fallback path imports `llm_registry`; explicit workload resolution keeps its
  existing helper order and behavior.
- The torch fallback only changes import-time availability when `torch` is not
  installed; environments with `torch` installed still use the real module.

## Deferred

- Apply migrations 354 and 356 out of band on the canonical CRM database, revoke
  the temporary admin membership, set live Atlas and tracker runtime env values,
  restart/redeploy both services, and smoke the portal Leads tab.
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
- Passed: `pytest tests/test_reasoning_graph_routing.py::test_anthropic_workload_uses_anthropic_primary_without_vllm_fallback tests/test_reasoning_graph_routing.py::test_openrouter_workload_normalizes_deprecated_gpt_oss tests/test_reasoning_graph_routing.py::test_vllm_workload_uses_anthropic_as_labeled_fallback -q`.
- Passed: `python -m pytest tests/test_reasoning_graph_routing.py -m "not integration and not e2e" --continue-on-collection-errors -rfE --tb=short -q -p no:cacheprovider`.
- Passed: `python scripts/check_unit_gate.py --baseline tests/unit_gate_baseline.txt --base-baseline /tmp/base_unit_gate_baseline.txt --selected-files /tmp/selected_reasoning_graph.txt --pytest-args tests/test_reasoning_graph_routing.py -m "not integration and not e2e" --continue-on-collection-errors -rfE --tb=no -q -p no:cacheprovider`.
- Passed: `python -m py_compile atlas_brain/main_eom.py atlas_brain/eom_api/config.py atlas_brain/eom_api/funnel_auth.py atlas_brain/eom_api/funnel_database.py tests/test_eom_render_profile.py tests/test_eom_lead_conversion.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_initializes_database_without_running_migrations tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app tests/test_eom_lead_conversion.py::test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup -q`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_initializes_database_without_running_migrations tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app tests/test_eom_lead_conversion.py::test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup -q`.
- Passed: `python -m py_compile atlas_brain/main_eom.py atlas_brain/eom_api/funnel_auth.py tests/test_eom_render_profile.py tests/test_eom_lead_conversion.py`.
- Passed: `pytest tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract tests/test_eom_lead_conversion.py::test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app -q`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_initializes_database_without_running_migrations tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract tests/test_eom_lead_conversion.py::test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup -q`.
- Passed: `python -m py_compile atlas_brain/main_eom.py atlas_brain/eom_api/funnel_database.py atlas_brain/eom_api/funnel_readiness.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout -q`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_initializes_database_without_running_migrations tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app tests/test_eom_render_profile.py::test_eom_profile_reaches_funnel_leads_through_real_app tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract tests/test_eom_lead_conversion.py::test_enabled_funnel_accepts_a_fresh_generated_service_token_at_startup tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup -q`.
- Passed: `python -m py_compile atlas_brain/eom_api/funnel_readiness.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_migration_helper_uses_curated_set tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract -q`.
- Passed: `python -m py_compile atlas_brain/main_eom.py atlas_brain/eom_api/funnel_readiness.py tests/test_eom_render_profile.py tests/test_eom_lead_conversion_integration.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout -q`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_receivables_and_funnel_auth tests/test_eom_render_profile.py::test_eom_startup_migrations_are_curated_for_api_readiness tests/test_eom_render_profile.py::test_eom_migration_helper_uses_curated_set tests/test_eom_render_profile.py::test_eom_startup_migration_set_expands_only_when_funnel_enabled tests/test_eom_render_profile.py::test_eom_startup_migration_selection_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_startup_migration_runner_uses_enabled_api_set tests/test_eom_render_profile.py::test_eom_startup_migration_runner_skips_uninitialized_pool tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_authoritative_database tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_requires_canonical_crm_confirmation tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_funnel_migration_pool_removes_command_timeout tests/test_eom_render_profile.py::test_eom_lifespan_rejects_unconfirmed_funnel_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_rejects_enabled_funnel_without_slim_dsn_before_database_init tests/test_eom_render_profile.py::test_eom_lifespan_runs_funnel_preflight_after_migrations tests/test_eom_lead_conversion.py::test_enabled_funnel_rejects_missing_or_malformed_token_digests_at_startup tests/test_eom_lead_conversion.py::test_shared_funnel_auth_validator_preserves_full_app_digest_only_contract -q`.
- Passed: `python -m py_compile atlas_brain/storage/migrations/__init__.py tests/test_migrations_runner.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_migrations_runner.py::test_default_runner_skips_out_of_band_bootstrap_migrations tests/test_migrations_runner.py::test_explicit_only_can_run_out_of_band_bootstrap_migrations tests/test_migrations_runner.py::test_marked_migration_records_its_ledger_entry_in_one_transaction tests/test_eom_render_profile.py::test_eom_runtime_grant_migration_requires_serving_login_target -q`.
- Passed: `python -m py_compile atlas_brain/eom_api/funnel_readiness.py tests/test_eom_render_profile.py tests/test_eom_lead_conversion_integration.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_runtime_grant_migration_requires_serving_login_target tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_migrations_runner.py::test_default_runner_skips_out_of_band_bootstrap_migrations tests/test_migrations_runner.py::test_explicit_only_can_run_out_of_band_bootstrap_migrations -q`.
- Passed: `python -m py_compile atlas_brain/eom_api/funnel_readiness.py atlas_brain/main_eom.py tests/test_eom_lead_conversion.py tests/test_eom_lead_conversion_integration.py tests/test_eom_render_profile.py`.
- Passed: `pytest tests/test_eom_render_profile.py::test_eom_funnel_startup_preflight_checks_handoff_schema tests/test_eom_render_profile.py::test_eom_runtime_grant_migration_requires_serving_login_target -q`.
- Could not run locally: `pytest tests/test_eom_lead_conversion.py::test_enabled_full_atlas_funnel_requires_authoritative_data_store` because this minimal EOM venv lacks optional full-app dependency `numpy`.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 8 |
| `.github/workflows/atlas_invoicing_checks.yml` | 22 |
| `atlas_brain/eom_api/config.py` | 14 |
| `atlas_brain/eom_api/funnel_database.py` | 145 |
| `atlas_brain/eom_api/funnel_readiness.py` | 183 |
| `atlas_brain/main.py` | 145 |
| `atlas_brain/main_eom.py` | 139 |
| `atlas_brain/pipelines/llm.py` | 4 |
| `atlas_brain/services/base.py` | 33 |
| `atlas_brain/storage/migrations/356_eom_customer_handoff_runtime_grants.sql` | 127 |
| `atlas_brain/storage/migrations/__init__.py` | 23 |
| `extracted_llm_infrastructure/pipelines/llm.py` | 4 |
| `extracted_llm_infrastructure/services/base.py` | 3 |
| `plans/PR-EOM-Funnel-Render-Env.md` | 404 |
| `render.eom.yaml` | 8 |
| `tests/test_eom_lead_conversion.py` | 18 |
| `tests/test_base_model_service_torch_optional.py` | 68 |
| `tests/test_eom_lead_conversion_integration.py` | 156 |
| `tests/test_eom_render_profile.py` | 654 |
| `tests/test_migrations_runner.py` | 38 |
| `tests/unit_gate_baseline.txt` | 4 |
| **Total** | **2200** |
