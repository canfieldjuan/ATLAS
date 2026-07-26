# PR-EOM-Render-Profile

## Why this slice exists

The operator asked to plan the Render-hosting arc up front because the EOM time
tracker is already on Render and Atlas contains the Postgres-backed receivables
logic that the time tracker should eventually consume. This slice is the first
bounded proof: create an Atlas API boot profile that can run as a separate
Render private service without importing the full Atlas app, local-model stack,
B2B routes, Content Ops routes, voice startup, or autonomous scheduler surfaces.

This PR intentionally does not deploy, connect production traffic, select the
target database, or change EOM customer/onboarding behavior. It only makes the
smallest bootable Atlas profile that future slices can wire to Render and the
existing EOM time tracker.

The diff is over the 400 LOC soft cap because the first bootable slice needs a
copied slim HTTP route surface for the existing receivables contract plus the
required Atlas plan/test artifacts. The scope is still indivisible for this PR:
the import boundary, route surface, candidate Render startup command, and
reachability test have to land together for the proof to mean anything.

### Problem-derived contract

- Root cause: The deployable Atlas entrypoint is too broad for an EOM private
  service proof. Importing the normal Atlas app/router path brings unrelated
  API surfaces and startup/config/module chains into scope, including local
  model and heavy service-package imports that are unsuitable for a small Render
  private service.
- Correct fix must touch/change: Add a separate EOM FastAPI entrypoint; add a
  small EOM settings/auth surface for receivables service-to-service auth; expose
  only liveness plus receivables endpoints; stop `atlas_brain.services` package
  import from eagerly loading the LLM/model registry stack for unrelated service
  submodules; provide a non-authoritative Render Blueprint candidate and minimal
  dependency file; and add focused tests that prove the slim app imports without
  loading the full Atlas API/config/model stack and that liveness is database
  independent.
- Must not change: Do not modify the existing Atlas `atlas_brain.main` app, the
  existing `atlas_brain.api` aggregate, current EOM time tracker deployment,
  existing callers of `atlas_brain.services.receivables`, customer-facing
  onboarding flow, Stripe enforcement timing, production Render wiring, database
  schema/migrations, or any open Atlas PR lane.

## Scope (this PR)

Ownership lane: eom/render-profile
Slice phase: vertical slice

1. Add a new `atlas_brain.main_eom` application profile that mounts only
   `/api/v1/ping` and the EOM receivables router.
2. Add an EOM-scoped config/auth layer so the slim profile does not instantiate
   Atlas's full application settings.
3. Make `atlas_brain.services` lazy enough that importing the existing
   receivables service submodule does not load the LLM/model stack, while
   preserving direct `from atlas_brain.services import llm_registry`
   compatibility.
4. Add `requirements.eom.txt` and `render.eom.yaml` as reviewable candidates,
   with DB disabled by default until the follow-up database slice chooses the
   Render Postgres mapping.
5. Add focused tests proving the EOM app route surface and import boundary.

### Review Contract

- Acceptance criteria:
  - `tests/test_eom_render_profile.py::test_eom_profile_import_does_not_load_full_api_package`
    proves importing `atlas_brain.main_eom` leaves `atlas_brain.api`,
    `atlas_brain.config`, `atlas_brain.reasoning`,
    `atlas_brain.services.llm`, `atlas_brain.services.embedding`, `torch`, and
    `pynvml` unloaded, and that B2B/Content Ops paths are not mounted.
  - `tests/test_eom_render_profile.py::test_services_package_keeps_llm_registry_compatibility`
    proves the lazy services package still supports direct
    `from atlas_brain.services import llm_registry` callers and preserves
    registered backend visibility.
  - `tests/test_eom_render_profile.py::test_direct_registry_import_preserves_registered_llm_backends`
    proves in a subprocess that direct `atlas_brain.services.registry`
    consumers still see the registered LLM backend list instead of an empty
    registry.
  - `tests/test_eom_render_profile.py::test_deferred_registry_registration_waits_for_concurrent_first_callers`
    proves concurrent first callers wait for deferred registry registration to
    complete and the loader runs once.
  - `tests/test_eom_render_profile.py::test_eom_env_loader_preserves_process_env_and_local_precedence`
    proves `.env.local` overrides `.env`, cross-file interpolation can see
    earlier `.env` values, and a real process environment value still wins over
    both local files.
  - `tests/test_eom_render_profile.py::test_eom_profile_ping_is_database_independent`
    proves the real FastAPI app responds to `GET /api/v1/ping` with
    `{"status": "ok", "profile": "eom"}` while DB persistence and receivables
    auth are disabled.
  - `tests/test_eom_render_profile.py::test_eom_profile_reaches_receivables_ready_through_real_app`
    proves the real `atlas_brain.main_eom:app` serves an authenticated
    `/api/v1/receivables/ready` request and returns the observable receivables
    service result.
  - `tests/test_eom_render_profile.py::test_eom_receivables_startup_rejects_unsafe_enabled_tokens`
    proves the new EOM auth boundary rejects missing, placeholder, 23/24
    character, and predictable repeated-character startup tokens when the
    receivables API is enabled.
  - `tests/test_eom_render_profile.py::test_eom_receivables_startup_rejects_repeated_generated_token_shapes`
    proves repeated-character payloads across the allowed token alphabet are
    rejected instead of closed over a finite denylist.
  - `tests/test_eom_render_profile.py::test_eom_receivables_startup_accepts_generated_service_token`
    proves the generated token helper emits a token accepted by the startup
    validator.
  - `tests/test_eom_render_profile.py::test_all_eom_receivables_routes_require_service_auth`
    proves every route on the slim EOM receivables router carries the EOM
    service-auth dependency.
  - `tests/test_eom_render_profile.py::test_eom_receivables_ready_route_is_fail_closed`
    proves the route boundary returns 401 for missing/invalid bearer tokens,
    200 for the configured bearer token, and 503 when the API is disabled.
  - `tests/test_eom_render_profile.py::test_eom_lifespan_closes_database_when_migration_startup_fails`
    proves a migration startup failure after database initialization still closes
    the database pool.
  - `render.eom.yaml` is deliberately not named render dot yaml, uses
    `atlas_brain.main_eom:app`, leaves secrets as `sync: false`, and keeps
    `ATLAS_DB_ENABLED=false` pending the database-mapping slice.
  - `atlas_brain/main_eom.py` includes only the EOM receivables router under
    `/api/v1` and does not import `atlas_brain.main` or `atlas_brain.api`.
- Reachability proof: `TestClient(app).get("/api/v1/ping")` and an
  authenticated `TestClient(app).get("/api/v1/receivables/ready")` exercise the
  real `atlas_brain.main_eom:app` object and assert observable HTTP JSON
  results.
- Affected surfaces: new EOM API profile, EOM API auth/config modules, services
  package import side effects, slim receivables route copy, candidate Render
  Blueprint, minimal EOM requirements, and focused tests.
- Risk areas: import-chain regressions, accidental public/full-Atlas route
  exposure, unsafe secret defaults, premature DB/migration activation,
  dependency bloat, and duplicated receivables route transport logic that must
  be consolidated in a later slice.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Files touched

- `atlas_brain/eom_api/__init__.py`
- `atlas_brain/eom_api/auth.py`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/receivables.py`
- `atlas_brain/main_eom.py`
- `atlas_brain/services/__init__.py`
- `atlas_brain/services/registry.py`
- `plans/PR-EOM-Render-Profile.md`
- `render.eom.yaml`
- `requirements.eom.txt`
- `tests/test_eom_render_profile.py`

## Mechanism

`atlas_brain.main_eom` is a separate FastAPI entrypoint. It merges `.env` and
`.env.local` so `.env.local` wins for local defaults, lets `.env.local`
interpolate earlier `.env` values, while real process environment variables keep
final authority, configures logging from EOM-only settings, validates the
receivables service token on startup when the receivables API is enabled,
initializes the shared database pool only when `ATLAS_DB_ENABLED=true`, keeps
database cleanup active across startup/migration failures, and mounts only the
EOM receivables router under `/api/v1`.

The new `atlas_brain.eom_api` package owns the slim HTTP/auth surface. Its
config module defines only the runtime fields needed by this profile, avoiding
Atlas's full app settings singleton. Its auth module fails closed when
receivables are enabled with a missing, placeholder, too-short, or predictable
token, and exposes a generated-token helper for operator secret creation.

`atlas_brain.services.__init__` now loads protocol, registry, embedding, and
reminder exports lazily. The registry itself defers implementation registration
until callers ask for available/active implementations, and serializes
concurrent first registration attempts with a condition, so both
`from atlas_brain.services import llm_registry` and direct
`atlas_brain.services.registry` consumers preserve the old public API behavior.
Unrelated submodules such as `atlas_brain.services.receivables` no longer pay
that cost during package import.

`render.eom.yaml` is a candidate Blueprint, not live auto-sync configuration. It
uses a private-service shape and starts `uvicorn atlas_brain.main_eom:app`.
Database persistence is disabled by default so a first boot cannot accidentally
attempt local Postgres or run migrations before Slice B selects the target
Render database mapping.

## Intentional

- The Render candidate is named `render.eom.yaml`, not render dot yaml, so this
  branch cannot accidentally change live Render infrastructure by convention.
- `ATLAS_DB_ENABLED` is false in the candidate until Slice B chooses
  `DATABASE_URL`/explicit Render env mapping and the target Postgres.
- The existing `atlas_brain.services.receivables` domain service remains the
  source of behavior. This PR duplicates only the FastAPI route transport so the
  EOM profile can avoid importing `atlas_brain.api`.
- The PR does not add Stripe card-on-file or funnel-state behavior; those are
  separate business-flow slices.

## Deferred

- Slice B: add Render-aware DB config (`DATABASE_URL` parsing or explicit env
  mapping tests), pick the target Postgres, and only then enable DB persistence
  for the candidate service.
- Slice C: verify private-service connectivity from the existing EOM time
  tracker, rotate/generate `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN`, and run
  the receivables ready/open-invoices/payment mutation contract against staging
  data.
- Slice D: extract lead/funnel API state without tying onboarding authority to
  Google Ads.
- Slice E: add Stripe customer/payment-method setup after first-clean completion,
  not before.
- Slice F: consolidate/deprecate duplicated receivables route transport and old
  local/manual paths after callers are proven migrated.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_eom_render_profile.py -q` -- 13 passed, 1 warning.
- `python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests
  --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json
  --min-score 8` plus the CI-sensitive storage globs -- ratchet gate passed
  with no new brittleness above baseline.
- Compileall passed for `atlas_brain/main_eom.py`, `atlas_brain/eom_api`,
  `atlas_brain/services/__init__.py`, `atlas_brain/services/registry.py`, and
  `tests/test_eom_render_profile.py`.
- Direct import probe:
  `python - <<'PY' ... importlib.import_module('atlas_brain.main_eom') ... PY`
  confirmed `atlas_brain.api`, `atlas_brain.config`,
  `atlas_brain.reasoning`, `atlas_brain.services.llm`,
  `atlas_brain.services.embedding`, `torch`, and `pynvml` were not loaded.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/__init__.py` | 8 |
| `atlas_brain/eom_api/auth.py` | 112 |
| `atlas_brain/eom_api/config.py` | 60 |
| `atlas_brain/eom_api/receivables.py` | 341 |
| `atlas_brain/main_eom.py` | 127 |
| `atlas_brain/services/__init__.py` | 36 |
| `atlas_brain/services/registry.py` | 48 |
| `plans/PR-EOM-Render-Profile.md` | 237 |
| `render.eom.yaml` | 30 |
| `requirements.eom.txt` | 5 |
| `tests/test_eom_render_profile.py` | 367 |
| **Total** | **1371** |
