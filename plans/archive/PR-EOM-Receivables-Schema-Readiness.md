# PR-EOM-Receivables-Schema-Readiness

## Why this slice exists

The EOM Render service is meant to host the slim Atlas receivables API against
its own `atlas-eom-postgres` database, but the deployed profile currently leaves
startup migrations disabled. Turning the existing switch on as-is would ask the
EOM app to run Atlas's entire migration chain, even though the migration runner
documents that the full chain is not fresh-applicable for standalone component
databases.

### Problem-derived contract

- Root cause: The EOM profile has a migration switch, but its startup path calls
  the default full-chain runner. A fresh EOM-only Render database needs the
  tables required by receivables readiness without inheriting Atlas B2B/V2V
  migrations or the known full-chain fresh-database failure mode.
- Correct fix must touch/change: the EOM app startup migration seam, the EOM
  Render blueprint's migration setting, and render-profile tests proving the
  app applies an explicit EOM-safe migration set when configured. The
  per-PR invoicing workflow must also enroll the changed EOM startup/config
  paths and the render-profile tests that prove them.
- Must not change: receivables auth, token format, API enablement, payment
  posting/allocation semantics, customer-facing onboarding shape, main Atlas app
  startup behavior, unrelated open PR lanes, or the existing packaged SQL
  migrations unless a dependency proof requires a minimal schema-safety patch.

## Scope (this PR)

Ownership lane: eom-render/receivables-schema-readiness
Slice phase: Vertical slice

1. Make the EOM app's `ATLAS_EOM_RUN_MIGRATIONS=true` path apply an explicit
   curated schema-readiness migration set instead of Atlas's whole migration
   chain.
2. Flip the EOM Render blueprint to run that curated set at startup while
   keeping the receivables API disabled until the service-token digest is
   provisioned separately.
3. Add tests that lock the curated migration set, the startup call shape, the
   Render blueprint values, and the real migration dependency proof needed for
   `/receivables/ready`.
4. Enroll the EOM startup/config paths and render-profile tests in per-PR
   invoicing CI so this deployed startup seam cannot merge untested.

### Review Contract

- Acceptance criteria:
  - `atlas_brain/main_eom.py` applies a private EOM migration helper from
    startup; that helper calls `run_migrations(..., only=...)` and the `only`
    value is a named, ordered EOM receivables readiness contract.
  - `tests/test_eom_render_profile.py` proves the migration helper calls the
    migration runner with exactly that curated set, and startup still skips
    migrations when the DB pool is uninitialized.
  - `tests/test_eom_render_profile.py` proves `render.eom.yaml` sets
    `ATLAS_EOM_RUN_MIGRATIONS` to `true`, leaves
    `ATLAS_INVOICING_RECEIVABLES_API_ENABLED` as `false`, and keeps
    `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256` configured as a synced
    secret rather than a raw/default token.
  - A real SQL migration test proves the curated set creates every table,
    column, index, and required foreign-key relationship on the tested pre-EOM
    schema.
  - Validation includes `tests/test_eom_render_profile.py`, the targeted
    receivables migration test, `python -m py_compile` for touched Python, and
    `git diff --check`.
- Reachability proof: EOM Render profile starts `uvicorn atlas_brain.main_eom:app`;
  the app lifespan reads `ATLAS_EOM_RUN_MIGRATIONS=true` and invokes the scoped
  migration runner before the `/api/v1/receivables/ready` router is served.
- Affected surfaces: `atlas_brain/main_eom.py`, `render.eom.yaml`,
  `.github/workflows/atlas_invoicing_checks.yml`,
  `tests/test_eom_render_profile.py`, and targeted receivables migration tests.
- Risk areas: fresh-database migration dependencies, avoiding the full Atlas
  migration chain, idempotent startup re-runs, auth/API enablement separation,
  and Render deployed-config drift.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: EOM app lifespan migration seam guarded by
  `EOMProfileConfig.run_migrations` / `ATLAS_EOM_RUN_MIGRATIONS`.
- Replaced-path behaviors: true used to run the full Atlas migration chain;
  true must now run only the named EOM receivables readiness migration set.
  false must still skip migrations.
- Guard-relevant fields: `ATLAS_EOM_RUN_MIGRATIONS`,
  `ATLAS_DB_ENABLED`, `ATLAS_DB_CONNECTION_STRING`,
  `ATLAS_INVOICING_RECEIVABLES_API_ENABLED`,
  `ATLAS_INVOICING_RECEIVABLES_SERVICE_TOKEN_SHA256`.
- Caller x input shape: Render service process starts
  `atlas_brain.main_eom:app` with the deployed `render.eom.yaml` env values.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `render.eom.yaml` must set
  `ATLAS_EOM_RUN_MIGRATIONS=true`, keep `ATLAS_DB_ENABLED=true`, keep the DB
  connection from `atlas-eom-postgres`, keep the receivables API disabled, and
  require a synced digest secret for later API enablement.
- Explicit value probe: test with `ATLAS_EOM_RUN_MIGRATIONS=true` must observe
  the scoped migration runner call.
- Absent value probe: default `EOMProfileConfig.run_migrations` remains false
  outside the deployed blueprint.
- Default-session/default-context probe: EOM profile startup with migrations
  false still initializes and closes DB without invoking migrations.
- Side-effect ordering: startup validates auth config, initializes the DB,
  runs scoped migrations only when the pool is initialized and the flag is true,
  and closes the DB on startup failure.

### Rollback and backout

- If EOM startup fails before any readiness migration is recorded, set
  `ATLAS_EOM_RUN_MIGRATIONS=false` in Render or redeploy the previous blueprint
  revision, then restart the service. The private receivables API remains
  disabled in this slice, so this backout returns the service to schema-idle
  startup without opening the API.
- If startup fails after one or more curated migrations are recorded, first set
  `ATLAS_EOM_RUN_MIGRATIONS=false` or revert the service version to stop repeat
  startup attempts. Keep the additive schema and `schema_migrations` rows in
  place unless a follow-up, database-specific rollback plan proves removal is
  safe; these migrations create/add receivables readiness tables, columns,
  indexes, and foreign keys, and the runner is idempotent on re-entry.
- After disabling the flag, inspect the failed migration name in
  `schema_migrations`/startup logs and either roll forward with a narrow SQL fix
  or cut a dedicated database-repair slice. Do not enable
  `ATLAS_INVOICING_RECEIVABLES_API_ENABLED` until the readiness check is green.

### Closure declaration

Set-valued dependency: `EOM_RECEIVABLES_READINESS_MIGRATIONS`, the migration
inventory that the EOM startup path passes to `run_migrations(..., only=...)`.

1. Membership is **CLOSED** for this slice. The set is finite: it contains the
   packaged Atlas migrations needed for a standalone EOM database to satisfy the
   current receivables readiness contract. Unlisted Atlas migrations are either
   unrelated to the slim EOM receivables API or are out of scope because this
   profile does not mount the full Atlas/B2B/V2V runtime.
2. Membership is **ENUMERATED** in `atlas_brain/main_eom.py`, sourced from two
   canonical surfaces: `ReceivablesService.is_ready()` via
   `_RECEIVABLES_REQUIRED_COLUMNS` / `_RECEIVABLES_REQUIRED_INDEXES`, and the
   SQL prerequisite chain in the packaged migrations. The chain is:
   `345_receivables_event_key_lookup` depends on `payment_events` from
   `344_receivables_payments`; `344_receivables_payments` alters
   `invoice_payments` and references `contacts`; `045_invoices` creates
   `invoice_payments` and references `contacts`; `035_contacts` alters
   `appointments`; `012_appointments` creates `appointments`.
3. Inputs outside the set are **not run** by the EOM profile. That is the safer
   default because the full Atlas migration chain is documented as unsafe for a
   fresh standalone component database and because unrelated Atlas/B2B/V2V
   migrations would pollute the EOM database. If a future receivables readiness
   table/index or SQL prerequisite falls outside the set, the readiness tests and
   `/receivables/ready` fail closed until a follow-up slice expands this
   declaration, the tuple, and the migration proof together.

### Files touched

- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/main_eom.py`
- `plans/PR-EOM-Receivables-Schema-Readiness.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`
- `tests/test_receivables.py`

## Mechanism

Define the EOM receivables readiness migration set at the EOM app boundary and
pass it through a private EOM startup helper to the existing
`run_migrations(..., only=...)` mechanism. That reuses the central migration
runner's advisory-lock and ledger behavior while avoiding unrelated full-chain
migrations. The Render profile can then enable `ATLAS_EOM_RUN_MIGRATIONS`
without starting B2B/V2V schema work in the EOM database.

## Intentional

- Do not enable the private receivables API in this slice; token digest
  provisioning and EOM time-tracker runtime cutover stay separate.
- Do not run the whole Atlas migration chain from the EOM profile; the existing
  migration runner documents that this is unsafe for fresh standalone databases.
- Do not rewrite the SQL migration history; this slice uses the smallest
  explicit dependency set needed for receivables readiness.

## Deferred

- Provision/rotate the EOM receivables service-token digest and enable the
  private API after schema readiness is deployed.
- Remove or deprecate any old EOM/Atlas receivables compatibility code once the
  Render-backed path is carrying production traffic.

## Parked hardening

Parking predicate: defer only hardening unrelated to safely enabling this EOM
startup migration boundary. Findings that affect migration membership,
fresh-database schema integrity, required foreign keys/indexes, Render rollback,
API-disabled separation, or per-PR CI enrollment are in scope for this slice.

Parked hardening under that predicate: none.

## Verification

- Command: python -m py_compile atlas_brain/main_eom.py tests/test_eom_render_profile.py tests/test_receivables.py — passed locally.
- Command: git diff --check — passed locally.
- Command: python -m pytest tests/test_eom_render_profile.py -q — passed locally.
- Command: python -m pytest tests/test_receivables.py -q — passed locally with live-Postgres tests skipped when `ATLAS_RECEIVABLES_TEST_DATABASE_URL` is unset.
- CI command enrolled in `.github/workflows/atlas_invoicing_checks.yml` with `ATLAS_RECEIVABLES_TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/atlas_receivables_test`: python -m pytest tests/test_eom_render_profile.py tests/test_receivables.py tests/test_invoice_repository.py -q.
- Command: python scripts/maturity_sweep.py atlas_brain/storage --tests-root tests --baseline tests/maturity_sweep/baseline_atlas_brain_storage.json --min-score 8 --sensitive-glob '**/auth/**' --sensitive-glob '**/auth*' --sensitive-glob '**/webhook*' --sensitive-glob '**/webhooks/**' --sensitive-glob '**/*webhook*/**' --sensitive-glob '**/payment*' --sensitive-glob '**/invoicing/**' --sensitive-glob '**/*invoice*' --sensitive-glob '**/*deletion*' --sensitive-glob '**/delete*/**' --sensitive-glob 'atlas_brain/security/**' --sensitive-glob 'atlas_brain/storage/**' — ratchet gate passed locally.
- Command: python scripts/audit_plan_doc.py plans/PR-EOM-Receivables-Schema-Readiness.md — passed locally.
- Command: python scripts/check_deployed_config_probing.py --base origin/main — passed locally.
- Command: bash scripts/local_pr_review.sh --allow-dirty --current-pr-body-file tmp/pr-body-eom-receivables-schema-readiness.md — passed locally.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_invoicing_checks.yml` | 7 |
| `atlas_brain/main_eom.py` | 40 |
| `plans/PR-EOM-Receivables-Schema-Readiness.md` | 223 |
| `render.eom.yaml` | 2 |
| `tests/test_eom_render_profile.py` | 57 |
| `tests/test_receivables.py` | 203 |
| **Total** | **532** |
