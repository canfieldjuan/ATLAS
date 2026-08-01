# PR-EOM-Render-Slim-Profile

## Why this slice exists

The operator selected the "slim Render EOM API profile" lane after inspecting
the current EOM deployment split.  The code already has a slim
`atlas_brain.main_eom` app and a Render candidate, but that profile still only
serves receivables while the EOM funnel/customer-handoff API remains reachable
only through the full Atlas app.

Diff-budget overage rationale: this slice is intentionally oversized because
the slim route mount, shared datastore guard extraction, Render env declaration,
raw-token admission guard, CI enrollment, and real-entrypoint tests are one
private API deployment boundary.  Splitting them would either expose an
unproven mounted funnel route, duplicate the datastore predicate, or land a
digest-only Render contract without the admission and CI proof that makes that
contract safe.

### Problem-derived contract

- Root cause: Atlas has EOM business-system APIs split across a full local
  Atlas app and a slim Render candidate.  A correct slim Render profile cannot
  depend on importing the full `atlas_brain.main` app, and it cannot expose a
  funnel/handoff write path without the same token and datastore preflight that
  protects the full app.
- Correct fix must touch/change: Mount the existing `eom-funnel` router in
  `atlas_brain.main_eom`; validate the existing funnel token config at slim-app
  startup when enabled; require the same authoritative CRM/handoff datastore
  guard without importing heavy Atlas surfaces; update the Render candidate to
  declare the funnel enable flag and digest-only secret placeholder; reject raw
  funnel bearer material on the Atlas service side; add tests that prove route
  reachability, fail-closed startup/request behavior, shared datastore guard
  behavior, raw-token rejection, Render env coverage, and CI path-filter
  enrollment for the newly mounted funnel router, auth, config, shared guard, and
  transitive service dependencies.
- Must not change: Do not change customer-facing website/portal copy, time
  tracker UI, checkout/card policy, Google Ads integration, live Render
  resources, local systemd services, full Atlas router enrollment beyond sharing
  the datastore guard helper, or unrelated open PR lanes.

## Scope (this PR)

Ownership lane: eom/render-slim-profile
Slice phase: production hardening

1. Make `atlas_brain.main_eom:app` include the existing private
   `/api/v1/eom-funnel/*` router and run the funnel startup guard only when
   `ATLAS_EOM_FUNNEL_API_ENABLED=true`.
2. Keep `render.eom.yaml` as a reviewed/manual Render candidate, but add the
   funnel enable flag and `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` `sync: false`
   placeholder alongside receivables.
3. Centralize the funnel datastore guard in a slim-safe helper so the full app
   and slim app use the same SQL readiness predicate without importing the full
   app from the slim profile.
4. Add regression coverage proving the slim app exposes the funnel routes
   without loading heavy Atlas modules and fails closed before any CRM write
   when config or datastore readiness is missing.

### Review Contract

- Acceptance criteria:
  - `atlas_brain.main_eom:app` exposes `/api/v1/eom-funnel/leads` and
    `/api/v1/eom-funnel/customer-handoffs` while the subprocess import proof
    still shows full Atlas/heavy model modules are not loaded.
  - The slim app validates `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` when
    `ATLAS_EOM_FUNNEL_API_ENABLED=true`, accepts a generated digest, and rejects
    missing/malformed/placeholder digests before request handling and before
    database initialization or startup migrations can mutate state.
  - With a valid funnel digest but missing CRM/handoff datastore readiness, the
    slim app rejects startup after DB pool initialization and before startup
    migrations can mutate state.
  - The slim settings object rejects raw `ATLAS_EOM_FUNNEL_SERVICE_TOKEN`
    material before projecting runtime config, matching the digest-only Render
    boundary.
  - The slim app requires an initialized database and the existing CRM lifecycle,
    handoff, `atlas_eom_handoff_owner`, and `atlas_nocodb` readiness predicate
    before enabling funnel writes.
  - The disabled/default funnel route returns 503 rather than public access, and
    missing/invalid bearer or actor headers reject before any CRM dependency
    mutation.
  - `render.eom.yaml` contains only digest placeholders for EOM private API
    tokens, declares funnel enablement explicitly, and does not add raw bearer
    token env vars.
  - The EOM lead-pipeline and invoicing GitHub Actions path filters include the
    newly mounted funnel dependency set:
    `atlas_brain/eom_api/config.py`, `atlas_brain/eom_api/funnel.py`,
    `atlas_brain/eom_api/funnel_auth.py`, and
    `atlas_brain/eom_api/funnel_store.py`, plus the router's service
    dependencies `atlas_brain/services/crm_provider.py` and
    `atlas_brain/services/eom_lead_conversion.py`, so later changes to the
    router, auth/config admission, shared guard, or mounted route service layer
    trigger the focused EOM CI suites.
- Reachability proof: `tests/test_eom_render_profile.py` imports the real
  `atlas_brain.main_eom:app` in a subprocess and exercises real route
  transport through `/api/v1/eom-funnel/customer-handoffs` with env-projected
  funnel settings.
- Affected surfaces: `atlas_brain.main_eom`, the shared EOM funnel datastore
  guard helper, `atlas_brain.main`'s full-app guard wrapper, `render.eom.yaml`,
  EOM workflow path filters, and the EOM render-profile/funnel tests.
- Risk areas: Slim import isolation, token fail-closed behavior, database
  readiness ordering, Render env defaults, and shared helper compatibility with
  the existing full-app tests.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R8, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: `atlas_brain.main_eom` router admission.
  - Replaced-path behaviors: preserved for ping and receivables; intentionally
    changed to include the existing `eom-funnel` router under `/api/v1`.
  - Guard-relevant fields: route path, router prefix, FastAPI dependency chain,
    and the profile's import graph.
  - Caller x input shape: EOM time tracker/private service calls to
    `GET /api/v1/eom-funnel/leads` and
    `POST /api/v1/eom-funnel/customer-handoffs` are admitted to the existing
    router; browser/public callers still hit the same bearer-token dependency
    and fail closed without service auth.
- Boundary path/seam: `atlas_brain.main_eom` startup admission for funnel API.
  - Replaced-path behaviors: previously the slim profile did not evaluate
    funnel startup readiness; intentionally changed so disabled remains a no-op,
    enabled validates token digest before DB work, and enabled validates
    CRM/handoff datastore readiness after DB initialization but before startup
    migrations.
  - Guard-relevant fields: `ATLAS_EOM_FUNNEL_API_ENABLED`,
    `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256`,
    `ATLAS_EOM_FUNNEL_SERVICE_TOKEN`, `ATLAS_DB_ENABLED`, initialized DB pool
    state, and the SQL readiness predicate for `contacts`,
    `eom_lead_lifecycle_events`, `eom_customer_handoffs`,
    `atlas_eom_handoff_owner`, and `atlas_nocodb`.
  - Caller x input shape: Render/process startup with funnel disabled, enabled
    with generated digest and ready DB, enabled with missing digest, enabled
    with disabled DB, enabled with uninitialized DB, and enabled with schema or
    privilege predicate false before startup migrations.
- Boundary path/seam: `EOMFunnelConfig` raw bearer settings-source admission.
  - Replaced-path behaviors: previously the slim EOM settings projection ignored
    raw `ATLAS_EOM_FUNNEL_SERVICE_TOKEN` material and could construct an Atlas
    service config while the caller-side bearer was present.  This is
    intentionally changed so any nonblank raw funnel token in an admitted
    settings source is rejected during settings construction, before lifespan
    startup or request handling; blank or absent raw-token values preserve the
    existing digest-only settings projection.
  - Guard-relevant fields: settings source (`os.environ`, `.env`, `.env.local`),
    raw-token key spelling/case, raw-token value blankness, `api_enabled`, and
    `service_token_sha256`.
  - Caller x input shape: process environment with raw token present while the
    funnel is enabled or disabled is rejected; process environment with raw token
    blank/absent is preserved; `.env` with raw token present while enabled or
    disabled is rejected; `.env` with raw token blank/absent is preserved;
    `.env.local` with raw token present while enabled or disabled is rejected;
    `.env.local` with raw token blank/absent is preserved.  Digest-only enabled
    and disabled configs remain admitted for the later startup/request guards.
- Boundary path/seam: shared funnel datastore guard helper.
  - Replaced-path behaviors: full-app behavior is preserved through a wrapper;
    slim app now calls the same helper through its own DB-pool import instead of
    importing the full app.
  - Guard-relevant fields: `config.api_enabled`, database-enabled flag,
    `pool.is_initialized`, and the SQL readiness predicate result.
  - Caller x input shape: full app startup and slim app startup both pass their
    own `get_db_pool` callable and receive the same accept/reject decisions.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: `render.eom.yaml` sets
  `ATLAS_DB_ENABLED=true`, maps `ATLAS_DB_CONNECTION_STRING` from
  `atlas-eom-postgres`, sets `ATLAS_EOM_RUN_MIGRATIONS=true`, keeps
  `ATLAS_INVOICING_RECEIVABLES_API_ENABLED=false`, and this slice keeps the new
  `ATLAS_EOM_FUNNEL_API_ENABLED=false` with only
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` as a `sync: false` digest
  placeholder; no repo-owned Render config declares raw
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN`.
- Explicit value probe: subprocess/import tests set
  `ATLAS_EOM_FUNNEL_API_ENABLED=true` plus a generated
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256`, then call the real
  `atlas_brain.main_eom:app` route and observe authenticated handoff behavior.
- Raw-token source probe: tests exercise raw
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN` admission from process environment, `.env`,
  and `.env.local`, with enabled and disabled funnel configs, and prove nonblank
  raw values reject while blank/absent raw-token values remain admitted for
  digest-only projection.
- Absent value probe: tests cover the default/absent
  `ATLAS_EOM_FUNNEL_API_ENABLED` state and assert the route returns 503 rather
  than becoming public.
- Default-session/default-context probe: tests keep `ATLAS_DB_ENABLED=false`
  for import and ping probes to prove the slim profile still boots without
  database work when APIs are disabled.
- Side-effect ordering: startup validates token config before database
  initialization or migrations, then validates datastore readiness after DB pool
  initialization but before startup migrations or serving enabled funnel
  requests; request tests assert missing auth or disabled API stops before the
  CRM dependency records any handoff call.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel_store.py`
- `atlas_brain/main.py`
- `atlas_brain/main_eom.py`
- `plans/PR-EOM-Render-Slim-Profile.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`

## Mechanism

`main_eom` imports the existing funnel router, funnel settings, and auth
validator from the slim-safe `atlas_brain.eom_api` package.  The app includes
that router under `/api/v1`, then extends lifespan startup with a funnel
preflight that is a no-op while the funnel API is disabled and fail-closed when
enabled without safe config or datastore readiness.  The enabled-funnel startup
order is token config, DB pool initialization, datastore readiness, then optional
startup migrations so a rejecting admission does not mutate schema first.

The SQL readiness predicate is moved into an EOM API helper module.  The full
app keeps the old private `_require_eom_funnel_data_store` function as a thin
wrapper so existing full-app callers/tests retain their seam, while the slim app
uses an equivalent wrapper that passes its own `get_db_pool` function.  That
keeps the heavy full-app import graph out of the Render profile.

The Render candidate adds the funnel env vars in the same pattern as
receivables: disabled by default, digest-only secret placeholder, no raw bearer
token on the Atlas service side.  `EOMFunnelConfig` now rejects raw funnel token
env material so an accidental Atlas-side raw bearer fails before request
handling.

## Intentional

- Keep the file named `render.eom.yaml` in this PR.  Promoting it to root
  `render.yaml` or creating/updating Render resources is an operational
  deployment step, not a code-profile safety proof.
- Do not relax the existing full-app funnel datastore guard for a fresh Render
  database.  If the Render database lacks the required roles/schema, the slim
  profile should fail closed until a provisioning slice addresses that directly.
- Do not mount broad Atlas routers or full-app startup in the slim app; the
  point of this profile is to avoid local models, B2B scraping, voice, and
  autonomous scheduler surfaces.

## Deferred

- Render resource creation or blueprint promotion to `render.yaml`.
- Render Postgres role/schema provisioning if the target database does not yet
  satisfy the funnel guard.
- Wiring the deployed time tracker to the new Render-hosted Atlas EOM API base
  URL and bearer tokens.
- Deprecation/removal of old local/Tailscale Atlas EOM endpoints after the
  Render profile is live and verified.

Parked hardening: none.

## Verification

- Completed:
  - `python -m pytest tests/test_eom_render_profile.py -k 'funnel_raw_token or raw_funnel or eom_funnel_runtime_config_rejects_raw_token_env or eom_profile_rejects_raw_funnel_token'`
    — 9 passed, 38 deselected.
  - `python -m pytest tests/test_eom_render_profile.py -k 'datastore_before_migrations or missing_digest_before_db_work'`
    — 2 passed, 46 deselected.
  - `python -m pytest tests/test_eom_render_profile.py tests/test_eom_lead_conversion.py`
    — 88 passed, 1 warning from the existing `torch`/`pynvml` import path.
  - `python -m py_compile atlas_brain/main.py atlas_brain/main_eom.py atlas_brain/eom_api/config.py atlas_brain/eom_api/funnel.py atlas_brain/eom_api/funnel_auth.py atlas_brain/eom_api/funnel_store.py tests/test_eom_render_profile.py tests/test_eom_lead_conversion.py`
    — passed.
  - `python scripts/sync_pr_plan.py plans/PR-EOM-Render-Slim-Profile.md`
    — passed and updated the committed file list/diff-size table.
  - Initial publish wrappers:
    `bash scripts/push_pr.sh /tmp/pr-body-eom-render-slim-profile.md -u origin HEAD`
    and
    `bash scripts/open_pr.sh /tmp/pr-body-eom-render-slim-profile.md --title "Add slim Render EOM funnel profile" --base main`
    — both passed local PR review before opening #2257.
  - Review-fix local checks for the startup-order and CI-filter repair:
    `python -m pytest tests/test_eom_render_profile.py tests/test_eom_lead_conversion.py`
    — 79 passed, 1 warning; `python -m py_compile ...` — passed.
  - Review-fix local plan audits after the raw-token/path-filter repair:
    `python scripts/audit_pr_plan_presence.py origin/main`;
    `python scripts/audit_plan_doc.py plans/PR-EOM-Render-Slim-Profile.md`;
    `python scripts/audit_plan_doc_files_touched.py plans/PR-EOM-Render-Slim-Profile.md origin/main`;
    `python scripts/audit_plan_doc_diff_size.py plans/PR-EOM-Render-Slim-Profile.md origin/main`;
    `python scripts/audit_review_rules_triggered.py origin/main --plan plans/PR-EOM-Render-Slim-Profile.md`
    — all passed.
  - Review-fix local checks for the datastore-before-migrations and service
    dependency CI-filter repair:
    `python -m pytest tests/test_eom_render_profile.py -k 'datastore_before_migrations or missing_digest_before_db_work'`
    — 2 passed, 46 deselected; workflow path sanity check for the mounted
    funnel dependency set — passed; full EOM render/funnel pair — 88 passed,
    1 warning; `python -m py_compile ...` — passed; plan audits — passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `.github/workflows/atlas_invoicing_checks.yml` | 12 |
| `atlas_brain/eom_api/config.py` | 58 |
| `atlas_brain/eom_api/funnel_store.py` | 155 |
| `atlas_brain/main.py` | 146 |
| `atlas_brain/main_eom.py` | 34 |
| `plans/PR-EOM-Render-Slim-Profile.md` | 304 |
| `render.eom.yaml` | 4 |
| `tests/test_eom_render_profile.py` | 504 |
| **Total** | **1219** |
