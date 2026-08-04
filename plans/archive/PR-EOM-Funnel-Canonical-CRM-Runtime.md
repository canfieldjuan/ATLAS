# PR-EOM-Funnel-Canonical-CRM-Runtime

## Why this slice exists

Issue canfieldjuan/ATLAS#2254 is at the enable-and-wire stage for the EOM
Leads tab. The website/tracker side is already in flight, and #2257 mounted the
private funnel routes in the slim Render profile. The remaining code-owned risk
is that the slim profile's primary `ATLAS_DB_CONNECTION_STRING` is still the
general EOM profile database slot, while the funnel routes must read and update
the authoritative Atlas CRM contact database that public lead intake writes.
Serving the funnel against a separate or newly-created Render database would
make real handoffs miss submitted leads.

Diff-budget override: this slice changes a deployment-critical database
boundary, so the config slots, startup admission, dedicated pool lifecycle,
route dependency, CI path enrollment, and real-app reachability/lifecycle
tests are one reviewable safety proof. Splitting would either leave the route
reachable on the wrong pool or add an unexercised database knob.

### Problem-derived contract

- Root cause: the slim EOM funnel runtime has no code-owned separation between
  the profile's default database and the canonical CRM database that owns EOM
  leads. Current funnel readiness and route dependencies use the generic Atlas
  DB pool, so enabling the Render profile can be configured against a database
  that has the schema but not the real public-intake contacts.
- Correct fix must touch/change: add a typed `ATLAS_EOM_FUNNEL_` canonical CRM
  DSN setting; add an explicit `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`
  profile admission bit; initialize/close a dedicated funnel CRM pool only when
  the funnel API is enabled; run the existing funnel readiness guard against
  that dedicated pool; install an app-scoped funnel CRM provider in the slim
  `main_eom` app that is bound to that pool while leaving the shared router's
  full-app fallback on the existing CRM provider; declare the new fail-closed
  Render env slots; and add focused tests for defaults, enabled admission,
  real-app route reachability, and shutdown cleanup.
- Must not change: the public website lead form, Website issue #59 UI shape,
  tracker endpoints, receivables auth/token behavior, receivables service pool,
  full Atlas `atlas_brain.main` funnel behavior, CRM SQL semantics, migrations,
  NocoDB privilege policy, scheduling/customer-onboarding behavior, and any
  unrelated open PR lane.

## Scope (this PR)

Ownership lane: eom-funnel
Slice phase: production hardening

1. Make the slim EOM funnel fail closed unless it has an explicit canonical CRM
   DSN and operator confirmation, then bind only funnel routes/readiness to that
   dedicated CRM pool.
2. Prove the disabled/default profile remains lightweight, enabled startup uses
   the dedicated funnel pool before serving, route calls use the dedicated CRM
   provider, and shutdown still closes both pools in the correct safety order.

### Review Contract

- Acceptance criteria:
  - [ ] `EOMProfileConfig` defaults canonical CRM confirmation to false and
    `EOMFunnelConfig` defaults the dedicated CRM DSN to blank.
  - [ ] `render.eom.yaml` declares `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`
    as `sync: false` and `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=false`,
    while keeping the funnel API disabled by default.
  - [ ] Enabled slim funnel startup rejects before generic DB initialization
    when the canonical confirmation bit is false or the dedicated CRM DSN is
    blank.
  - [ ] Enabled slim funnel startup initializes the generic EOM DB pool, then
    the dedicated funnel CRM pool, then runs the existing funnel readiness
    guard against the dedicated pool before serving routes.
  - [ ] Funnel request handlers obtain a `DatabaseCRMProvider` bound to the
    dedicated funnel pool; receivables and the full Atlas app keep their
    existing generic CRM/DB behavior.
  - [ ] Shutdown closes the dedicated funnel pool and still closes the generic
    EOM DB pool if dedicated close raises.
- Reachability proof: the real `atlas_brain.main_eom.app` handles
  `/api/v1/eom-funnel/customer-handoffs` in a subprocess with no dependency
  overrides, and the observed response plus captured call proves the route uses
  the dedicated funnel CRM provider.
- Affected surfaces: `render.eom.yaml`, `atlas_brain/eom_api/config.py`,
  a new slim-only funnel database adapter, `atlas_brain/eom_api/funnel.py`,
  `atlas_brain/main_eom.py`, and focused EOM render/profile tests.
- Risk areas: wrong authoritative DB, secret/config drift, startup ordering,
  app-scoped route dependency binding, shutdown cleanup, disabled/default compatibility,
  and CI path enrollment.
- Reviewer rules triggered: R1, R2, R3, R4, R5, R6, R10, R11, R12, R14.

### Boundary-change enumeration

Required when this diff changes a guard, validator, normalizer, resolver,
router/classifier, or admission boundary. Name each changed boundary path or
seam in the enumeration; otherwise write "N/A - no boundary change."

- Boundary path/seam: slim EOM funnel startup admission, funnel CRM dependency,
  and dedicated database pool lifecycle.
- Replaced-path behaviors: enabled slim funnel no longer uses the generic Atlas
  DB pool for funnel readiness or CRM route work; disabled funnel still performs
  no dedicated funnel DB work.
- Guard-relevant fields: `ATLAS_EOM_FUNNEL_API_ENABLED`,
  `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256`,
  `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`, and
  `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`.
- Caller x input shape: tracker service-authenticated calls to
  `/api/v1/eom-funnel/leads` and `/api/v1/eom-funnel/customer-handoffs`.
- Closure declaration: the boundary/caller/field inventory above is CLOSED,
  ENUMERATED from this slice's changed slim EOM runtime surfaces
  (`render.eom.yaml`, `atlas_brain/eom_api/config.py`,
  `atlas_brain/eom_api/funnel.py`, `atlas_brain/eom_api/funnel_database.py`,
  and `atlas_brain/main_eom.py`), and outside that set the safe/default
  behavior is no new slim-funnel behavior: the funnel remains disabled unless
  explicitly enabled, unconfigured canonical CRM inputs fail startup before DB
  work, and the shared full-Atlas router falls back to its existing CRM
  provider.

### Deployed-config probing

Required for guard, validator, resolver, admission-boundary, or env/config
fallback changes; otherwise write "N/A - no guard/config boundary change."

- Deployed/default config values: funnel API remains disabled by default;
  canonical CRM confirmation defaults false; dedicated funnel CRM DSN defaults
  blank.
- Explicit value probe: enabled funnel with a generated token digest,
  canonical confirmation true, and nonblank DSN initializes the dedicated
  funnel pool and reaches the real route.
- Absent value probe: enabled funnel with missing confirmation or DSN fails
  before database initialization.
- Default-session/default-context probe: disabled funnel imports and ping remain
  database independent.
- Side-effect ordering: startup validates auth/canonical config before DB work,
  initializes generic DB before dedicated funnel DB, checks funnel readiness
  before migrations/serving, and shutdown attempts both pool closes.

### Files touched

- `.github/workflows/atlas_eom_lead_pipeline_checks.yml`
- `.github/workflows/atlas_invoicing_checks.yml`
- `atlas_brain/eom_api/config.py`
- `atlas_brain/eom_api/funnel.py`
- `atlas_brain/eom_api/funnel_database.py`
- `atlas_brain/main_eom.py`
- `plans/PR-EOM-Funnel-Canonical-CRM-Runtime.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`

## Mechanism

Add a small slim-profile-only funnel database module that owns one asyncpg pool
created from `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING`. `main_eom` admits an
enabled funnel only when the generated-token digest is present, the dedicated
DSN is nonblank, and `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED=true`. After
generic EOM DB initialization, it initializes the dedicated funnel pool and
runs the existing readiness SQL against that pool. `main_eom` installs an
app-scoped funnel CRM provider factory on `app.state`, and the shared funnel
router uses that factory when present. The full Atlas app does not set that
state value, so it keeps the existing default CRM provider. Lead review and
handoff SQL in the slim app therefore use the canonical CRM connection while
receivables continues using the generic EOM DB pool.

## Intentional

- This PR does not try to infer "canonical" from a hostname or database name;
  hosted DB identity is an operator deployment fact, so code requires an
  explicit confirmation bit and a nonblank dedicated DSN before enabling.
- This PR does not copy or synchronize contacts between databases. The
  durable model is that the funnel points at the authoritative CRM store.
- The full Atlas app keeps its existing primary-pool funnel behavior because
  it already runs on the authoritative Atlas DB path.

## Deferred

- Live Render/systemd env writes, token generation, service restart/redeploy,
  and portal Leads smoke remain operational follow-up under #2254.
- Out-of-band database role provisioning, if the canonical CRM DSN needs a
  narrower serving role, remains deployment work.
- Website #59 pipeline/product-shape work remains out of this Atlas runtime
  safety slice.

Parking predicate: park only non-blocking polish/tech-debt/hardening findings
that do not break this slice's stated slim EOM funnel runtime safety contract,
startup admission, route binding, or CI/review gates.

Parked hardening: none under that predicate.

## Verification

- Passed: `python -m py_compile atlas_brain/eom_api/config.py atlas_brain/eom_api/funnel.py atlas_brain/eom_api/funnel_database.py atlas_brain/main_eom.py tests/test_eom_render_profile.py`.
- Passed: `python -m pytest tests/test_eom_render_profile.py -q --tb=short -rfE` — 60 passed, 1 warning from `torch`/`pynvml`.
- Passed: `python -m pytest tests/test_eom_lead_conversion.py -q --tb=short -rfE` — 40 passed, 1 warning from `torch`/`pynvml`.
- Passed with local DB-gated skips: `python -m pytest tests/test_eom_lead_conversion_integration.py -q --tb=short -rfE` — 21 skipped, 1 warning because `ATLAS_MIGRATION_TEST_DATABASE_URL` is not configured locally.
- Passed: plan audits, diff checks, PR body audit, and the managed
  `scripts/push_pr.sh`/`scripts/open_pr.sh` local review wrappers.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_eom_lead_pipeline_checks.yml` | 2 |
| `.github/workflows/atlas_invoicing_checks.yml` | 2 |
| `atlas_brain/eom_api/config.py` | 14 |
| `atlas_brain/eom_api/funnel.py` | 9 |
| `atlas_brain/eom_api/funnel_database.py` | 167 |
| `atlas_brain/main_eom.py` | 29 |
| `plans/PR-EOM-Funnel-Canonical-CRM-Runtime.md` | 204 |
| `render.eom.yaml` | 4 |
| `tests/test_eom_render_profile.py` | 350 |
| **Total** | **781** |
