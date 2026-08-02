# PR-EOM-Funnel-Env-Placeholders

## Why this slice exists

Effingham Office Maids issue #59 needs Atlas-side runtime slots before live
funnel wiring can be configured. This slice deliberately stops at fail-closed
configuration placeholders so follow-up routing, datastore, migration, and CI
hardening work can converge independently.

### Problem-derived contract

- Root cause: the slim EOM Render blueprint and settings surface do not expose
  the funnel enable flag, service-token digest, canonical CRM DSN, or canonical
  confirmation bit needed for later live configuration.
- Correct fix must touch/change: add those env placeholders to `render.eom.yaml`
  and expose matching settings fields in `atlas_brain/eom_api/config.py`.
- Must not change: route mounting, startup behavior, database pools, migrations,
  auth validation, or LLM/runtime infrastructure.

## Scope (this PR)

Ownership lane: eom-funnel
Slice phase: production hardening

1. Add disabled-by-default funnel env placeholders to `render.eom.yaml`.
2. Add matching settings fields to `atlas_brain/eom_api/config.py`.
3. Add focused tests for the new Render keys and settings env projections.

### Review Contract

- Acceptance criteria:
  - [ ] `render.eom.yaml` declares `ATLAS_EOM_FUNNEL_API_ENABLED` as `"false"`.
  - [ ] `render.eom.yaml` declares `ATLAS_EOM_FUNNEL_SERVICE_TOKEN_SHA256` with
    `sync: false`.
  - [ ] `render.eom.yaml` declares `ATLAS_EOM_FUNNEL_DB_CONNECTION_STRING` with
    `sync: false`.
  - [ ] `render.eom.yaml` declares `ATLAS_EOM_CANONICAL_CRM_DATABASE_CONFIRMED`
    as `"false"`.
  - [ ] `atlas_brain/eom_api/config.py` exposes the same values through settings
    fields without changing serving behavior.
- Affected surfaces: EOM Render blueprint and slim EOM settings.
- Risk areas: environment naming drift and accidental live enablement.
- Reviewer rules triggered: R1, R2, R11, R12.

### Files touched

- `atlas_brain/eom_api/config.py`
- `plans/PR-EOM-Funnel-Env-Placeholders.md`
- `render.eom.yaml`
- `tests/test_eom_render_profile.py`

## Mechanism

- The blueprint remains fail-closed because the funnel enable flag and canonical
  confirmation bit default to false.
- Atlas receives only the funnel service-token digest slot; the raw tracker
  bearer remains outside Atlas.
- The dedicated funnel DSN is only a settings placeholder in this slice.

## Intentional

- No route is mounted.
- No database pool is created.
- No migration set is changed.
- No auth validator or startup preflight is changed.

## Deferred

- Slim router mount, datastore preflight, dedicated pool wiring, migration
  354/356 privilege work, domain CI enrollment, and LLM/torch CI fixes move to
  separate bounded PRs.

Parked hardening: none.

## Verification

- Passed: `/tmp/atlas-eom-funnel-env-1785688582/.venv/bin/python -m py_compile atlas_brain/eom_api/config.py`.
- Passed: `/tmp/atlas-eom-funnel-env-1785688582/.venv/bin/python -m pytest tests/test_eom_render_profile.py::test_eom_render_blueprint_maps_database_and_receivables_auth tests/test_eom_render_profile.py::test_eom_funnel_placeholders_load_from_env_and_default_fail_closed -q`.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/eom_api/config.py` | 14 |
| `render.eom.yaml` | 8 |
| `tests/test_eom_render_profile.py` | 54 |
| `plans/PR-EOM-Funnel-Env-Placeholders.md` | 88 |
| **Total** | **164** |
