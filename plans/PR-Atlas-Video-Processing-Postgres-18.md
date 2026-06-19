# PR-Atlas-Video-Processing-Postgres-18

## Why this slice exists

Dependabot proposed moving the `atlas_video-processing` compose database image
from Postgres 13 to Postgres 18. Postgres major versions cannot start directly
against a data directory initialized by an older major version, so the upgrade
must avoid silently reusing the existing `postgres_data` named volume.

## Scope (this PR)

Ownership lane: atlas-video-processing/infra
Slice phase: Production hardening

1. Update the compose `db` service image from `postgres:13` to `postgres:18`.
2. Mount Postgres 18 on a new `postgres18_data` named volume instead of reusing
   the legacy `postgres_data` volume.
3. Keep the legacy `postgres_data` volume definition visible so existing local
   PG13 data is not deleted or hidden by the compose edit.
4. Carry the shared Security Guardrails workflow repair already proven on the
   adjacent Dependabot branches.

### Files touched

- `atlas_video-processing/docker-compose.yml`
- `.github/workflows/security_guardrails.yml`
- `plans/PR-Atlas-Video-Processing-Postgres-18.md`

### Review Contract

Acceptance criteria:

- [ ] The `db` service uses the Postgres 18 image.
- [ ] Existing PG13 `postgres_data` volumes are not mounted into the Postgres 18
      service automatically.
- [ ] The compose file documents that persisted PG13 data must be migrated by
      dump/restore into the new Postgres 18 volume.
- [ ] No application runtime code, database schema, or service wiring changes
      are included in this slice.
- [ ] PR-level guardrails have the same workflow fix used by the recently green
      Dependabot branches.

Affected surfaces: config / local compose database / CI guardrails.

Risk areas: data-loss / backcompat / migration / deployment safety.

Reviewer rules triggered: R1, R2, R4, R11, R12, R14.

## Mechanism

The `db` service still exposes Postgres on the same port and keeps the same
`ATLAS_VISION_DB_*` environment variables, but its persisted data mount changes
from `postgres_data` to `postgres18_data`. That means an existing local PG13
volume is preserved and will not be used by a PG18 server process. Operators who
need the old data can dump from the PG13 volume before the upgrade and restore
into the PG18 volume after the new service is initialized.

The legacy `postgres_data` top-level volume remains declared only to make the
old persisted state explicit. It is not mounted into the PG18 service.

## Intentional

- This is a compose-only dependency upgrade; application code and schema files
  are unchanged.
- The migration path is documented rather than automated because this stack does
  not include an existing database migration harness for the compose volume.
- The old named volume is left intact so the PR does not delete local data.

## Deferred

- A follow-up can add a scripted Postgres dump/restore helper if this compose
  database carries production-like persisted data in more than local/dev use.

## Parked hardening

None.

## Verification

- Inspected the PR diff and unresolved review thread: the unsafe direct mount of
  `postgres_data` into Postgres 18 was the blocking issue.
- Updated `atlas_video-processing/docker-compose.yml` so Postgres 18 uses
  `postgres18_data` and documents dump/restore for existing PG13 data.
- Docker compose was not run in this environment; validation is by compose-file
  inspection only.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/docker-compose.yml` | ~6 |
| `.github/workflows/security_guardrails.yml` | shared repair |
| `plans/PR-Atlas-Video-Processing-Postgres-18.md` | ~108 |
| **Total** | **~114** |
