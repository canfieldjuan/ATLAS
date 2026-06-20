# PR-Atlas-Video-Processing-Postgres-18

## Why this slice exists

Dependabot proposed moving the `atlas_video-processing` compose database from
Postgres 13 to Postgres 18. A major-version Postgres server must not silently
reuse an older PGDATA directory, so this slice isolates PG18 on a new named
volume and documents the dump/restore path for existing PG13 data.

## Scope (this PR)

Ownership lane: atlas-video-processing/infra
Slice phase: Production hardening

1. Update the compose `db` service image from `postgres:13` to `postgres:18`.
2. Mount the Postgres 18 service on a new `postgres18_data` named volume.
3. Keep the legacy `postgres_data` volume declared but unmounted so existing
   PG13 data is preserved and not reused silently.

### Files touched

- `atlas_video-processing/docker-compose.yml`
- `plans/PR-Atlas-Video-Processing-Postgres-18.md`

### Review Contract

Acceptance criteria:

- [ ] The `db` service uses the Postgres 18 image.
- [ ] Existing PG13 `postgres_data` volumes are not mounted into the Postgres 18
      service automatically.
- [ ] The compose file documents that persisted PG13 data must be migrated by
      dump/restore into the new Postgres 18 volume.
- [ ] No application runtime code, database schema, service wiring, workflow, or
      repository-wide guardrail changes are included in this slice.

Affected surfaces: config / local compose database.

Risk areas: data-loss / backcompat / migration / deployment safety.

Reviewer rules triggered: R1, R2, R4, R11, R12, R14.

## Mechanism

The compose `db` service now uses `postgres:18` with a new `postgres18_data`
named volume. The legacy `postgres_data` volume remains declared so existing
PG13 data is visible and preserved, but it is not mounted into the PG18 service.

## Intentional

- Compose-only dependency upgrade; no application runtime code, schema, or
  service wiring changes.
- Migration path is documented rather than automated because this stack does not
  currently include a database volume migration harness.
- The old named volume is left intact to avoid deleting local persisted data.

## Deferred

- Add a scripted Postgres dump/restore helper if this compose database carries
  production-like persisted data in more than local/dev use.
- Parked hardening: none.

## Verification

- Inspected the existing #1633 compose diff and review thread.
- Updated `atlas_video-processing/docker-compose.yml` so Postgres 18 uses
  `postgres18_data` and keeps the legacy `postgres_data` volume unmounted.
- Docker compose was not run in this environment; validation is by compose-file
  inspection plus CI.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_video-processing/docker-compose.yml` | ~6 |
| `plans/PR-Atlas-Video-Processing-Postgres-18.md` | ~74 |
| **Total** | **~80** |
