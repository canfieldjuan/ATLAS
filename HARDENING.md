# HARDENING.md

Park non-blocking hardening discoveries here when they are not required for the
current thin slice to function. Newest entries go first.

Do not use this file to defer issues that break the slice's real flow, AGENTS
contract, tests, CI, security, or data truthfulness. Those must be fixed inline
or the slice must stop.

When starting a slice, scan this file for entries touching the same ownership
lane or files. Fix only entries required for that slice to function; otherwise
leave them parked and mention the reason in the plan's Deferred section if they
were considered. Periodically drain stale entries or promote them into the debt
register under `docs/technical-debt/`.

## Entry Format

```md
## YYYY-MM-DD

### <short title>
- File/location:
- Description:
- Why it matters:
- Effort: S / M / L
- Category: correctness / polish / tech-debt / security
- Owner/session:
- Found during:
```

## Parked Items

## 2026-05-23

### FILECONCURRENCY-2 - Uploaded-file imports need cross-process admission
- File/location: Hosted `/content-ops/ingestion/files/import` deployment topology; in-process runner lives at `scripts/smoke_content_ops_ingestion_file_route_inprocess_load.py`.
- Description: The shared-pool route gate and in-process load runner prove one-router admission, but multi-worker or multi-process deployments can still admit one gate window per process unless backed by a distributed queue, shared semaphore, or job boundary.
- Why it matters: Production deployments with multiple app workers can multiply the configured import concurrency and pressure Postgres even though each worker is locally bounded.
- Effort: L
- Category: correctness
- Owner/session: content-ops/backend-file-ingestion-validation
- Found during: PR-Content-Ops-File-Cross-Process-Hardening-Register

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
