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

### FILECONCURRENCY-2 - Uploaded-file imports need hosted multiprocess proof
- File/location: Hosted `/content-ops/ingestion/files/import` deployment topology; host provider lives at `atlas_brain/_content_ops_import_admission.py`.
- Description: Atlas now has a Postgres advisory-lock admission provider for shared cross-process capacity, but the remaining production hardening is a live Atlas-mounted multiprocess proof plus durable background job/queue visibility.
- Why it matters: Production deployments need evidence that the mounted host route stays bounded across workers and enough job visibility to diagnose long-running imports.
- Effort: M
- Category: correctness
- Owner/session: content-ops/backend-file-ingestion-validation
- Found during: PR-Content-Ops-File-Cross-Process-Hardening-Register

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
