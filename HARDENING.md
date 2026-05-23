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

### FILECONCURRENCY-1 - Uploaded-file write pressure needs bounded DB concurrency
- File/location: `scripts/smoke_content_ops_ingestion_file_route.py`, hosted `/content-ops/ingestion/files/import` write path.
- Description: Local Postgres pressure validation passed at 5 concurrent 10,000-row writes, 20 concurrent 1,000-row writes, and 100 concurrent 1,000-row writes, but 150 concurrent 1,000-row write processes produced 141 successes and 9 `asyncpg.exceptions.TooManyConnectionsError` failures during pool creation.
- Why it matters: concurrent customer uploads can exhaust database connection slots unless hosted file-route writes use bounded concurrency, queue backpressure, async jobs, or a shared pool/admission-control layer.
- Effort: M
- Category: correctness
- Owner/session: content-ops/backend-file-ingestion-validation
- Found during: PR-Content-Ops-File-Route-Concurrency-Validation

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
