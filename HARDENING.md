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

### FAQSTRESS-2 - FAQ lifecycle concurrency needs DB pressure limits and failure artifacts
- File/location: `scripts/smoke_content_ops_faq_lifecycle.py::_create_pool`, `atlas_brain/_content_ops_services.py::build_content_ops_execution_services`, hosted FAQ execution path.
- Description: Real Postgres-backed concurrency probe passed through 50 simultaneous 10,000-row lifecycle runs, but 100 simultaneous 5,000-row runs produced 97 successes and 3 `asyncpg.exceptions.TooManyConnectionsError` failures. The failed smoke processes exited before writing `--output-result`.
- Why it matters: concurrent customer uploads can exhaust database connection slots unless hosted FAQ execution has bounded concurrency, queue backpressure, request limits, or equivalent controls; missing failure artifacts also make saturation harder to diagnose.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-generator-validation
- Found during: PR-Content-Ops-FAQ-Scale-Stress-Probe

### FAQSTRESS-1 - FAQ large uploads need async/job boundary or explicit hosted limits
- File/location: `scripts/smoke_content_ops_faq_scale_run.py`, `scripts/smoke_content_ops_faq_lifecycle.py`, hosted FAQ upload/execution path.
- Description: Real CFPB-derived stress probe passed correctness through 50,000 rows, but 50k deterministic generation took 1:49.48 and 590,612 KB RSS; the DB lifecycle took 1:05.32 and 681,848 KB RSS. This is batch-safe but not request/response-safe.
- Why it matters: large customer uploads can tie up request workers and memory unless the hosted path uses explicit row/file limits, background jobs, queue backpressure, or similar controls.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-generator-validation
- Found during: PR-Content-Ops-FAQ-Scale-Stress-Probe

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
