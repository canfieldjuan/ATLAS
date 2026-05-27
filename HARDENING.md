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

## 2026-05-27

### SaaS demo seeder runtime setup result artifact
- File/location: `scripts/seed_content_ops_faq_saas_demo.py`, `_run(...)` / `main(...)`
- Description: Pool creation or runtime seed/cleanup exceptions can still abort before `--output-result` is written.
- Why it matters: Operators get a preflight artifact after this slice, but missing `asyncpg`, connection failures, or unexpected repository errors can still leave no machine-readable failure payload.
- Effort: S
- Category: correctness
- Owner/session: content-ops/faq-generator
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Preflight-Result

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
