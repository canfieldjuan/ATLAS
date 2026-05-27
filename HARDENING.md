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

### SaaS demo seeder pool-close result preservation
- File/location: `scripts/seed_content_ops_faq_saas_demo.py`, `_run(...)` / `main(...)`
- Description: If seed or cleanup succeeds but `pool.close()` raises, the runtime failure payload can replace the successful operation payload instead of reporting close failure as lifecycle metadata.
- Why it matters: Operators get a result artifact, but a successful seed could be obscured by a close-time failure during repeated demo setup.
- Effort: S
- Category: correctness
- Owner/session: content-ops/faq-generator
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Seeder-Runtime-Result review

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
