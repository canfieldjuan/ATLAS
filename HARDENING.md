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

### FAQ search DB smoke cleanup failures can mask original result
- File/location: `scripts/smoke_content_ops_faq_search_concurrency.py`
  `run_smoke(...)` cleanup `finally` block.
- Description: Cleanup exceptions can still replace an earlier setup/search
  result instead of being reported as cleanup metadata.
- Why it matters: A failed cleanup could hide whether retrieval passed, failed,
  or never ran, making go-live smoke artifacts harder to interpret.
- Effort: M
- Category: correctness
- Owner/session: Codex FAQ lane
- Found during: PR-Content-Ops-FAQ-Search-DB-Setup-Failure-Result

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
