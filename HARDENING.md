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

### Reject Contradictory FAQ Route Detail Concurrency Flags
- File/location: `scripts/smoke_content_ops_faq_search_route_concurrency.py` argument preflight.
- Description: `--require-detail --allow-empty-results` currently fails loudly per request instead of rejecting the contradictory flags before the run starts.
- Why it matters: Preflight rejection would make operator mistakes clearer and avoid spending concurrency requests on a configuration that cannot hydrate detail rows.
- Effort: S
- Category: polish
- Owner/session: Codex FAQ lane
- Found during: PR #1025 review follow-up while starting the next FAQ search slice.

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
