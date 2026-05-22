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

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.

## 2026-05-22

### Remove landing-page repair legacy lock after rollout
- File/location: `extracted_content_pipeline/api/generated_assets.py`, `_landing_page_repair_lock`
- Description: The repair lock still acquires the legacy `hashtext()` advisory lock for rolling-deploy compatibility while also acquiring the widened `hashtextextended()` lock.
- Why it matters: After the widened-lock release is fully deployed, keeping the legacy compatibility lock preserves the old 32-bit collision surface as a transition guard.
- Effort: S
- Category: tech-debt
- Owner/session: landing-page repair session
- Found during: PR-Landing-Page-Repair-Lock-Hash-Key review

### Revisit repair lock connection hold time
- File/location: `extracted_content_pipeline/api/generated_assets.py`, `repair_landing_page_draft`
- Description: The advisory-lock connection stays checked out while the LLM repair runs.
- Why it matters: This is acceptable for operator-triggered repair, but higher repair volume could turn LLM latency into pool pressure.
- Effort: M
- Category: tech-debt
- Owner/session: landing-page repair session
- Found during: PR-Landing-Page-Repair-Cost-Guard review
