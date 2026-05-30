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

## 2026-05-30

### FAQ deflection blob submit DNS-rebinding TOCTOU
- File/location: `extracted_content_pipeline/api/control_surfaces.py` `_validate_blob_host_resolution` / `_read_bounded_https_blob`
- Description: Blob submit validates resolved host IPs before fetch, but urllib re-resolves during connect, leaving a DNS-rebinding time-of-check/time-of-use gap.
- Why it matters: The endpoint is customer-reachable for paid deflection report submits, so broad GA should pin the validated IP at connection construction or restrict fetches to trusted blob hosts.
- Effort: M
- Category: security
- Owner/session: Codex FAQ deflection blob redirect hardening
- Found during: PR-FAQ-Deflection-Blob-Redirect-Hardening

## 2026-05-29

### atlas-intel-ui npm audit vulnerabilities
- File/location: `atlas-intel-ui/package-lock.json`
- Description: `npm ci` reports 6 dependency audit findings (2 moderate, 4 high).
- Why it matters: Dependency vulnerabilities can become deploy-time security exposure, but resolving them may require package upgrades outside this UI rendering slice.
- Effort: M
- Category: security
- Owner/session: Codex FAQ deflection report UI slice
- Found during: PR-FAQ-Deflection-Report-UI-Readonly

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
