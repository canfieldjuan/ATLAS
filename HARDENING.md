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

## 2026-05-23

### Extracted migration dry-run reports applied migrations as pending
- File/location: `extracted_content_pipeline/storage/migration_runner.py` and `scripts/run_extracted_content_pipeline_migrations.py --dry-run --json`
- Description: After the local extracted migrations were already applied, dry-run still reported `applied_count=28` with `status="dry_run"` while the actual non-dry run reported `applied_count=0`, `skipped_count=28`.
- Why it matters: Operators can misread a dry-run as pending database work even when the migration table says the database is already current.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-generator-io-tests
- Found during: PR-Content-Ops-FAQ-Lifecycle-DB-1000-Run
