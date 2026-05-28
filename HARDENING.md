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

## 2026-05-28

### LLM usage storage schema mismatch hides per-run cost telemetry
- File/location: `atlas_brain/observability` LLM usage storage path, local `llm_usage` table schema.
- Description: Live support-ticket blog generation logged LLM usage storage failures, including `column "account_id" of relation "llm_usage" does not exist` and `pool is closing`.
- Why it matters: Generation can still run, but per-run model/cost telemetry is incomplete and warning noise can hide other live-generation issues.
- Effort: M
- Category: tech-debt
- Owner/session: content-ops/support-ticket-provider
- Found during: PR-Support-Ticket-Blog-Observed-Shell-Live-Retry

### Atlas startup migration check warns on missing b2b_campaigns.updated_at
- File/location: `atlas_brain/storage/migrations/309_campaign_sequences_unique_active_recipient.sql`, Atlas startup migration check.
- Description: Local API startup logged `column "updated_at" of relation "b2b_campaigns" does not exist` while checking pending host migrations.
- Why it matters: The FAQ route still starts and passes, but startup migration noise can hide real route-readiness issues and suggests local/host B2B campaign schema drift.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-search
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Proof

### Voice ASR auto-start blocks non-voice route validation on CUDA-less hosts
- File/location: `atlas_brain/main.py` lifespan voice startup, `atlas_brain/config.py` voice defaults.
- Description: Starting Atlas without voice overrides attempted to auto-start ASR on CUDA and failed because no CUDA GPU was available.
- Why it matters: Non-voice route validation can waste time or appear broken unless operators know to set `ATLAS_VOICE_ENABLED=false ATLAS_VOICE_AUTO_START_ASR=false`.
- Effort: S
- Category: polish
- Owner/session: content-ops/faq-search
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Local-Route-Proof

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
