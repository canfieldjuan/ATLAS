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

### FAQ hosted route proof preflight accepts local API URLs
- File/location: `scripts/smoke_content_ops_faq_saas_demo_route_e2e.py` preflight and `docs/extraction/validation/content_ops_faq_saas_demo_route_case_runbook.md`
- Description: The hosted FAQ route proof preflight passed with `ATLAS_API_BASE_URL` set to a local HTTP host with a port, then the route phase failed all requests with connection refused.
- Why it matters: Operators can think the deployed-host proof is ready while the configured URL only points to a stopped local server.
- Effort: S
- Category: correctness
- Owner/session: content-ops/faq-search
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Hosted-Route-Proof

### FAQ route concurrency result top-level query can disagree with case-file query
- File/location: `scripts/smoke_content_ops_faq_search_route_concurrency.py` result payload summary
- Description: A case-file run used the SaaS demo query, but the route result's top-level `query` field still reflected the default/env query.
- Why it matters: The per-case summaries are correct, but the compact top-level artifact can mislead readers skimming a failed run.
- Effort: S
- Category: polish
- Owner/session: content-ops/faq-search
- Found during: PR-Content-Ops-FAQ-SaaS-Demo-Hosted-Route-Proof

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
