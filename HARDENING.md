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

### Support-ticket blog generation needs contract-level descriptive mode before SaaS demo acceptance
- File/location: `atlas_brain/skills/digest/blog_post_generation.md`, `extracted_content_pipeline/blog_generation.py`, support-ticket blog live runs in `tmp/support_ticket_saas_demo_generated_content_acceptance_20260528/`.
- Description: The 36-row SaaS demo CSV repeatedly drove Haiku blog output toward unsupported benefit/outcome language even after prompt and evaluator tightening. The final retry is now caught as known-bad, but no accepted SaaS demo blog fixture is committed in this slice.
- Why it matters: The small 4-row blog path is accepted, but the broader support-ticket blog path still needs a stronger contract than prompt prohibitions plus pattern blockers before it can be treated as production-accepted.
- Effort: M
- Category: correctness
- Owner/session: content-ops/support-ticket-provider
- Found during: PR-Support-Ticket-SaaS-Demo-Generated-Content-Acceptance

### LLM usage storage schema mismatch hides per-run cost telemetry
- File/location: `content_ops.llm.complete` usage-storage path, live smoke stderr during `scripts/smoke_content_ops_live_generation.py`.
- Description: Live landing/blog generation succeeded, but each LLM call logged `_store_local failed for span=content_ops.llm.complete: column "account_id" of relation "llm_usage" does not exist`.
- Why it matters: Generation still works, but local cost/usage surfacing is incomplete, which blocks the cost visibility work the product needs before heavier validation and production use.
- Effort: M
- Category: correctness
- Owner/session: content-ops/support-ticket-provider
- Found during: PR-Support-Ticket-SaaS-Demo-Generated-Content-Acceptance

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
