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

### Blog save-time GEO gate and export readiness disagree on citable sections
- File/location: `extracted_quality_gate/blog_pack.py`, `extracted_content_pipeline/blog_post_export.py`, live run in `tmp/support_ticket_saas_demo_blog_acceptance_20260528_after_geo/blog-post-draft.json`.
- Description: After the GEO repair guidance landed, the 36-row SaaS demo blog saved and passed support-ticket generated-content evaluation, but the exported draft still reported GEO `needs_review` because `citable_section_structure` was missing. Diagnostic replay showed the save-time quality pack passed the same draft with only a warning, so save-time and export citable-section rules are not equivalent.
- Why it matters: The blog path can save a draft that is not export-ready for GEO, which would let an apparently successful generation miss the SEO/GEO/AEO acceptance bar.
- Effort: M
- Category: correctness
- Owner/session: content-ops/support-ticket-provider
- Found during: PR-Support-Ticket-SaaS-Demo-Blog-Accepted-Fixture

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
