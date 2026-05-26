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

## 2026-05-26

### Support-ticket descriptive blog output is long and repetitive on tiny uploads
- File/location: `atlas_brain/skills/digest/blog_post_generation.md` and `extracted_content_pipeline/skills/digest/blog_post_generation.md`, support-ticket descriptive draft path.
- Description: The live Haiku validation saved a truth-safe support-ticket blog from a 4-row upload, but the article repeated similar sections and used 10k+ output tokens.
- Why it matters: The current output is acceptable for truthfulness validation, but small support-ticket uploads need tighter length/section guidance so review drafts are cheaper, easier to edit, and less repetitive.
- Effort: S
- Category: polish
- Owner/session: content-ops/support-ticket-provider-descriptive-blog-live-validation
- Found during: PR-Support-Ticket-Descriptive-Blog-Live-Validation.

## 2026-05-23

### FAQSCALE-1 - Large synchronous FAQ generation needs hosted limits / backpressure / background execution
- File/location: Hosted FAQ generation path (deterministic generator `extracted_content_pipeline/ticket_faq_markdown.py` via the `/content-ops` execute route); offline proof harness `scripts/smoke_content_ops_faq_scale_run.py`.
- Description: A gated 50,000-row deterministic FAQ run completes correctly but synchronously, taking ~1:41.86 wall and ~593 MB RSS on one core (recorded in `docs/extraction/validation/content_ops_faq_50k_gated_validation_2026-05-23.md` and the earlier `content_ops_faq_scale_stress_probe_2026-05-23.md`). Hosted paths have no explicit upload-size limit, backpressure, or background-job execution for large uploads, and no bound on concurrent large runs.
- Why it matters: Exposing large uploads as synchronous customer requests would tie up a worker for ~100s and ~600 MB per run with no concurrency bound -- a latency/availability and memory-pressure risk. Limits / backpressure / background execution should land before large uploads are offered as a synchronous hosted endpoint.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-generation-scale
- Found during: PR-Content-Ops-FAQ-50K-Gated-Validation (#914); previously noted in the FAQ scale stress probe.

> **Atlas blog / deep-dive content pipeline** (`content-ops/blog-*` ownership
> lanes): parked items live in [`ATLAS-HARDENING.md`](./ATLAS-HARDENING.md),
> kept separate to avoid append-collisions with the concurrent
> content-ops-station sessions. Scan that file too when working those lanes.
