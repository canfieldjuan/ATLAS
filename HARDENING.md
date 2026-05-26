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

### FAQ action steps are intent-template based, not resolution-evidence gated
- File/location: `extracted_content_pipeline/ticket_faq_markdown.py` (`_ACTION_RULES`, `_article_steps`).
- Description: The deterministic FAQ generator emits action steps from intent templates. The steps are grounded to the detected topic and source IDs, but they are not gated on explicit support-resolution evidence from the uploaded rows.
- Why it matters: This is acceptable for draft/review FAQ output, but publish-ready support answers should not imply verified product-specific resolution steps unless the upload includes resolution evidence or the output labels the steps as draft guidance.
- Effort: M
- Category: correctness
- Owner/session: content-ops/faq-generator
- Found during: PR-FAQ-Generator-Output-Proof.

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
