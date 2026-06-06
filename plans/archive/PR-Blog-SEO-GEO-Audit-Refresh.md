# PR-Blog-SEO-GEO-Audit-Refresh

## Why this slice exists

The AI Content Ops blog GEO contract still described the implementation as a
future sequence even though later PRs wired draft readiness, quality-gate
checks, and publish verification.

This docs-only slice refreshes the source-of-truth contract language so the
next engineering slices target real remaining gaps instead of stale ones.

## Scope (this PR)

1. Mark generated-asset GEO readiness output as implemented.
2. Mark draft-level SEO/AEO/GEO quality-gate wiring as implemented.
3. Mark current publish-level verification as implemented where it exists.
4. Narrow remaining and active publish work to chart/FAQ fallbacks and shared
   source metadata parsing.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Blog-SEO-GEO-Audit-Refresh.md` | Plan doc for this slice. |
| `docs/audits/ai_content_ops_blog_geo_contract_2026-05-20.md` | Refresh implementation status and customer-facing language guardrails. |

## Mechanism

This change does not alter runtime code. It cross-checks the audit language
against the implemented files already on `origin/main`:

- `extracted_content_pipeline/blog_generation.py`
- `extracted_content_pipeline/blog_post_postgres.py`
- `extracted_content_pipeline/blog_post_export.py`
- `extracted_quality_gate/blog_pack.py`
- `atlas-intel-ui/scripts/verify-blog-geo-prerender.mjs`
- `atlas-intel-ui/vite.config.ts`

## Intentional

- No product copy changes.
- No runtime code changes.
- No test behavior changes.
- No claim that GEO guarantees AI-engine placement.

## Deferred

- Implementing or preserving chart evidence fallbacks for prerendered blog
  pages.
- Implementing FAQPage/static FAQ publish verification when source posts include
  FAQ entries.
- Sharing frontend/verifier source parsing through a generated manifest.

## Verification

- Markdown/audit review.
- Whitespace diff check.
- Local PR review.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~60 |
| GEO contract refresh | ~35 |
| Total | ~95 |
