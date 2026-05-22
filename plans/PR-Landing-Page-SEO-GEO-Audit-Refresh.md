# PR: Landing Page SEO/GEO Audit Refresh

## Why this slice exists

The landing-page SEO/AEO/GEO audit document still describes readiness helpers,
quality gates, review UI visibility, and publish verification as future work.
Those slices have now landed. Leaving the audit stale makes future sessions plan
against false gaps.

This slice refreshes the audit doc to match the current implementation.

Ownership lane: content-ops/landing-page-seo-geo-audit-refresh

## Scope (this PR)

1. Update the current implementation baseline for generated landing pages.
2. Mark the roadmap as implemented through draft checks, review UI, edit/repair,
   public rendering, sitemap/prerender, and publish verification.
3. Refresh the safe customer-facing language so it matches the current proof.
4. Resolve the open decisions that now have concrete implementation choices.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-SEO-GEO-Audit-Refresh.md` | Plan doc for this audit refresh slice. |
| `docs/audits/ai_content_ops_landing_page_seo_aeo_geo_contract_2026-05-21.md` | Refresh implementation status and remaining language. |

## Mechanism

This is a docs-only truthfulness update. It references the files and behavior
now present in the repo without changing runtime code.

## Intentional

- No runtime code changes.
- No test changes.
- No new SEO/GEO claims beyond readiness and verification.

## Deferred

- `HARDENING.md` still tracks landing-page repair legacy-lock rollout cleanup
  and repair lock connection hold time. Both are parked under
  `Owner/session: landing-page repair session` and are not required for this
  audit refresh.

## Parked hardening

- None added.

## Verification

- Local PR review -> passed.

## Estimated diff size

| File | Estimated LOC |
| --- | ---: |
| `docs/audits/ai_content_ops_landing_page_seo_aeo_geo_contract_2026-05-21.md` | 120 |
| `plans/PR-Landing-Page-SEO-GEO-Audit-Refresh.md` | 55 |
| **Total** | **175** |
