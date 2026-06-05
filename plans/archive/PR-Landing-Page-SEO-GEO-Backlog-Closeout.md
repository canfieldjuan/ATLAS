# PR: Landing Page SEO/GEO Backlog Closeout

## Why this slice exists

The landing-page SEO/AEO/GEO implementation has now landed through input
capture, prompt alignment, save-time readiness, review UI, draft edit/repair,
public rendering, sitemap/prerender, publish verification, and generation smoke
coverage. The active AI Content Ops deferred backlog still names the blog
SEO/GEO closeout but does not explicitly close the equivalent landing-page arc.

That leaves future sessions at risk of rediscovering stale landing-page
deferrals from historical plan docs.

## Scope (this PR)

Ownership lane: content-ops/landing-page-seo-geo-backlog-closeout

1. Update the active AI Content Ops deferred backlog timestamp.
2. Add landing-page SEO/AEO/GEO work to retired historical deferrals.
3. Add a current-pick note that the landing-page SEO/AEO/GEO arc is closed for
   the current product contract.
4. Keep this docs-only; no runtime behavior changes.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Landing-Page-SEO-GEO-Backlog-Closeout.md` | Plan doc for this docs-only closeout. |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Mark landing-page SEO/AEO/GEO as closed in the active backlog. |

## Mechanism

The backlog already has a "Retired Historical Deferrals" list and a "Current
Pick Recommendation" section that explicitly closes completed arcs. This slice
adds the landing-page SEO/AEO/GEO chain to both places and points future work
at live-output failures, operator needs, or publish-verifier regressions
instead of stale historical deferrals.

## Intentional

- No runtime code changes.
- No test changes beyond mechanical local review.
- No claim that generated landing pages are guaranteed to rank or appear in AI
  answer engines; this closeout is about implemented readiness and verification.

## Deferred

- Parked hardening: none. Root `HARDENING.md` has no landing-page backlog items.
- Real operator acceptance testing for generated landing pages remains the next
  practical product check, but it is not a docs backlog closeout change.

## Verification

- `git diff --check` -> passed.
- Local PR review wrapper -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~65 |
| Backlog update | ~30 |
| **Total** | **~95** |
