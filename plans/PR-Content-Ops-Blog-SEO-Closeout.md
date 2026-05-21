# Content Ops Blog SEO Closeout

## Why this slice exists

The AI Content Ops blog SEO/AEO/GEO audit still describes several gaps as
open even though the follow-up PR chain has now landed:

- PR #665 persists extracted blog SEO fields into first-class `blog_posts`
  columns.
- PR #669 and PR #674 expose SEO/AEO and GEO readiness in generated-asset
  export rows.
- PR #671 and PR #676 block incomplete SEO/AEO and GEO drafts before save.
- PR #682, PR #683, PR #691, PR #693, and PR #698 cover publish-level blog
  GEO/SEO/JSON-LD/crawler-visible checks.
- PR #688 and PR #690 surface readiness in the review UI.

The stale audit text caused the next-slice picker to re-select completed SEO
persistence work. This PR closes that coordination gap so future sessions do
not spend time rediscovering already-merged work.

## Scope (this PR)

1. Update the blog SEO/AEO/GEO discovery audit with the merged closeout status.
2. Update the AI Content Ops deferred backlog so blog SEO/GEO and the merged
   FAQ source-context slice are no longer presented as active work.
3. Clean the in-flight coordination row for the merged FAQ PR and claim this
   doc-only closeout slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Blog-SEO-Closeout.md` | Plan doc for this slice. |
| `docs/audits/ai_content_ops_blog_seo_aeo_geo_discovery_2026-05-20.md` | Mark merged SEO/AEO/GEO follow-up work as closed and update safe claim language. |
| `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md` | Retire the merged blog SEO/GEO and FAQ source-context follow-ups from active selection. |
| `docs/extraction/coordination/inflight.md` | Remove stale FAQ in-flight row and claim this closeout. |

## Mechanism

This is a documentation and coordination update only. It records the actual
merged PR sequence next to the stale audit findings, then restates the current
selection rule: do not take another AI Content Ops blog SEO/GEO slice unless a
new live output, UI need, or publish verifier regression exposes a concrete
gap.

## Intentional

- No runtime code changes. The runtime paths are already covered by the merged
  PRs listed above.
- No new product claim beyond the verified wording. The docs still avoid
  guaranteeing AI-engine placement or "fully optimized" outcomes.
- No cleanup of unrelated blog content PRs. Open blog content work remains
  separate from this AI Content Ops closeout.

## Deferred

- Remove this in-flight coordination row after the PR merges.
- Future Content Ops blog SEO/GEO work should start from a new concrete
  failure, not from this now-closed audit.

## Verification

Planned commands:

```bash
bash scripts/local_pr_review.sh
git diff --check
```

No package tests are required because this slice changes Markdown docs only.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Audit/backlog docs | ~165 |
| Coordination ledger | ~5 |
| Total | ~250 |
