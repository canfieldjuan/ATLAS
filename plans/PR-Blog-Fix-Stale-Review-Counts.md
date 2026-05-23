# PR-Blog-Fix-Stale-Review-Counts

Ownership lane: `content-ops/blog-fix-stale-review-counts`

## Why this slice exists

Deep-pass recon of the landscape/aggregate post types surfaced inflated headline
review counts (e.g. b2b-software-landscape claimed "23,873 enriched reviews" --
more than the 22,176 enriched rows in the ENTIRE `b2b_reviews` table). The
maintainer asked to investigate the generator count logic first; the
investigation found a clear, bounded root cause:

**Root cause = pre-D7 mention-scoped overcounts (generator already fixed).** The
current generator (`b2b_blog_post_generation.py` ~L6151) scopes category-level
review counts to `product_category` (D7 fix), explicitly because the older
vendor-MENTION scoping overcounts via off-category community cross-mentions (the
code comment cites "CRM ~5,931 mention-scoped vs ~1,133 category-scoped"). These
published posts were generated with the OLD mention-scoped logic, so their
headline counts are stale/inflated. The generator logic is now correct -- this is
a one-time data correction of legacy posts, not a recurring bug.

(The parked pain-radar chart-provenance question resolved in the same pass:
`avg_urgency_when_mentioned` (`_b2b_shared.py:10495`) is a legitimate
aspect-scoped average -- correct, just statistically fragile for low-N categories.
No generator fix; informational -- it downgrades the parked chart-provenance
question from "is the chart wrong?" to "no, it's a legitimate low-N-fragile
average," which is why the DD3 fix (#855) correctly avoided crowning a low-N
urgency outlier.)

## Scope (this PR)

Recomputed each affected post's count from the DB using the CURRENT generator
logic (category-scoped, allowlist sources, `duplicate_of_review_id IS NULL`,
distinct-by-review, windowed to the post's collection period). Corrected only the
posts whose published count is clearly impossible (>1.5x the category corpus);
left posts within ~3-6% (windowing/dedup noise) untouched.

| Post | published -> corrected | basis |
|---|---|---|
| b2b-software-landscape-2026-04 | 23,873 -> 9,137 | B2B enriched, windowed |
| b2b-software-landscape-2026-03 | 26,335 -> 9,137 | B2B enriched |
| best-b2b-software-for-1000-2026-03 | 11,399 & 26,335 -> 9,137 | (post stated TWO different enriched counts) |
| top-complaint-every-b2b-software-2026-03 | 11,399 & 26,335 -> 9,137 | B2B enriched |
| helpdesk-landscape-2026-04 | 1501 -> 352 | Helpdesk enriched |
| communication-landscape-2026-04 | 1,512 -> 630 | Communication enriched |
| marketing-automation-landscape-2026-04 | 1,568 -> 640 | Marketing Automation enriched |
| best-hr-hcm-for-51-200-2026-04 | 943 -> 586 | HR/HCM TOTAL ("943 reviews", not enriched) |
| top-complaint-every-project-management-2026-04 | 3,019 -> 1,246 enriched / 2,633 total | PM (post used 3,019 for BOTH) |

### Files touched

- `plans/PR-Blog-Fix-Stale-Review-Counts.md`
- `atlas-churn-ui/src/content/blog/b2b-software-landscape-2026-03.ts`
- `atlas-churn-ui/src/content/blog/b2b-software-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-b2b-software-for-1000-2026-03.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/communication-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-b2b-software-2026-03.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-project-management-2026-04.ts`

## Mechanism

Per-occurrence number replacement (total vs enriched distinguished by phrasing).
Numbers grounded in DB queries run this session against `localhost:5433/atlas`.
Side effect: helpdesk's "11-68 reviews per vendor" now reconciles with the
corrected 352 (5x~70), resolving its prior internal contradiction.

## Intentional

- **Category-scoped, windowed counts** (matches the current/D7 generator + the
  post's stated collection window), not all-time or mention-scoped.
- **Total vs enriched kept distinct**: best-hr-hcm "943 reviews" -> total (586);
  top-complaint-PM "3,019 enriched" -> 1,246, its "3,019 reviews" -> 2,633.
- **Marginal overages left**: crm-landscape (1,054), project-management-landscape
  (1,464), best-crm (1,163), and the e-commerce/communication/marketing-auto
  top-complaints are within ~3-6% of the category corpus -- windowing/dedup noise,
  not worth the churn/risk of "correcting" to a fuzzy windowed figure.
- **No detector added**: this is stale legacy data, not a recurring generator bug
  (D7 already fixed it), and the static auditor cannot DB-recompute. A future
  regen would produce correct counts.

## Deferred

- Other DERIVED numbers in these posts (per-vendor sample sizes beyond helpdesk's,
  "N churn signals", chart `signal_count`s) may also be pre-D7 mention-scoped --
  not corrected here (headline count was the credibility issue). Parked.
- Marginal-overage posts (above) -- left as windowing noise.

Parked hardening: none new (the derived-number recompute is noted in Deferred;
the chart-provenance resolution is recorded in this plan + the roadmap memory).

## Verification

- All corrected numbers traced to DB queries (category-scoped, allowlist, deduped,
  windowed) this session.
- `audit-published-posts.js` full corpus (78 posts): total detector findings = 0
  (files parse; no regression from the edits).
- grep confirms all 9 stale numbers removed; new numbers present and internally
  consistent (b2b posts now state a single 9,137; PM total 2,633 > enriched 1,246).

## Estimated diff size

| Area | LOC |
|---|---:|
| 9 posts (number corrections) | ~42 |
| Plan doc | ~95 |
| **Total** | **~137** |
