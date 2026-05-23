# PR-Blog-Headlines-Landscapes

Ownership lane: `content-ops/blog-headlines-landscapes`

## Why this slice exists

Headlines phase, scaling slice 3 / final (after sample #892, deep-dives #896, vs #897).
Rewrites the 4 category-landscape posts whose titles use the dry "N Vendors Compared by
Real User Data" pattern, matching the approved crm-landscape style ("N Vendors on
{axes} (count)"). The other landscape posts (b2b-software ×2, communication) already
carry a count/finding and are left as-is.

## Scope (this PR)

4 `title` rewrites. No `seo_title`/body changes. Each leads with the buyer axes (cost
from the validated keyword + churn + a category-specific pain) and surfaces the count.

### Files touched

- `plans/PR-Blog-Headlines-Landscapes.md`
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`

## Mechanism

Each edit replaces one `title:` line via an assert-exact-match script (1 match per file).
The vendor count (5/4/5/10) is preserved from the old title. The added count is taken
from each post's own description **with the correct unit** — verified: helpdesk "352
enriched reviews" → "(352 Reviews)"; hr-hcm "306 churn signals" → "(306 Churn Signals)";
marketing-automation "640 enriched reviews" → "(640 Reviews)"; project-management "1,464
churn signals" → "(1,464 Churn Signals)". Reviews and churn signals are NOT conflated.

## Intentional

- **Cost axis = each post's validated keyword** (help desk software cost / HR software
  cost / marketing automation cost / project management software cost).
- **Correct metric unit per post** — the two posts that count churn signals say "Churn
  Signals", the two that count reviews say "Reviews"; this avoids the reviews-vs-signals
  conflation that the earlier correctness audit fixed elsewhere.
- **`seo_title` untouched**; b2b-software/communication landscapes left as-is (already
  carry a count).

## Deferred

- Headline scaling is complete after this slice. Intentionally-untouched (already punchy):
  top-complaint posts, why-teams-leave posts, finding-led deep-dives, switch posts, the
  two cross-category "Urgency Gap" vs-posts, b2b-software/communication landscapes.

Parked hardening: none new.

## Verification

- All 4 edits applied via assert-exact-match (1 match/file); `git diff` = 4 `title`
  lines only; vendor counts preserved; added counts match each description's figure +
  unit.

## Estimated diff size

| Area | LOC |
|---|---:|
| 4 post files (title) | ~8 |
| Plan doc | ~52 |
| **Total** | **~60** |
