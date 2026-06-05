# PR-Blog-B2B-Complaint-Conflation

Ownership lane: `content-ops/blog-b2b-complaint-conflation`

## Why this slice exists

Pre-launch correctness patch. `top-complaint-every-b2b-software-2026-03` presented
**9,137 as both the complaint count AND the review count** ("9,137 complaints across
9,137 reviews", "Total complaint volume: 9,137") — an artifact of the #859 stale-count
fix collapsing the old distinct "11,399 complaints / 26,335 reviews" both to 9,137. It
also still carried the stale "7,985 verified + 18,350 community-sourced reviews"
breakdown (= 26,335), contradicting the corrected 9,137 figure.

## Scope (this PR)

Make the post numerically coherent: 9,137 means **reviews** consistently (the figure
#859 established and DB-grounded), drop the duplicated "complaints" count, and remove the
stale 26,335 verified/community breakdown. **No new numbers invented.**

5 lines changed in `top-complaint-every-b2b-software-2026-03.ts`:
- seo_description: "9,137 complaints across 54 vendors from 9,137 reviews" → "9,137 reviews across 54 vendors"
- FAQ answer: "Based on 9,137 complaints across 9,137 reviews" → "Based on 9,137 reviews"
- content intro: "examines 9,137 complaints across 54 vendors" → "examines complaints across 54 vendors"; dropped "—7,985 verified reviews and 18,350 community-sourced reviews"
- landscape-at-a-glance: "Total complaint volume: 9,137." → "54 vendors analyzed across 9,137 reviews."
- methodology footer: dropped "The dataset includes 7,985 verified reviews … and 18,350 community-sourced reviews."

### Files touched

- `plans/PR-Blog-B2B-Complaint-Conflation.md`
- `atlas-churn-ui/src/content/blog/top-complaint-every-b2b-software-2026-03.ts`

## Mechanism

Exact-match substring replacements (assert 1 match each), substrings chosen to avoid the
apostrophe in "tool's". Verified after: zero remaining "9,137 complaints"/"complaints
across 9,137", zero "Total complaint volume", zero 7,985/18,350/26,335; all 8 surviving
9,137 mentions are review-context.

## Intentional

- **9,137 kept as the single, accurate figure** (reviews) rather than inventing a distinct
  complaint count — the generator's model is total_complaints = enriched_reviews, so a
  separate complaint count was always redundant/misleading here. Qualitative complaint
  analysis (pricing dominates, urgency, etc.) is unchanged.
- **Stale 26,335 breakdown removed, not relabeled** — it contradicts 9,137 and no verified
  verified/community split for the corrected scope is available (same "drop the unfounded
  number" discipline used on the Copper 928 fix).
- **D6 (#802) NOT included** — that item is already MERGED (zoho G2→Slashdot attribution);
  the earlier "D6 open" flag was stale memory.

## Deferred

- The post's methodology still lists some non-allowlist sources (Capterra/TrustRadius/
  Trustpilot) — a separate parked item (methodology-declares-non-allowlist-sources), out
  of scope here.

Parked hardening: none new.

## Verification

- 5 edits applied via assert-exact-match; `git diff` = 1 file, 5 lines; grep confirms no
  conflation, no stale breakdown, no "Total complaint volume" line remain.

## Estimated diff size

| Area | LOC |
|---|---:|
| 1 post file | ~10 |
| Plan doc | ~55 |
| **Total** | **~65** |
