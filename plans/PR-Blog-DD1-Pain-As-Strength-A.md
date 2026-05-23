# PR-Blog-DD1-Pain-As-Strength-A

Ownership lane: `content-ops/blog-dd1-pain-as-strength-a`

## Why this slice exists

First slice of the deep-pass DD1 fix. The strengths-weaknesses chart splits areas
by a per-area satisfaction score; `overall_dissatisfaction` is a catch-all PAIN
signal that gets a spurious high score and lands in the STRENGTHS series. So the
chart shows dissatisfaction AS a strength and the prose narrates it as "overall
satisfaction" -- the opposite of the data (DB-verified: overall_dissatisfaction is
a complaint category; e.g. Asana has 100 reviews tagged overall_dissatisfaction
primary, no positive "satisfaction" signal exists). A new `pain_as_strength`
detector finds 19 posts; this batch covers the 6 that also NARRATE it as
satisfaction in prose (the substantive rewrites). The other 13 are chart-only and
follow in sibling slices.

(pricing/support also appear in some posts' strengths, but DB-verified their
scores are legitimately computed -- azure pricing 3.84, support 3.73, in the same
band as real strengths -- so those are NOT touched; only the unambiguous
`overall_dissatisfaction` catch-all is flipped.)

## Scope (this PR)

Per post: (a) move `overall_dissatisfaction` from the strengths series to
weaknesses in the strengths-weaknesses chart; (b) rewrite the prose that called
it "overall satisfaction" to reflect dissatisfaction (a weakness).

- **asana** (208): chart move; delete the false "Overall satisfaction with core
  workflow" strength bullet; rewrite the weakness note to state the real 208
  overall-dissatisfaction count (was "exact count withheld ... overall
  satisfaction is high").
- **clickup** (252): chart move; drop "overall satisfaction (252 mentions)" from
  the FAQ praise list; reframe the "Overall satisfaction (252 mentions)" strength
  bullet to "Value for the price" (keeps the real value/features quote).
- **power-bi** (534): chart move; remove the invented "appears in both strength
  and weakness columns because some report satisfaction" justification; fix two
  "534 mentions of overall satisfaction" -> overall-dissatisfaction (largest pain
  signal). The "retention via lock-in, not enthusiasm" thesis is preserved (now
  consistent with high dissatisfaction).
- **slack** (293): chart move; fix three "293 ... overall satisfaction" references
  -> overall dissatisfaction is the largest pain signal; the ease-of-use retention
  anchor is kept.
- **woocommerce** (396): chart move; delete the false "Overall satisfaction"
  strength bullet; drop "strong overall satisfaction" from two retention-driver
  sentences (keeps UX/ecosystem drivers).
- **looker** (115): chart move; rewrite the strengths sentence (UX is the real
  top strength after the move; overall dissatisfaction is the top weakness).

### Files touched

- `plans/PR-Blog-DD1-Pain-As-Strength-A.md`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/power-bi-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/woocommerce-deep-dive-2026-04.ts`

## Mechanism

The chart datum (`{strengths: N, weaknesses: 0}` -> `{strengths: 0, weaknesses:
N}`) is one line per chart; the substantive change is the prose, which flips each
post from "dissatisfaction is the top strength / satisfaction is high" toward the
data-accurate "overall dissatisfaction is the largest pain signal." Real strengths
(UX, features, ease of use) and the posts' retention theses are preserved; only
the false satisfaction framing is removed. No invention -- the dissatisfaction
counts are the same numbers, now placed and described truthfully.

## Intentional

- **Only `overall_dissatisfaction` flipped.** pricing/support-in-strengths are
  legitimately-computed scores (DB-verified), left as-is.
- **Theses preserved.** power-bi's lock-in framing and slack's "scope not quality"
  point survive (both consistent with high dissatisfaction).

## Deferred

- DD1 for the other 13 chart-only posts (basecamp, brevo, fortinet, insightly,
  intercom, magento, mailchimp, microsoft-defender, sentinelone, tableau,
  teamwork, wrike, zoho-crm) -- sibling slices.
- DD2 (fabricated specifics: "Ira", incoherent multipliers) and DD3 (prose-vs-chart
  "X dominates" the detector misses) -- after DD1.

## Verification

- New `pain_as_strength` detector added to the untracked auditor; `--self-test`
  ALL PASS (3 fixtures). Corpus run found exactly 19 posts.
- All 6 posts re-audited (`--slug=`): `pain_as_strength` = 0; no residual "overall
  satisfaction" false claims (grep clean). clickup + woocommerce still show
  `excluded_source_quote`/`undeclared_quoted_source` -- PRE-EXISTING, fixed by the
  open #837/#838 (different regions; this branch is off origin/main). Not touched
  here.

## Estimated diff size

| Area | LOC |
|---|---:|
| 6 posts (chart move + prose rewrites) | ~50 |
| Plan doc | ~101 |
| **Total** | **~151** |
