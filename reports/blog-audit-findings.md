# Blog correctness audit — findings & pattern log

Per-post grounding audits (correctness, grounded data, contradictions,
hallucinations) of published posts, logged by **defect class** so the
recurring patterns can be fixed at the source (the generator,
`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`) rather than only
patched per post.

**Grounding method (verified on pipedrive):** headline counts reconcile to a
**source-allowlisted** window (exclude `capterra`, `trustpilot`,
`stackoverflow`, `twitter`; `imported_at` in the post's window;
`is_primary` vendor; dedup). Quotes must exist verbatim in `b2b_reviews`
(watch for G2 form-prompt boilerplate). Cross-check prose vs charts vs
internal counts.

---

## Defect classes (root-cause candidates for the code fix)

### D1 — Form-prompt-as-quote (evidence integrity)
G2-style review-form prompts ("What do you like best about <vendor>?")
presented as genuine reviewer quotes. They exist in the corpus as boilerplate
(e.g. "What do you like best about Pipedrive?" appears 19x) but carry no
opinion.
- **Root cause:** quote extraction treats the G2 prompt boilerplate as
  quotable text.
- **Code fix target:** drop form-prompt patterns from quote candidates
  (`What do you like best about`, `What do you dislike about`, etc.) in the
  quote-grade / blueprint quote selection.
- **Also a detector candidate** in the skill audit (same family as
  empty-blockquote / orphaned-ref).
- Instances: `pipedrive-deep-dive` (1); previously seen in
  `top-complaint-every-crm`, `real-cost-of-copper`, `close-deep-dive`,
  `marketing-automation-landscape`.

### D2 — Count-vs-list mismatch + un-deduped vendor variants
A stated count that disagrees with the rendered list, and/or vendor-name
variants double-counted.
- Example: "six primary alternatives: HubSpot, Salesforce, Zoho, Monday, and
  Zoho CRM" — says **six**, lists **five**, and "Zoho"/"Zoho CRM" duplicate
  (4 distinct).
- **Root cause:** competitor count is stated independently of the deduped
  list; vendor aliases not normalized (same family as the Monday/Monday.com
  and HubSpot/HubSpot CRM merges).
- **Code fix target:** derive the count from the deduped competitor list;
  normalize vendor aliases before counting/listing.
- Instances: `pipedrive-deep-dive`.

### D3 — Prose vs chart contradiction (frequency vs urgency)
Prose names a "dominant" pain by one metric while the chart ranks by another.
- Example: "Pricing emerges as the dominant pain category" vs pain-radar chart
  where UX urgency (10.0) > Pricing (7.2).
- **Root cause:** prose conflates frequency-rank and urgency-rank.
- **Code fix target:** make the "dominant pain" sentence state the metric it
  means (frequency) and not contradict the urgency chart, or rank both
  consistently.
- Instances: `pipedrive-deep-dive`.

### D4 — Strengths/weaknesses chart mislabeling
The strengths-vs-weaknesses chart buckets complaint categories under
"strengths."
- Example: `pricing` (51) and `overall_dissatisfaction` (118) appear as
  "strengths"; only `support` (22) is a "weakness."
- **Root cause:** the chart-data builder mislabels pain-mention counts as
  strengths (likely all positive-polarity mention counts dumped into
  "strengths" regardless of category sentiment).
- **Code fix target:** correct the strengths/weaknesses bucketing in the chart
  builder.
- Instances: `pipedrive-deep-dive`. (Suspected systemic across deep-dives.)

### D8 — Vendor-count claim vs incomplete coverage
A "compare N vendors" claim where the charts/profiles cover fewer than N.
- Example (crm-landscape): title/desc say "**8 vendors** compared," but the
  churn-urgency chart lists **7** (Salesforce missing) and only **5** get
  strength/weakness profiles (Close, Copper, Freshsales, Insightly, Nutshell --
  no Pipedrive/Salesforce/Zoho/HubSpot profile).
- **Root cause:** the headline vendor count is stated independently of the
  vendors actually charted/profiled.
- **Code fix target:** derive "N vendors" from the set actually rendered, or
  ensure all N are charted + profiled.
- Instances: `crm-landscape`.

### D9 — Markdown bullet lists inside `<p>` (broken rendering, ANALYZER-BLIND)
Bare-hyphen markdown lists emitted inside an HTML `<p>` block, which the
markdown converter leaves un-converted, so they render as literal "- item"
text instead of a bulleted list.
- Example (crm-landscape): `<p><strong>Top pain points:</strong>\n- Overall
  dissatisfaction\n- Pricing\n- User experience</p>` -- **15 such blocks** in
  one post (every vendor profile's pain/strength/weakness lists).
- **The skill's `detectMarkdownInHtml` does NOT catch this** (returned 0;
  analyzer flagged NOTHING on this post). High-priority detector gap.
- **Root cause:** the generator wraps a strong-label + markdown list in a
  single `<p>`, so md->html skips the inner list.
- **Code fix targets:** (1) generator -- emit real `<ul><li>` (or don't wrap
  the list in `<p>`); (2) skill detector -- flag `^\s*-\s` lines inside `<p>`.
- Instances: `crm-landscape` (15).

### D6 — Quote source attribution mismatch (prose vs blockquote/DB)
The prose lead-in names a different platform than the blockquote attribution
and the actual source in the corpus.
- Example (zoho): prose says "One verified reviewer on **G2** describes...",
  blockquote says "reviewer on **Slashdot**", and the DB confirms the quote is
  from **slashdot** (1 occurrence). The prose "G2" is wrong.
- **Root cause:** the prose lead-in's source label is generated independently
  of the quote's real source field.
- **Code fix target:** derive the prose source label from the quote's actual
  `source`, or drop the platform claim from the lead-in.
- Instances: `zoho-crm-deep-dive`.

### D7 — Headline "total reviews" scope mismatch (overstated corpus)
The headline "total reviews" uses a broader scope (all sources) than the
"enriched" / analysis numbers (allowlist sources only), so the stated total
overstates the base that was actually analyzed -- and it is computed
inconsistently across posts.
- Example (zoho): "940 reviews collected ... with 268 enriched" -- but "940"
  matches all-source window primary (~915), while "268 enriched" matches the
  **allowlist** window (DB: 262 enriched from 431 allowlist total). The 940
  includes Capterra/Trustpilot/etc. that were excluded from the analysis.
- **Contrast (pipedrive):** its "262 total" WAS the allowlist count (264) --
  so the two posts disagree on what "total" means. Inconsistent generator
  behavior.
- **Root cause:** the "total reviews" stat is computed with a different
  source/scope filter than the enriched/churn stats.
- **Code fix target:** compute the headline total over the SAME
  allowlist+window+primary scope as the enriched/churn numbers, so
  "N reviews, M enriched" share a denominator.
- Instances: `zoho-crm-deep-dive` (940 vs ~431 allowlist). NOT in pipedrive
  (used allowlist total) -- the inconsistency itself is the bug.
- **Stronger instance (crm-landscape):** "3,287 enriched reviews" is
  **impossible** -- it exceeds the CRM category's all-time enriched ceiling
  (1,959); window all-source enriched is 1,784, per-vendor-mention double-count
  only 1,821. The "4,990 total" ~ all-source window total (4,609), so a
  plausible total is paired with a fabricated-but-internally-consistent
  enriched count (172 verified + 3,115 community = 3,287). This is the
  headline corpus size, materially overstated.

### D5 — Source-list incompleteness (minor accuracy)
Prose names a partial source set that omits sources the post actually uses and
quotes.
- Example: states "G2, Gartner Peer Insights, PeerSpot... Reddit" but the
  allowlist also includes Slashdot, Software Advice, SourceForge, HackerNews
  (the post quotes a Slashdot analyst and a Software Advice reviewer).
- **Root cause:** prose lists a hardcoded/partial source set, not the actual
  allowlist.
- **Code fix target:** state the real source set (or generalize the phrasing).
- Instances: `pipedrive-deep-dive`.

---

## Per-post audit results

### pipedrive-deep-dive-2026-04 (HubSpot affiliate)
- **Grounded data: PASS.** 262/143/10 reconcile to the allowlisted window
  (264/143/10 -- enriched + churn exact, total off by 2 dedup). "Bird's eye
  view" quote verified (1 occurrence).
- Issues: D1 (form-prompt quote "What do you like best about Pipedrive?"),
  D2 (six-vs-five + Zoho dup), D3 (pricing-dominant vs UX-urgency),
  D4 (pricing/overall_dissatisfaction as "strengths"), D5 (source list).

### zoho-crm-deep-dive-2026-04 (HubSpot affiliate)
- **Grounded data: PARTIAL.** churn 14 **exact**; enriched 268 ~ DB allowlist
  262 (off 6); but **total 940 does NOT match** the same scope -- 940 is
  all-source window (~915), while the analysis is allowlist-only (431 total ->
  262 enriched). "28 verified + 240 community = 268" and "14/268 = 5.2%" are
  internally consistent.
- Issues: **D7** (940 overstates vs ~431 allowlist base; scope inconsistent
  with pipedrive), **D6** (quote attributed to G2 in prose but is from
  Slashdot), **D2** ("six alternatives" lists five + Zoho self-ref),
  **D3** ("Pricing dominates"/"UX ranks second" but radar shows
  contract_lock_in 6.2 > pricing 4.4 and UX only 1.5 -- near the bottom),
  **D4** (overall_dissatisfaction 186 mislabeled "strengths").
- D1 (form-prompt quote) NOT present here.
- **Recurring (systemic -> generator):** D2, D3, D4 now seen in both deep-dives.

### crm-landscape-2026-04 (market_landscape, HubSpot affiliate) -- WORST so far
- **Analyzer flagged NOTHING**, yet 5 defect classes present -- shows the
  analyzer's blind spots (D9, D1, D7, D8 all uncaught).
- **D9 (NEW, major):** 15 markdown bullet-lists inside `<p>` -> render as
  literal "- item" text. Broken formatting on a flagship category post.
- **D7 (strong):** "3,287 enriched / 4,990 total" -- 3,287 exceeds the
  category's all-time enriched ceiling (1,959); overstated ~2-3x. (4,990 ~
  all-source window total 4,609.)
- **D1 (recurs):** 2 G2 form-prompt quotes ("What do you like best about
  Agentforce Sales...", "What do you like best about Zoho CRM") presented as
  reviewer quotes -- the same prompts cleaned from other posts.
- **D8 (NEW):** "8 vendors" but chart shows 7, only 5 profiled.
- **D4 (recurs, in prose):** vendor profiles list `overall_dissatisfaction`
  and `pricing` as "Strengths" (hand-waved as "counterintuitively").
- D3: milder here (prose "top pain category" = frequency, matches chart freq).

### best-crm-for-51-200-2026-04 (best_fit_guide, HubSpot affiliate)
- **D7 (blatant INTERNAL contradiction):** headline says **"1,163 reviews"**
  (title, seo_desc, body, FAQ) but the source breakdown says **"172 verified +
  3,115 community"** = **3,287** -- the same post states two incompatible
  totals. (1,163 was crm-landscape's "total churn signals" number, reused here
  as a review count; 3,287 is the overstated enriched figure pasted from the
  landscape.) Reader-visible factual contradiction.
- **D8 (recurs):** "8 vendors" but the ratings chart omits Zoho's rating
  ("chart does not display its exact rating") and Salesforce/HubSpot get no
  profile; 6 profiled.
- **D4 (milder):** strength/weakness segmentation (pricing/features in both
  lists), hand-waved; reworded to "overall satisfaction" (not the
  overall_dissatisfaction-as-strength mislabel).
- **D9 NOT present** (this post uses prose, not markdown bullet lists) ->
  confirms D9 is post/path-specific, not universal.
- Minor: `<br />` artifact inside a blockquote; "reddit" lowercased in one
  attribution.
- Cross-post number drift: Pipedrive shows **88 reviews** here vs **262** in
  its deep-dive vs **18** in top-complaint-every-crm -- three different
  "in scope" counts for the same vendor across posts.

## Corpus scan with hardened analyzer (D1 + D9 now detected)

Added two detectors to `audit-published-posts.js`:
- `form_prompt_quote` (high) -- G2 boilerplate ("What do you like best about
  ...") presented as quotes.
- extended `markdown_in_html` (critical) -- now scans the whole `<p>` block,
  catching bullet lists that follow a `<strong>label:</strong>` (the old regex
  stopped at the first inner tag and missed them entirely -> previously 0).

Corpus result: **42 clean, 10 with CRITICAL** (was a false "78 clean / 0
CRITICAL").

**D9 -- markdown-in-`<p>` (10 posts, 61 occurrences, broken rendering):**
crm-landscape (15), marketing-automation-landscape (11), helpdesk-landscape
(10), jira-vs-trello (8), mailchimp-deep-dive (5), tableau-deep-dive (4),
salesforce-deep-dive (3), copper-deep-dive (2), switch-to-woocommerce (2),
microsoft-defender-for-endpoint-deep-dive (1).

**D1 -- form-prompt quotes (31 posts, 47 occurrences):**
hr-hcm-landscape (4), best-hr-hcm-for-51-200 (3),
best-project-management-for-201-1000 (3), project-management-landscape (3),
close-vs-zoho-crm (2), crm-landscape (2), insightly-vs-zoho-crm (2),
power-bi-deep-dive (2), switch-to-salesforce (2), top-complaint-every-e-commerce
(2), top-complaint-every-project-management (2), + 20 posts with 1 each
(basecamp, best-crm-for-51-200, fortinet, gusto-vs-workday, intercom,
jira-vs-mondaycom, looker, marketing-automation-landscape, metabase-vs-tableau,
microsoft-defender, pipedrive, salesforce, shopify, slack, switch-to-asana,
switch-to-clickup, switch-to-woocommerce, top-complaint-every-helpdesk,
woocommerce, zoom).

These two lists are the cleanup targets. D2/D3/D4/D7/D8 remain manual/DB-checks
(not yet automated) -- candidates for the generator fix.

## Pattern summary (after 4 posts)
- **Systemic, every applicable post:** D4 (strength/weakness mislabel),
  D7 (corpus-size scope/overstatement), D2/D8 (count vs rendered list).
- **Recurs where quotes exist:** D1 (G2 form-prompt-as-quote).
- **Analyzer-blind (need new/expanded detectors):** D1, D9, D7, D8 (the
  structural detectors only catch D-form issues, not grounding/labeling).
- **Highest publish-risk:** D9 (broken markdown rendering, visible to readers),
  D7 (overstated/false corpus numbers, credibility + factual), D1 (fake quotes).
