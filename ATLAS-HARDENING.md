# ATLAS-HARDENING.md

Parked non-blocking hardening discoveries for the **Atlas blog / deep-dive
content pipeline** work (the `content-ops/blog-*` ownership lanes). Kept separate
from the root `HARDENING.md` so this session's parked items don't collide with
the concurrent content-ops-station sessions. Newest entries first.

Same rules as root `HARDENING.md` (AGENTS.md §3d): do NOT park anything that
breaks the slice's real flow, the AGENTS contract, tests, CI, security, or data
truthfulness -- those are fixed inline. When starting a slice, scan this file for
entries touching the same lane/files; fix only what the slice needs, else leave
parked and note in the plan's Deferred.

## Entry Format

```md
## YYYY-MM-DD

### <short title>
- File/location:
- Description:
- Why it matters:
- Effort: S / M / L
- Category: correctness / polish / tech-debt / security
- Found during:
```

## Parked Items

## 2026-05-25

### Support-ticket FAQ drafts can invent procedural answer steps when tickets lack resolutions
- File/location: `atlas_brain/skills/digest/blog_post_generation.md`, `extracted_content_pipeline/skills/digest/landing_page_generation.md`, support-ticket generated-content evaluator.
- Description: Live support-ticket blog validation showed the model drafting concrete FAQ answers such as account-setting navigation, export menu paths, and verification steps even though the synthetic uploaded tickets only contained subjects/descriptions and no support-resolution fields. This slice tightened outcome/time claims inline, but a broader guard should distinguish "tickets include support resolution text" from "tickets only include customer questions" and require placeholder/review-needed answers when resolution evidence is absent.
- Why it matters: Customer wording can safely drive FAQ questions and prioritization, but final answer steps must come from real support resolutions or product docs. Otherwise the system can create polished but fabricated help-center instructions.
- Effort: M
- Category: correctness
- Owner/session: content-ops/support-ticket-outcome-live-validation
- Found during: PR-Support-Ticket-Outcome-Live-Validation live Haiku blog smoke.

## 2026-05-22

### Some posts DECLARE excluded sources (Capterra/Trustpilot) in their methodology/corpus list
- File/location: methodology source-lists, e.g. `hubspot-deep-dive-2026-03.ts` L143/L144 ("from G2, Capterra, Reddit, Trustpilot ..."; "verified platforms like G2 and Capterra ... community sources like Reddit and Trustpilot"), `switch-to-clickup-2026-03.ts` L124; generator-side `_gather_data` corpus/source counts vs the `_blog_source_allowlist()` used for quotes.
- Description: surfaced while fixing excluded-source QUOTES. Several posts name allowlist-EXCLUDED platforms (Capterra, Trustpilot) in the prose source declaration / corpus counts -- i.e. the ANALYSIS corpus appears to include excluded sources, not just the quote pool. DB-verified, Trustpilot reviews for some vendors ARE enriched (e.g. clickup 110, mailchimp 90), so the enriched counts may genuinely include excluded-source reviews. The excluded-source-QUOTE slices remove the quotes but deliberately leave the methodology declarations untouched (that's a corpus-composition question, not a quote).
- Why it matters: if the blog allowlist is meant to govern the analysis corpus (not just quotes), these posts' counts/source-lists are contaminated and need a recompute; if the allowlist governs only quotes, the declarations are honest and only the quotes needed fixing. Needs the policy clarified before touching the counts. Pairs with the D5 source-list cleanup (52 posts) and the zoho L158 quote-pool-vs-count item below.
- Effort: M (policy decision) / L (recompute if corpus must be allowlist-restricted)
- Category: correctness
- Found during: excluded-source-quote slice (Phase-2)

### pain-radar chart urgency values don't reproduce from a naive all-time aggregate
- File/location: deep-dive `pain-radar` chart data (e.g. `workday-deep-dive-2026-04.ts` L73-111, `microsoft-defender-...-2026-04.ts` L57-96); generator-side `_fetch_blog_signal_rows` / scorecard `avg_urgency_when_mentioned`.
- Description: surfaced while fixing prose_vs_chart. The chart plots `avg_urgency_when_mentioned`, but those per-category values do NOT match a naive all-time aggregate over `b2b_reviews.enrichment` (e.g. Workday chart shows technical_debt 3.2 / data_migration 3.0 as top urgency, but an all-time aggregate has technical_debt freq=1 and pricing/integration higher; MSFT chart Security 3.5 vs all-time security urgency ~1.8). Likely a windowed and/or scorecard-derived computation, not necessarily wrong -- but its provenance isn't reproducible from the obvious query, so "is the chart itself right?" is unverified.
- Why it matters: the deep pass needs a single source of truth for pain ranking; today the prose was made to match the chart (chart treated as authoritative reader-facing artifact), but if the chart's urgency window is itself off, the whole pain section inherits it. Pairs with the D3-followup frequency-view chart item.
- Effort: M
- Category: correctness
- Found during: prose-vs-chart slice (Phase-2)

### teamwork-deep-dive competitive list: "Slack" ungrounded + FAQ(5)/body(6) count mismatch
- File/location: `atlas-churn-ui/src/content/blog/teamwork-deep-dive-2026-04.ts` L145 (FAQ) and L230 (competitive landscape); generator-side `_blueprint_vendor_deep_dive` competitive section sourced from `b2b_product_profiles.commonly_compared_to`.
- Description: surfaced while clearing the `count_vs_list` class (teamwork was a detector FP -- the body lists 6 and says "six", the dotted "Monday.com" had broken the auditor regex). Two real-but-separate issues remain: (1) **grounding** -- L230 names "Slack" as a primary alternative, but DB-verified, Slack does NOT appear in Teamwork's `commonly_compared_to` (Trello 4, Basecamp 3, Asana 2, Monday.com 2, Chaser 1, Coda 1, ...); the top-6 by mentions would not include Slack. (2) **prose-vs-prose** -- L145 (FAQ) enumerates 5 alternatives (Trello, Asana, Basecamp, Monday.com, Slack; no Chaser) while L230 enumerates 6 (adds Chaser, mentions=1). Both numbers can't be the canonical count.
- Why it matters: live content names a comparison vendor the source data doesn't support, and two sections disagree on how many alternatives there are. Not a `count_vs_list` defect (each sentence is internally count-consistent); belongs to the grounding/prose-consistency class for the Phase-2 deep pass (alongside the D5 source-list pattern and a candidate prose-vs-data detector).
- Effort: S (data: align the two lists to the DB top-N, drop ungrounded "Slack") / M (generator: derive the FAQ count from the same capped `comp_names` the body uses)
- Category: correctness
- Found during: count-vs-list slice (Phase-2)

### zoho-crm-deep-dive L158 source list omits Slashdot (a quoted source) -- quote-pool vs corpus-count scope mismatch
- File/location: `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts` L158; generator-side, the quote pool (`_fetch_quotable_reviews`) vs the corpus counts (`_fetch_source_distribution`).
- Description: L158 reads "The data comes from G2, Gartner, PeerSpot, and Reddit -- ... (24 reviews) ... (237 reviews)", but the post quotes a Slashdot reviewer (L166/L168, the D6 fix). The naive fix ("add Slashdot to the community bucket and the 237 reconciles", per the #802 review) does NOT work: DB-verified, the windowed-ENRICHED zoho corpus is reddit(237)/g2(11)/peerspot(8)/gartner(5) -- no Slashdot. The Slashdot quote's review is `enrichment_status=not_applicable` (in-window, Zoho-mention, but NOT enriched), so it's excluded from the 261/237 counts. So the quote pool includes non-enriched reviews the corpus counts exclude.
- Why it matters: live-misleading (a source quoted but undeclared, and it can't simply be added without breaking the verified/community reconciliation). Real fix is a Phase-2 reconciliation of the quote-pool scope vs the corpus-count scope (either restrict quotes to the counted corpus, or count the sources actually quoted). This is the D5 source-list pattern (52 posts) with an extra quote-provenance wrinkle.
- Effort: M
- Category: correctness
- Found during: D6 review (#802); part of the Phase-2 source-list cleanup.

### Deep-dive quote lead-ins can name a platform the quote isn't from
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`, the reviewer/strengths section that surfaces `quote_highlights` to the LLM (around the `quote_highlights = _blog_quote_highlights(...)` call in `_blueprint_vendor_deep_dive`).
- Description: the LLM-written prose lead-in for a quote can name a platform that doesn't match the quote's `source_name`. The blockquote attribution uses `source_name` correctly; only the free-text lead-in drifts. Found in zoho-crm-deep-dive: "verified reviewer on G2" prose against a Slashdot-source quote (fixed in the published post by D6; the generator can recur).
- Why it matters: a prose-vs-attribution mismatch that's data-untruthful but only caught at audit; future deep-dives can repeat it.
- Effort: M (prompt-guidance change in the section goal/data_summary; uncertain effectiveness -- the LLM doesn't always honor narrow phrasing constraints).
- Category: correctness
- Found during: D6

### Orphaned quote lead-in in salesforce-deep-dive (no quote follows)
- File/location: `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts` L159; generator-side, the strengths section in `_blueprint_vendor_deep_dive` that emits a colon-terminated quote lead-in.
- Description: L159 reads "...One verified reviewer on G2 notes the platform's ability to handle intricate workflows:" but no blockquote follows (L160 is unrelated prose) -- a dangling lead-in promising a quote that isn't there. Separate defect class from D6's platform-mismatch (this is an orphaned intro, not a source mismatch); pre-existing, surfaced while scoping D6.
- Why it matters: misleading published content (promises a reviewer quote that never appears). Likely a stripped/never-emitted quote leaving its lead-in. Candidate for a follow-up data fix (drop the dangling lead-in) + a generator guard (don't emit a quote lead-in without a quote).
- Effort: S (data: drop the line) / M (generator guard)
- Category: correctness
- Found during: D6
### LLM may still narrate a partial source set despite the generalized field (D5 residual)
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`, blog generation payload + the external `digest/b2b_blog_post_generation` skill (system prompt).
- Description: D5 stops feeding the per-platform `source_distribution.sources` list to the LLM payload and provides a generalized `source_description`, but the model could still infer/mention platform names from quote `source_name`s or chart labels in the payload, or just not use `source_description`. LLM-fidelity is empirically unverified (same class as the D6 lead-in drift).
- Why it matters: the deterministic lever is closed, but corpus-wide adherence needs checking; this is what the Phase-2 deep pass should verify (and a detector could flag prose source-lists that omit a quoted source).
- Effort: M
- Category: correctness
- Found during: D5

### Unused `_top_source_summary` helper after D5 dropped its only caller
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L2461 (and mirror).
- Description: `_build_coverage_snapshot_note` no longer calls `_top_source_summary`; it's now dead code. (It also had a latent bug — it was fed the `{sources, verified_count, community_count}` dict instead of `{name: count}`.) Remove next pass.
- Why it matters: dead code + a latent bug; removing it is cleanup, deliberately out of D5's thin scope.
- Effort: S
- Category: tech-debt
- Found during: D5

### Deep-dive strengths/weaknesses chart fallback cannot show true strengths
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L8092 (and the byte-identical `extracted_content_pipeline` mirror), the `if len(strengths) + len(weaknesses) < 3 and signals:` fallback in `_blueprint_vendor_deep_dive`.
- Description: when the product profile is too thin, the "Strengths vs Weaknesses" chart is built only from pain-category signals, which carry weakness data only -- so the `strengths` series is always 0 (a one-sided chart). D4 made the bucketing truthful (all pain -> weaknesses) but did not give the fallback a real strengths source.
- Why it matters: thin-profile deep-dives render a two-axis chart that can structurally never populate one axis. A real fix pulls strengths from profile.strengths or positive-mention counts so the chart is genuinely two-sided.
- Effort: M
- Category: polish
- Found during: D4

### "Strengths vs Weaknesses" chart title is two-sided while fallback data is one-sided
- File/location: `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py` ~L8124 (`chart_id="strengths-weaknesses"` title) and the rendered chart titles in published deep-dive posts.
- Description: the chart title implies a strengths-vs-weaknesses comparison; in the signals fallback the chart shows weaknesses only. Title/content mismatch for thin-profile deep-dives.
- Why it matters: minor reader confusion -- a "vs" framing with one empty side. Pairs with the entry above; resolving the strengths source likely resolves this too.
- Effort: S
- Category: polish
- Found during: D4
