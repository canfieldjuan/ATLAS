# Data Issues Log

Tracking data collection, extraction, and enrichment issues discovered during report audits.
Each issue is framed holistically — focusing on systemic pipeline impact, not individual vendors.

---

## Issue 1: Single-Source Entity Resolution — The "Unnamed Signal" Problem

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports can surface hundreds of evaluation signals but resolve only a tiny fraction to named companies. The extraction pipeline identifies company names only when a reviewer explicitly self-attributes (e.g., "I work at X and we're leaving Y"). The vast majority of signals remain anonymous.

### Downstream Impact

- **Signal density is high, but actionable intelligence is low.** A report may show 71 active churn signals yet name only 1 company — making the report feel thin despite strong underlying data.
- **Battle cards and competitive briefings lack specificity.** Sales teams need to know *who* is moving, not just *that someone is moving*.
- **Trend analysis is weakened.** Without entity resolution, we can't detect patterns like "mid-market SaaS companies are all leaving Vendor X" — the cohort stays invisible.

### Potential Investigation Directions

- Cross-reference Source IDs (HackerNews/Reddit links) with external identity signals (LinkedIn activity, company tech-stack databases, IP-to-Company resolution).
- Enrich at ingestion time rather than report-generation time to avoid repeated re-processing.

### Code Investigation (2026-03-26)

**Root cause confirmed.** Entity resolution is purely LLM-based with no external enrichment.

- **Extraction path:** `b2b_reviews.enrichment->>'reviewer_context'->>'company_name'` — populated by Tier 2 LLM extraction (`atlas_brain/skills/digest/b2b_churn_extraction_tier2.md`). The LLM infers company name from review text only.
- **Company signal population:** `_fetch_high_intent_companies()` in `_b2b_shared.py:3083` filters on `reviewer_company IS NOT NULL` — only reviews where the reviewer explicitly named their company or the LLM could extract it.
- **Eligibility gate:** `_company_signal_name_is_eligible()` at line 3126 further filters out names that match vendor names or competitor names — correct behavior but further reduces yield.
- **No external cross-referencing exists.** No CRM lookup, no LinkedIn scraping, no IP-to-Company resolution, no tech-stack database integration. The pipeline has `prospect_org_cache` and `b2b_vendor_firmographics` tables but these enrich *vendors*, not *reviewers*.
- **Normalization exists but doesn't help discovery:** `company_normalization.py` normalizes names (strip suffixes, lowercase) for dedup — useful once you have a name, useless for finding one.

**Conclusion:** The pipeline cannot resolve companies it doesn't already know about. The gap is architectural — no external identity resolution step exists between ingestion and report generation.

---

## Issue 2: Contradictory Narrative Detection — Aggressive Pricing Nuance

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports can label a vendor as a "clear loser" in a competitive landscape while simultaneously surfacing evidence that the same vendor is winning deals back through aggressive discounting (e.g., 90% off). The pipeline does not flag or reconcile these contradictions — both statements appear in the same report without commentary.

### Downstream Impact

- **Misleading competitive positioning.** A battle card that says "Vendor X is losing" while ignoring their fire-sale pricing strategy gives sales reps a false sense of security.
- **Lost strategic insight.** Aggressive discounting is a *signal in itself* — it indicates desperation pricing, budget capture plays, or land-and-expand tactics. Burying it as a footnote wastes high-value intelligence.
- **Report credibility suffers.** Readers who spot the contradiction will question the entire analysis.

### Potential Investigation Directions

- Add a contradiction-detection pass in report generation (or post-processing) that flags when a vendor is simultaneously categorized as "losing" and "winning deals."
- Surface pricing strategy signals as a distinct section in battle cards (e.g., "Competitive Pricing Alerts").

### Code Investigation (2026-03-26)

**Root cause: LLM synthesis without contradiction guardrails.**

- **Battle card rendering** is handled by `battle_card_sales_copy.md` skill, which receives a "render payload" built by `_build_battle_card_render_payload()` in `b2b_battle_cards.py:723`.
- **The payload includes** `vendor_core_reasoning`, `displacement_reasoning`, and `locked_facts` — but no contradiction-detection pass runs before the LLM synthesizes.
- **The LLM skill instructs:** "Every number, quote, and metric MUST come from input. No fabrication allowed." — but it does NOT instruct the LLM to flag when input data contradicts itself.
- **`b2b_cross_vendor_conclusions`** table stores pairwise battle conclusions (`winner`, `loser`, `durability`, `confidence`) but these are computed independently from pricing signals. A vendor can be labeled "loser" in a battle conclusion while pricing data shows aggressive win-back behavior.

**Conclusion:** No contradiction-detection layer exists. The LLM faithfully reproduces contradictory input data without reconciliation. Fix requires either a pre-render coherence check or explicit LLM instructions to flag conflicting signals.

---

## Issue 3: Variable Bloat / Raw Identifiers in Report Output

**Discovered:** 2026-03-26
**Report Type:** Battle Card
**Status:** Open

### Description

Reports contain raw technical artifacts — numbered section markers like `#1, 13`, `#2, 12`, and full Source ID strings — that leak through to the final output. This makes reports look like database exports rather than polished intelligence briefings.

### Downstream Impact

- **Degrades report readability.** End users (sales reps, strategists) see noise that erodes trust in the product.
- **Signals a template/rendering gap.** The report generation layer is not fully abstracting the underlying data model from the presentation layer.
- **Increases support burden.** Users will ask "what does #1, 13 mean?" — generating unnecessary back-and-forth.

### Potential Investigation Directions

- Audit the report template / LLM prompt to ensure raw IDs and internal markers are excluded or mapped to human-readable labels before rendering.
- Add a post-processing sanitization step that strips or transforms technical identifiers.

### Code Investigation (2026-03-26)

**Root cause: `_sid` internal tracking fields leaking into LLM context.**

- **Source IDs** are attached as `_sid` fields in `_b2b_pool_compression.py:157` — `entry["_sid"] = si.source_ref.source_id`. These are internal provenance markers for tracing data back to source reviews.
- **Section markers** (`#1, 13` patterns) likely come from the LLM echoing compressed pool indices that include both rank position and source count.
- **No stripping step exists** between the data assembly layer and the LLM prompt. The `_sid` fields and pool metadata are passed through to the render payload, and the LLM sometimes includes them in output.
- **Related to Issues 6, 7, 13** — all are variants of "internal data structures leaking into rendered output."

**Additional finding:** Quote fallback logic in `_b2b_shared.py:9180` falls back to `str(q)` when neither `quote` nor `text` fields exist on a dict — this stringifies the entire dict including internal keys. Same pattern at line 8146: `quotes[0].get("quote", str(quotes[0]))`.

**Conclusion:** The rendering pipeline has no sanitization pass. Internal tracking fields (`_sid`, pool indices, raw IDs) are included in LLM context and sometimes echoed verbatim. Quote fallback logic also stringifies entire dicts. A post-render regex filter or pre-render field stripping would catch all variants.

---

## Issue 4: De-Anonymization Ceiling — Enrichment Fails on Vague Identifiers

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

The enrichment layer cannot resolve vague organizational descriptors like "Fortune 200 company" or "Big 4 Firm" to specific entities. The intent signal is present in the source data (Reddit, Trustpilot, G2), but the pipeline has no mechanism to map these fuzzy labels to actual companies. This is a variation of Issue 1 (entity resolution), but distinct — here the source *does* provide a partial identifier, and the pipeline still can't close the gap.

### Downstream Impact

- **Near-miss intelligence.** The data is tantalizingly close to actionable — a "Fortune 200" evaluating a switch is valuable, but without knowing *which one*, sales can't act on it.
- **Segment-level analysis is blocked.** We can't aggregate signals by company size, vertical, or account tier if the entities behind vague labels remain unresolved.
- **Report feels speculative.** Citing anonymous archetypes ("a Big 4 firm") reads as hearsay rather than evidence-backed intelligence.

### Potential Investigation Directions

- Build a fuzzy-match enrichment step that cross-references vague descriptors with known tech-stack databases, job postings, or procurement signals to narrow candidates.
- Flag "partially identified" signals separately so reports can at least segment by confidence tier (named → partially identified → anonymous).

### Code Investigation (2026-03-26)

**Root cause: Same as Issue 1, but with an additional missed opportunity.**

- The enrichment JSONB stores `reviewer_context` with fields: `company_size_segment`, `industry`, `role_level`, `decision_maker` — these are populated even when `company_name` is null.
- **Vague descriptors** ("Fortune 200", "Big 4 Firm") may appear in `reviewer_company` raw text from the source, but `_company_signal_name_is_eligible()` would likely reject them as non-specific.
- **`b2b_vendor_buyer_profiles`** table aggregates by `role_type` and `buying_stage` — this data exists but isn't surfaced when company names are missing.

**Conclusion:** Partial identity signals (title, industry, company size) ARE extracted but aren't used as fallback display when full company name resolution fails. The pipeline has a binary view: named or anonymous. A tiered display would immediately improve perceived data quality.

---

## Issue 5: Contradictory Signals Between Report Sections — Narrative Coherence Gap

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

Different sections of the same report can assert contradictory facts without reconciliation. For example, a Target Accounts section may show "0 companies considering Vendor Y," while the Head-to-Head section says "Vendor Y is displacing Vendor X." This creates a narrative gap — the displacement signal exists but isn't reflected in the pipeline counts. This is a broadening of Issue 2 (contradiction detection), now observed across *structural sections* of a report, not just within a single narrative.

### Downstream Impact

- **Erodes trust in the report as a whole.** If two sections disagree, readers don't know which to believe.
- **Signals a data flow disconnect.** The displacement graph and the target-account pipeline are likely pulling from different query scopes or time windows, producing inconsistent views of the same reality.
- **Missed early-stage signals.** A displacement trend emerging in reviews may not yet appear in structured "consideration" data — but the report should explain that gap, not ignore it.

### Potential Investigation Directions

- Audit whether displacement signals and target-account counts query the same underlying data or different snapshots/time ranges.
- Add a coherence-check pass that cross-references section assertions and flags contradictions or explains the discrepancy (e.g., "Displacement signal is emerging but no accounts have formally entered evaluation stage yet").

### Code Investigation (2026-03-26)

**Root cause confirmed: Target accounts and displacement signals query different data sources.**

- **Target accounts** come from `b2b_company_signals` table (migration 099) — requires explicit `company_name` + `vendor_name` pairs. Only populated when a reviewer names their company AND expresses churn intent.
- **Displacement signals** come from `b2b_displacement_edges` table — aggregated from review mentions of switching between vendors, regardless of whether the reviewer is identified.
- **Different populations:** Displacement edges can exist with zero corresponding company signals. A reviewer can say "we're moving from X to Y" without naming their company — this creates a displacement edge but no company signal.
- **No cross-reference exists** between these two query paths in the challenger brief assembly (`b2b_challenger_brief.py`).

**Conclusion:** The two data sources have fundamentally different coverage. Displacement signals are high-coverage (any mention counts), while target accounts are low-coverage (requires named company). The report should acknowledge this asymmetry rather than presenting both as if they draw from the same pool.

---

## Issue 6: Template Variable Leakage — Code Strings in Report Output

**Discovered:** 2026-03-26
**Report Type:** Challenger Brief
**Status:** Open

### Description

Raw template variable names (e.g., `displacement_detail.primary_driver`) are appearing verbatim in rendered report text. This indicates the LLM prompt or report template has unresolved placeholders that are not being substituted before output — the code is "leaking" into the final deliverable. This is related to Issue 3 (variable bloat) but more severe: Issue 3 involved raw IDs/markers, while this involves actual code-level variable paths.

### Downstream Impact

- **Immediate credibility hit.** A report containing `displacement_detail.primary_driver` looks broken, not professional.
- **Indicates fragile template logic.** If one variable leaks, others may be silently failing too — meaning some report sections could be rendering with missing or default data without anyone noticing.
- **Compounds across report types.** If the same template engine serves multiple report types, the leakage pattern likely affects more than just challenger briefs.

### Potential Investigation Directions

- Audit all report templates/prompts for unresolved variable references and ensure fallback/default handling exists for every placeholder.
- Add a post-render validation step that rejects or flags output containing common variable patterns (e.g., anything matching `*.primary_driver`, `*.detail.*`, etc.).

### Code Investigation (2026-03-26)

**Root cause: LLM echoing data structure paths from render payload.**

- **`displacement_detail`** is a key in the render payload passed to the LLM. The `build_displacement_dynamics()` function (`_b2b_shared.py:7245-7431`) builds objects with fields like `primary_driver`, `signal_strength`, `key_quote`.
- **The battle card / challenger brief skill** receives these as nested JSON. When the LLM can't resolve a field (e.g., `primary_driver` is null or empty), it sometimes outputs the *path name* instead of omitting it.
- **No output validation** checks for dotted-path patterns in rendered text.

**Exact code found:** `b2b_challenger_brief.py:1311-1324` — the `evidence` field is **hardcoded as a literal string**, not a variable reference:
```python
insights.append({
    "insight": "Primary displacement driver: %s" % driver,
    "evidence": "displacement_detail.primary_driver"  # ← LITERAL STRING, not resolved
})
```
Same pattern at line 1317 (`"evidence": "stratified_reasoning.archetype"`) and line 1324 (`fallback to "product_profile"`).

**Conclusion:** This is NOT the LLM echoing field paths — it's the Python code itself writing literal variable-path strings into the `evidence` field. These strings are then passed to the LLM and rendered into the report verbatim. Fix requires replacing the hardcoded strings with actual resolved values from the data.

---

## Issue 7: Unlabeled Ranked Lists — Data Without Context

**Discovered:** 2026-03-26
**Report Type:** Displacement Report
**Status:** Open

### Description

Ranked lists in reports render as bare numbers without labels (e.g., `#2: 66`, `#3: 61`). The underlying data contains the entity names, but the rendering step only outputs the rank position and count — stripping the most important piece: *what* is being ranked. This is a sibling of Issues 3 and 6 (rendering/template gaps), but distinct because the data exists and is correct; it's purely a presentation failure.

### Downstream Impact

- **Entire report sections become useless.** A top-50 list with no labels is dead space — readers can't act on it, reference it, or even understand it.
- **Wastes the strongest signal.** Displacement flow rankings are among the highest-value outputs the pipeline produces. Rendering them without names throws away the payoff.
- **Suggests the serialization layer is dropping fields.** If the LLM or template receives a list of `{rank, name, count}` objects but only prints `{rank, count}`, there may be a field-mapping or context-window issue at play.

### Potential Investigation Directions

- Check whether the ranked data passed to the LLM/template includes entity names or only IDs that require a separate lookup.
- If names are present in the data, audit the prompt/template to ensure the name field is referenced in the output format.
- If names are missing, trace back to the query layer to confirm the join to the vendor registry is happening.

### Code Investigation (2026-03-26)

**Root cause: Displacement map returns `from_vendor` and `to_vendor` names, but LLM drops them in output.**

- **`_build_deterministic_displacement_map()`** at `_b2b_shared.py:8618-8727` returns results with `from_vendor`, `to_vendor`, `mention_count`, `primary_driver`, `signal_strength` — vendor names ARE present.
- **Results sorted by `mention_count` descending** (line 8726) and capped at top N (typically 50).
- **The data reaches the LLM** via the render payload, but the LLM skill (`b2b_churn_intelligence.md`) outputs a ranked list that sometimes drops the vendor names, rendering only `#rank: count`.
- **Context window pressure** may be a factor — with 50 displacement edges, each with multiple fields, the LLM may be truncating or simplifying output.

**Conclusion:** The data is complete. The presentation failure is in the LLM's output formatting. The skill prompt should explicitly require `"from → to: count (driver)"` format for ranked lists, or the rendering should be deterministic (not LLM-generated) for structured data like ranked lists.

---

## Issue 8: Competitor Names Extracted as Feature Drivers — Misclassification Opportunity

**Discovered:** 2026-03-26
**Report Type:** Displacement Report
**Status:** Open — Flagged as Potential Feature

### Description

The driver extraction pipeline sometimes surfaces competitor names as displacement drivers (e.g., "Flexibility and depth comparable to Tableau" classified as a 6% driver). This happens because reviewers use well-known products as shorthand for feature expectations — "I want Tableau-like dashboards" — and the extraction treats the entire phrase as a driver string.

### Downstream Impact

- **Not a bug in isolation, but needs classification.** This is actually valuable signal — it reveals which products serve as the *mental benchmark* in a category. Knowing that users measure alternatives against a specific product is strategic intelligence.
- **Without classification, it muddies driver analysis.** If competitor-as-benchmark signals are mixed in with true feature drivers (pricing, UX, integrations), aggregate driver breakdowns become noisy. "Tableau" isn't a *feature* — it's a *reference point*.
- **Opportunity for a new signal type.** Separating "benchmark mentions" from "feature drivers" would create a distinct intelligence layer: which products define the category standard in users' minds.

### Potential Investigation Directions

- Add a classification step in driver extraction that detects when a driver string contains a known vendor/product name and tags it as "benchmark reference" vs. "feature driver."
- Surface benchmark references as their own report section (e.g., "Category Benchmarks — products users compare against").
- Cross-reference with the vendor registry to auto-detect product names in driver strings.

### Code Investigation (2026-03-26)

**Root cause: Driver extraction uses a fixed priority taxonomy, no vendor-name detection.**

- **Primary driver determination** in `_b2b_shared.py:8651-8682` uses a `canonical_priority` dict with 10 categories: pricing (9), features (8), integration (7), ux (6), support (5), reliability (4), security (3), compliance (2), performance (1), migration (1).
- **`_normalize_displacement_driver_label()`** normalizes raw driver strings to these canonical categories via keyword matching.
- **No cross-reference with `b2b_vendors` registry.** If a driver string contains "Tableau-like dashboards," it would match "features" (if it matches at all) — the vendor name embedded in the phrase is invisible to the classification.
- **Vendor registry** (`b2b_vendors` table, migration 095) has `canonical_name` and `aliases` (JSONB) — this could be used for detection but isn't wired into driver extraction.

**Conclusion:** Adding a pre-classification step that checks driver strings against `b2b_vendors.aliases` (GIN-indexed) would separate benchmark references from feature drivers at near-zero cost.

---

## Issue 9: Zeroed-Out Metrics — Boolean/Null Parsing Failure

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open

### Description

Computed metrics (e.g., "Recommend Ratio") render as `0` for every vendor in the scorecard, regardless of review volume. A vendor with 500+ reviews cannot have a 0% recommendation rate — this points to a parsing failure where a boolean, null, or missing field is being coerced to `0` instead of being calculated or flagged as unavailable.

### Downstream Impact

- **Immediate credibility killer.** The Vendor Scorecard is positioned for C-Suite and Strategic Partnerships — the most scrutinizing audience. A column of zeros will be spotted instantly and discredit the entire report.
- **Silent data corruption pattern.** If one metric is zeroed out, others may be too — but less obviously. Metrics that *should* be low might appear correct by coincidence while actually being wrong.
- **Breaks comparative analysis.** The scorecard's core value is vendor-vs-vendor comparison. If a key metric is uniformly zero, that comparison dimension is completely eliminated.

### Potential Investigation Directions

- Trace the "recommend" field from source ingestion → extraction → aggregation → report rendering to find where the value drops to 0 or null.
- Check whether the source data uses a non-standard representation (e.g., "Yes"/"No" string instead of boolean, or a nested field the extractor isn't reaching).
- Add validation that flags any metric that is identical across all vendors in a scorecard — statistically impossible uniformity is a reliable bug signal.

### Code Investigation (2026-03-26)

**Root cause identified: `would_recommend` enrichment field is likely NULL for most/all reviews.**

- **Formula** at `_b2b_shared.py:8827`: `recommend_ratio = round(((recommend_yes - recommend_no) / total_reviews) * 100, 1) if total_reviews else 0.0`
- **`recommend_yes` / `recommend_no`** extracted via SQL FILTER clause at line 2927-2932: `count(*) FILTER (WHERE (enrichment->>'would_recommend')::boolean = true)`.
- **`would_recommend`** is a Tier 2 enrichment field (`b2b_churn_extraction_tier2.md:195`): "True, false, or null (not expressed). Infer from overall tone and explicit recommendation language."
- **When `would_recommend` is NULL for all reviews:** both `recommend_yes` and `recommend_no` = 0, formula yields `(0 - 0) / total * 100 = 0.0` — exactly what we're seeing.
- **Four possible causes:** (1) Tier 2 enrichment hasn't run on these reviews, (2) Tier 2 ran but LLM returns null for ambiguous reviews, (3) Reviews were enriched before `would_recommend` was added to the schema, (4) The `::boolean` cast fails silently on non-boolean strings.
- **Contrast with `b2b_product_profiles.recommend_rate`** (line 295-296) which uses `AVG(CASE WHEN ... THEN 1.0 ELSE 0.0 END) FILTER (WHERE ... IS NOT NULL)` — a more robust formula that excludes NULLs from the denominator. The scorecard formula divides by `total_reviews` (all reviews), not just reviews with the field populated.

**Diagnostic query:**
```sql
SELECT vendor_name,
       COUNT(*) as total,
       COUNT(*) FILTER (WHERE enrichment->>'would_recommend' IS NOT NULL) as have_recommend
FROM b2b_reviews
GROUP BY vendor_name ORDER BY total DESC LIMIT 20;
```

**Conclusion:** Most likely Tier 2 enrichment gap — `would_recommend` is NULL across the board. Even if populated, the formula should use `recommend_yes + recommend_no` as denominator (like product_profiles does), not `total_reviews`, to avoid dilution by reviews without the field.

---

## Issue 10: Uncategorized Pain Points — The "Other" Bucket Problem

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open

### Description

The pain-point extraction pipeline produces "Other" as a top category for some vendors. In a scorecard designed for strategic decision-makers, "Other" is a wasted slot — it means the LLM could not confidently classify the complaint into an existing taxonomy and gave up. This is especially damaging when "Other" is the *#1 pain point*, as it implies the pipeline doesn't understand the most common complaint.

### Downstream Impact

- **Wastes the most valuable real estate in the report.** The top pain point slot in a scorecard drives strategic narratives. Filling it with "Other" communicates nothing.
- **Hides actionable intelligence.** The underlying reviews that got bucketed into "Other" likely *do* contain classifiable complaints — technical debt, integration complexity, contract lock-in, poor support — but the extraction gave up too early.
- **Skews competitive comparison.** If Vendor A's top pain is "Pricing" and Vendor B's is "Other," the comparison is meaningless. Vendor B looks better by default because its real pain is hidden.

### Potential Investigation Directions

- Force the classification model to choose from a defined taxonomy rather than allowing an "Other" escape hatch. If confidence is low, use a secondary classification pass or expand the taxonomy.
- Post-process: when "Other" exceeds a threshold (e.g., >10% of complaints for a vendor), re-run classification on those reviews with a more granular prompt.
- Audit what's actually *in* the "Other" bucket — it may reveal taxonomy gaps (categories the pipeline doesn't have yet).

### Code Investigation (2026-03-26)

**Root cause: Fixed 10-category taxonomy with "other" as a valid value.**

- **`b2b_vendor_pain_points`** table (migration 100) has a CHECK constraint limiting `pain_category` to: `'pricing', 'support', 'features', 'ux', 'reliability', 'performance', 'integration', 'security', 'onboarding', 'other'`.
- **The LLM extraction** classifies each pain into one of these 10 categories. When the complaint doesn't clearly fit, "other" is the escape hatch.
- **No secondary classification** runs on the "other" bucket. Once categorized as "other," the data stays there permanently.
- **The `canonical_priority` dict** in displacement driver extraction (line 8651) mirrors this taxonomy — "other" is not in the priority list, meaning it gets lowest weight but still surfaces as a category in reports.

**Conclusion:** The taxonomy is too coarse for some vendor domains. Common complaints like "technical debt," "contract lock-in," "data migration difficulty," and "API limitations" don't map cleanly to the 10 categories. Expanding the taxonomy or adding a sub-classification step for "other" would reduce the bucket size.

---

## Issue 11: Emerging Displacement Signals in Unexpected Segments

**Discovered:** 2026-03-26
**Report Type:** Vendor Scorecard
**Status:** Open — Flagged as Potential Feature

### Description

The scorecard surfaces low-volume but strategically significant displacement signals in unexpected product segments (e.g., a community/collaboration tool threatening an enterprise meeting platform — not the expected enterprise competitors). These signals are currently buried alongside high-volume flows without differentiation.

### Downstream Impact

- **High-value signal hidden in plain sight.** Unexpected displacement patterns (a tool from an adjacent category encroaching) are often the earliest indicators of market disruption. They deserve amplification, not burial.
- **Segment misattribution.** The pipeline treats all displacement equally — it doesn't distinguish "enterprise competitors swapping" from "category boundary erosion." These are fundamentally different strategic signals.
- **Missed narrative opportunity.** A scorecard that says "your competition isn't just who you think it is" is far more valuable to a C-Suite audience than one that confirms known rivalries.

### Potential Investigation Directions

- Tag displacement flows with a "same-category" vs. "cross-category" flag based on vendor registry metadata.
- Surface cross-category displacements in a dedicated "Emerging Threats" or "Category Boundary Shifts" section.
- Weight low-volume cross-category signals higher in strategic reports, as they indicate early-stage disruption.

---

## Issue 12: Self-Referential Challenger — Incumbent Echo Loop

**Discovered:** 2026-03-26
**Report Type:** Category Overview
**Status:** Open

### Description

The "Emerging Challenger" logic sometimes returns the incumbent vendor as its own primary challenger (e.g., "AWS is the primary alternative to AWS," "Gusto is the primary challenger to Gusto"). The pipeline has no filter to exclude the subject vendor from its own challenger/alternative output. When no clear challenger exists in the data, the query falls back to the highest-signal vendor — which is often the incumbent itself.

### Downstream Impact

- **Reads as a hallucination.** To a C-Suite reader viewing a Market Heatmap, a vendor challenging itself looks like the system is broken or the LLM is hallucinating. This is the single fastest way to lose credibility.
- **Masks real competitive dynamics.** The self-reference hides what's actually happening — either the market is fragmented (no clear challenger) or the pipeline failed to surface a real one. Both are useful signals; the echo loop is not.
- **Systemic, not one-off.** Observed in multiple categories (Cloud Infrastructure, HR/HCM), meaning the logic gap is in the shared challenger-resolution code, not a category-specific edge case.

### Potential Investigation Directions

- Add an exclusion filter in the challenger/alternative resolution query that removes the incumbent vendor from the candidate set.
- When no challenger clears the threshold after exclusion, return a structured label like "Fragmented" or "No clear challenger" rather than defaulting to the top result.
- Audit all report types that use challenger resolution — this likely affects Displacement Reports and Battle Cards too.

### Code Investigation (2026-03-26)

**Root cause confirmed: No incumbent exclusion filter in challenger selection.**

- **Exact code** at `_b2b_shared.py:9107-9113`:
  ```python
  category_flows: dict[str, int] = {}
  for flow in competitive_disp:
      source_vendor = _canonicalize_vendor(flow.get("vendor") or "")
      if any(_canonicalize_vendor(r.get("vendor_name") or "") == source_vendor for r in rows):
          competitor = _canonicalize_competitor(flow.get("competitor") or "")
          category_flows[competitor] = category_flows.get(competitor, 0) + int(flow.get("mention_count") or 0)
  emerging = max(category_flows.items(), key=lambda item: item[1])[0] if category_flows else "Insufficient data"
  ```
- **The bug:** `category_flows` aggregates ALL competitors mentioned in displacement flows for the category. If the incumbent (`highest_vendor`, set at line 9103) appears as a `competitor` in any flow, it enters the candidate set. With enough self-mentions, it wins the `max()`.
- **No exclusion:** There is zero filtering of `highest_vendor` from `category_flows` before the `max()` selection.
- **Result flows to:** line 9227 (`"emerging_challenger": emerging`) → line 2693 (narrative: `"{challenger} is emerging as the primary alternative to {incumbent}"`).
- **Fix is surgical:** Add `if competitor != highest_vendor:` guard before the `category_flows` accumulation, and change fallback from `max()` to `"Fragmented"` when `category_flows` is empty after filtering.

**Conclusion:** One-line fix. Add incumbent exclusion before line 9112. This is the most straightforward fix of all 18 issues.

---

## Issue 13: Object Serialization Leak — `[object Object]` in Output

**Discovered:** 2026-03-26
**Report Type:** Category Overview
**Status:** Open

### Description

The Cross Vendor Analysis section contains `[object Object]` strings — a JavaScript/JSON rendering error where a nested data structure is being coerced to a string instead of being properly serialized or destructured. This is the third variant of the rendering failure pattern (joining Issues 3, 6, 7): raw IDs → template variables → now raw object references.

### Downstream Impact

- **Looks like broken software.** `[object Object]` is universally recognized as a code bug, even by non-technical readers. In a report positioned as a Market Heatmap for strategic buyers, this is disqualifying.
- **Data is present but invisible.** The object *contains* the intended data — it's just not being accessed correctly. This means the report is actively throwing away information it already has.
- **Points to a type mismatch in the rendering pipeline.** Somewhere between data retrieval and report output, a structured object is being passed where a string is expected. This class of bug tends to be widespread once it appears.

### Potential Investigation Directions

- Trace the Cross Vendor Analysis data flow to find where an object is passed to a string context (likely a template interpolation or JSON.stringify gap).
- Add type checking in the rendering layer that catches non-string values before they reach output — log them for debugging rather than printing `[object Object]`.
- Related to Issues 3, 6, 7 — consider a unified post-render sanitization pass that catches all rendering leak patterns (`#1, 13`, `displacement_detail.*`, `[object Object]`, raw Source IDs).

### Code Investigation (2026-03-26)

**Root cause: Nested dicts/objects passed to string context in report assembly.**

- **Category overview** data is assembled in `_build_category_overview()` (~line 9050 in `_b2b_shared.py`). Cross-vendor analysis sections include nested objects (e.g., pairwise battle conclusions from `b2b_cross_vendor_conclusions` with structure `{winner, loser, key_insights, durability, confidence}`).
- **When these objects are interpolated into narrative strings** (f-strings or LLM prompt context), Python renders them as `{'winner': 'X', ...}` but if the report passes through any JavaScript layer (e.g., a frontend renderer or PDF export), nested objects become `[object Object]`.
- **The `intelligence_data` JSONB column** stores the full report payload. If a downstream consumer (web UI, PDF exporter) reads this JSONB and tries to render nested objects as strings, `[object Object]` appears.
- **This is likely a consumer-side bug**, not a pipeline bug — the data in PostgreSQL is correct, but the rendering client doesn't handle nested objects.

**Exact code found:** `b2b_battle_cards.py:1685` uses `json.dumps(payload, default=str)` — the `default=str` fallback stringifies any non-serializable object (dicts, datetimes, custom objects) into their Python `str()` representation. Same pattern at `_b2b_shared.py:1729` and `2299`. If any downstream consumer (web UI, PDF exporter) parses this JSON and re-renders nested objects, JavaScript produces `[object Object]`.

**Conclusion:** Two-pronged: (1) Replace `default=str` with a custom JSON encoder that properly flattens nested structures, and (2) any frontend consumer should handle nested objects gracefully. The pipeline-side fix is more reliable since it catches all consumers.

---

## Issue 14: Low-Sample-Size Signals in Aggregate Reports — Statistical Noise

**Discovered:** 2026-03-26
**Report Type:** Category Overview
**Status:** Open

### Description

The Category Overview includes "Market Shift Signals" based on vendors with extremely low review counts (e.g., 8 reviews showing 0.0% churn density). While technically accurate for that sample, featuring it alongside vendors with 500+ reviews in an aggregate market view presents statistical noise as if it were a meaningful trend. There is no minimum review threshold gating inclusion in overview-level reports.

### Downstream Impact

- **Ruins statistical integrity.** A Market Heatmap is an aggregate strategic tool. Including data points with n=8 alongside n=500+ makes the entire visualization unreliable — readers can't tell which signals are robust and which are noise.
- **Creates false confidence or false alarm.** A vendor showing "0.0% churn" on 8 reviews looks like a fortress; one showing "50% churn" on 4 reviews looks like a disaster. Both are meaningless at that sample size.
- **Dilutes high-value signals.** The strong, statistically significant displacement flows (like the 110-mention patterns in Issue 7) get crowded out by noise from vendors that barely register in the data.

### Potential Investigation Directions

- Implement a minimum review threshold (e.g., 50 reviews) for inclusion in aggregate/overview reports. Individual vendor reports can still show low-count data with a confidence disclaimer.
- Add a confidence tier to all metrics: "High confidence (n>100)," "Moderate (50-100)," "Low (<50, interpret with caution)."
- Separate "statistically significant signals" from "emerging/early signals" in report layout so readers can weight them appropriately.

### Code Investigation (2026-03-26)

**Root cause: Confidence levels exist but don't gate inclusion in aggregate reports.**

- **`_build_deterministic_vendor_feed()`** at `_b2b_shared.py:8454-8463` assigns confidence levels by review count:
  - `>= 50` reviews → "high" confidence
  - `>= 20` reviews → "medium" confidence
  - `< 20` reviews → "low" confidence
- **No minimum threshold prevents inclusion.** Vendors with even 1 review pass the filter if they meet the OR-gate: `churn_density >= 15 OR avg_urgency >= 6 OR dm_rate >= 0.3`.
- **`churn_pressure_score`** formula does apply a confidence multiplier (line ~8480): `confidence = 1.0` for n>=50, `0.85` for n>=20, `0.65` for n<20. But even at 0.65x, a vendor with extreme metrics on 8 reviews can still score high enough to appear.
- **Category overview** has no separate threshold — it inherits whatever vendors are in the feed.

**Conclusion:** The confidence multiplier dampens scores but doesn't exclude. A hard minimum (e.g., n>=20 for feeds, n>=50 for aggregate overviews) would eliminate statistical noise without losing emerging signals — those can go in a separate "Early Signals" section.

---

## Issue 15: Bidirectional Flow Blindness — No Net Flow Metric

**Discovered:** 2026-03-26
**Report Type:** Weekly Feed
**Status:** Open — Highest Priority Logic Issue

### Description

The pipeline tracks churn *from* a vendor and displacement *to* a vendor independently, but never reconciles them. A vendor can simultaneously appear as "High Churn Risk" (everyone leaving) and "Top Displacement Target" (everyone arriving) in the same report. Without a net flow calculation, the report presents two contradictory truths with no resolution.

### Downstream Impact

- **The most damaging contradiction yet.** Unlike Issues 2/5/12 where contradictions appear in different sections or require close reading, this one is front-and-center: the same vendor name appears in both the "losing" and "winning" lists.
- **Blocks the most basic strategic question.** "Is Vendor X growing or shrinking?" is the first thing a reader asks. The report currently can't answer it.
- **Misallocates sales effort.** If a vendor is a "Leaking Bucket" (losing more than gaining), it's a prime target. If it's a "Market Aggregator" (gaining more than losing despite visible churn), attacking it is a waste of resources. Without net flow, sales teams can't tell the difference.

### Potential Investigation Directions

- Introduce a **Net Flow metric**: `inbound displacement signals - outbound churn signals` per vendor, per time window.
- Classify vendors into archetypes based on net flow: "Leaking Bucket" (net negative), "Market Aggregator" (net positive), "Churning Equally" (net zero with high volume on both sides).
- Surface net flow prominently in any report that shows both churn and displacement data — it should be impossible to show one without the other.

### Code Investigation (2026-03-26)

**Root cause confirmed: No net flow calculation exists anywhere in the pipeline.**

- **`_build_deterministic_vendor_feed()`** (line 8374) operates on outbound churn data only — `churn_density`, `avg_urgency`, `dm_churn_rate`. It does not query inbound displacement.
- **`_build_deterministic_displacement_map()`** (line 8618) operates on displacement edges only — `from_vendor → to_vendor` pairs. It does not cross-reference with the vendor feed.
- **Head-to-head query** (`_read_head_to_head_displacement()` at `b2b_churn_intelligence.py:4491`) DOES fetch bidirectional edges for a specific vendor pair, but returns them as separate rows — never nets them.
- **`b2b_displacement_edges`** table has `from_vendor` and `to_vendor` columns. A net flow query would be: `SUM(mention_count) WHERE to_vendor = X` minus `SUM(mention_count) WHERE from_vendor = X` — straightforward SQL but not implemented.
- **The two functions (`vendor_feed` and `displacement_map`) share no cross-referencing.** They operate on separate input datasets assembled at different pipeline stages.

**Conclusion:** Net flow is a missing calculation, not a hidden one. The data exists in `b2b_displacement_edges` to compute it. Implementation would require a new aggregation query and a new field in the vendor feed output.

---

## Issue 16: Zero Entity Resolution in High-Volume Feeds

**Discovered:** 2026-03-26
**Report Type:** Weekly Feed
**Status:** Open

### Description

A Weekly Feed with 3,447 reviews across 32 vendors surfaces zero named companies. This is a reappearance of Issues 1 and 4 (entity resolution), but at the worst possible scale — a feed designed for Sales Managers who need to act *this week* on specific accounts. Even a 5% resolution rate would yield ~170 named signals. Currently: zero.

### Downstream Impact

- **Renders the feed non-actionable for its primary audience.** Sales Managers need "Director at [Company] is evaluating alternatives to [Vendor]" — not anonymous aggregate counts. Without names, the feed is a market research document, not a sales tool.
- **The gap between data volume and actionability is at its widest here.** 3,447 reviews is an impressive number that sets high expectations. Delivering zero names against that volume feels like a broken promise.
- **Compounds Issue 1 and Issue 4.** This is now the third report type where entity resolution failure is the primary blocker. It's no longer an edge case — it's the pipeline's single biggest capability gap.

### Potential Investigation Directions

- Prioritize entity resolution for the Weekly Feed above other report types, since it has the most time-sensitive audience.
- Even partial resolution adds massive value: "VP-level at a Fortune 500 *[financial services]* evaluating alternatives" is 5x more actionable than fully anonymous.
- Consider a tiered resolution display: Named → Partially Identified (title + industry) → Anonymous, so readers can see the pipeline is working even when full resolution isn't possible.

### Code Investigation (2026-03-26)

**Root cause: Same as Issue 1 — plus the feed's company list construction confirms the gap.**

- **Company list in vendor feed** (`_b2b_shared.py:8521-8535`) includes fields: `company`, `urgency`, `title`, `company_size`, `industry`, `source`, `buying_stage`, `confidence_score`, `decision_maker`, `first_seen_at`, `last_seen_at`.
- **These come from merged `company_lookup`** which combines `_fetch_high_intent_companies()` (review-extracted) and `_fetch_existing_company_signals()` (persisted `b2b_company_signals`).
- **Both sources require `reviewer_company IS NOT NULL`** — if no reviews for a vendor have an extractable company name, the list is empty.
- **The feed correctly structures the data for display** — the problem is upstream (no names to display), not in the feed builder itself.

**Conclusion:** Confirms that the Weekly Feed's zero-name problem is purely an extraction yield issue, not a rendering bug. The feed builder is ready to display companies — it just gets an empty list. Fixing Issue 1 (external entity resolution) would immediately populate this.

---

## Issue 17: Scroll Fatigue — Flat List Format in Time-Sensitive Reports

**Discovered:** 2026-03-26
**Report Type:** Weekly Feed
**Status:** Open

### Description

The Weekly Feed renders as a long, flat list of vendor-level signals. In a weekly cadence report, readers need to triage quickly — but the current format requires scrolling through everything to find what matters. There is no grouping by strategic theme, urgency tier, or action type.

### Downstream Impact

- **Time-to-insight is too high.** A weekly report that takes 20 minutes to read defeats its purpose. Sales Managers will skim, miss critical signals, or stop reading entirely.
- **All signals appear equally important.** Without grouping, a "Fire Drill" vendor (high urgency + high pressure) sits alongside a "Slow Burn" (low urgency + high pressure) with no visual or structural differentiation.
- **Missed opportunity for strategic framing.** Grouping by archetype (e.g., "Pricing Crisis" vendors vs. "UX Decay" vendors) would transform the feed from a data dump into a strategic playbook with implied action items.

### Potential Investigation Directions

- Group feed entries by displacement driver archetype (Pricing Crisis, UX Decay, Feature Gap, Integration Pain) with per-group action guidance.
- Add an executive summary / "Top 3 This Week" section at the top for rapid triage.
- Limit the full feed to a configurable depth (e.g., top 10 by urgency) with a "See all" expansion, rather than rendering everything flat.

### Code Investigation (2026-03-26)

**Root cause: Feed builder produces a flat sorted list with no grouping logic.**

- **`_build_deterministic_vendor_feed()`** sorts vendors by `churn_pressure_score` descending (line ~8540) and outputs a flat array.
- **Archetype data exists** but isn't used for grouping: `b2b_churn_signals.archetype` stores per-vendor archetypes (`pricing_shock`, `feature_gap`, `acquisition_decay`, `support_collapse`, `category_disruption`, etc.) — these map directly to the suggested groupings.
- **The LLM skill** (`b2b_churn_intelligence.md`) outputs `weekly_churn_feed` as a flat ranked list. No instruction to group by archetype or create an executive summary.

**Conclusion:** The archetype taxonomy already exists in the data. Grouping the feed by archetype requires: (1) including `archetype` in the feed output, (2) updating the LLM skill to group entries and add per-group action guidance.

---

## Issue 18: Composite Vulnerability Score — Missing Strategic Metric

**Discovered:** 2026-03-26
**Report Type:** Weekly Feed
**Status:** Open — Flagged as Premium Feature Opportunity

### Description

The pipeline tracks "Pressure" and "Urgency" as separate dimensions but doesn't combine them into a single actionable score. Sales teams must mentally cross-reference two metrics to prioritize accounts. A composite **Vulnerability Score (0-100)** would collapse this into one number that directly maps to action priority.

### Downstream Impact

- **Without a composite score, prioritization is manual and inconsistent.** Each sales rep will weigh Pressure vs. Urgency differently, leading to inconsistent outreach strategies across the team.
- **Misses the strategic quadrant.** The two dimensions create four natural archetypes that map to distinct sales motions:
  - **High Urgency + High Pressure ("Fire Drill"):** Act now — these accounts are actively leaving.
  - **Low Urgency + High Pressure ("Slow Burn"):** Nurture — unhappy but sticky, respond to educational/long-term plays.
  - **High Urgency + Low Pressure ("Window Shopper"):** Evaluating but not in pain — needs a compelling trigger.
  - **Low Urgency + Low Pressure ("Stable"):** Deprioritize — not a near-term opportunity.
- **Premium positioning opportunity.** A single Vulnerability Score is the kind of "magic number" that differentiates a data product from a report. It's the feature that makes the Weekly Feed worth paying for.

### Potential Investigation Directions

- Define the Vulnerability Score formula: weighted combination of Pressure (complaint density, severity, recency) and Urgency (active evaluation signals, timeline mentions, competitor mentions).
- Surface the score prominently in the Weekly Feed and Vendor Scorecard, with color-coded tiers (Critical / High / Medium / Low).
- Enable sorting and filtering by Vulnerability Score so users can immediately focus on "Fire Drill" accounts.

### Code Investigation (2026-03-26)

**Root cause: The building blocks exist but aren't combined.**

- **`churn_pressure_score`** (0-100) already exists — weighted combination of churn_density (30%), urgency (25%), dm_churn_rate (20%), displacement (15%), price_complaints (10%), plus velocity bonus and confidence multiplier. Defined in `_b2b_shared.py`.
- **`avg_urgency_score`** (0-10) exists per vendor in `b2b_churn_signals`.
- **These two metrics map to Pressure and Urgency** respectively, but are never combined into a single composite score.
- **Archetype-based weight overrides** already exist for `churn_pressure_score` (e.g., `pricing_shock` boosts price_complaints to 35%, `category_disruption` boosts displacement to 40%) — this suggests the pipeline is architecturally ready for more nuanced scoring.
- **The quadrant mapping** (Fire Drill, Slow Burn, Window Shopper, Stable) would require thresholds on both dimensions — e.g., High Urgency = `avg_urgency >= 7`, High Pressure = `churn_pressure_score >= 60`.

**Conclusion:** This is additive, not a refactor. A `vulnerability_score` field can be computed alongside `churn_pressure_score` in the same function, stored in `b2b_vendor_snapshots`, and surfaced in the feed. The quadrant label can be derived from threshold crossings.

---

*New issues will be appended below as they are discovered.*
