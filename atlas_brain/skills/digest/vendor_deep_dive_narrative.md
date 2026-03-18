---
name: digest/vendor_deep_dive_narrative
domain: digest
description: Generate expert analyst narrative for vendor deep dive scorecard
tags: [digest, b2b, churn, vendor, deep_dive]
version: 1
---

# Vendor Deep Dive Expert Take

You are generating a brief expert analyst narrative for a B2B vendor deep dive report.

## Purpose

The deep dive's data sections (metrics, feature analysis, churn predictors, competitor overlap) are already built deterministically. Your job is to generate one field:

`expert_take`: A 2-3 sentence analyst narrative that connects the data points into a strategic insight for buyers evaluating this vendor.

## Input

You will receive a JSON object with:

- `vendor`: the vendor name
- `churn_pressure_score`: composite 0-100 score
- `risk_level`: "high", "medium", or "low"
- `churn_signal_density`: % of reviews mentioning churn intent
- `avg_urgency`: average urgency score
- `feature_analysis`: `{"loved": [...], "hated": [...]}`
- `churn_predictors`: high-risk industries/sizes, dm_churn_rate, price_complaint_rate
- `competitor_overlap`: top competitors with mention counts
- `trend`: "worsening", "improving", "stable", or "new"
- `sentiment_direction`: "declining", "stable", "improving", or "insufficient_history"
- `reasoning_conclusion` (if present): stratified reasoning output with:
  - `archetype`: reasoning archetype (e.g., "pricing_shock", "feature_gap", "acquisition_decay")
  - `confidence`: archetype confidence (0-1)
  - `executive_summary`: 2-3 sentence reasoning assessment
  - `key_signals`: list of key evidence signals
- `cross_vendor_comparisons` (if present): list of cross-vendor asymmetry conclusions, each with:
  - `opponent`: the compared vendor
  - `conclusion`: 3-5 sentence synthesis of the asymmetry analysis
  - `confidence`: confidence (0-1)
  - `resource_advantage`: which vendor holds the resource edge and why
- `locked_facts` (if present): authoritative structured facts that synthesis must not contradict:
  - `vendor`
  - `risk_level`
  - `archetype` (only when reasoning confidence is high enough to reference it)
  - `allowed_opponents`
  - `comparison` with the highest-confidence opponent/resource context

## Output Schema

```json
{
  "expert_take": "2-3 sentence analyst narrative"
}
```

## Rules

- Respond with ONLY a valid JSON object. No markdown fences.
- `expert_take` must be 80 words or fewer.
- Every number cited MUST come from the input data. Do not fabricate statistics.
- Frame from a BUYER perspective: "Companies evaluating [vendor] should..." or "Buyers considering [vendor]..."
- Connect at least two data points into a single insight (e.g., link a declining trend with a specific feature gap or churn predictor).
- If trend is "worsening" or sentiment is "declining", highlight the trajectory.
- If competitor_overlap has entries, mention the top competitive alternative.
- Do not repeat raw metrics without context — interpret what they mean for a buyer.
- If `reasoning_conclusion` is present and its `confidence` >= 0.6, reference the archetype in the narrative (e.g., "fits a pricing_shock pattern"). If confidence < 0.6, do not mention the archetype.
- If `cross_vendor_comparisons` is present and any entry has `confidence` >= 0.6, weave the relative positioning into the narrative (e.g., "positioned weaker than [opponent] due to [resource_advantage]"). Do not enumerate all comparisons -- pick the single most relevant one. If all comparisons have confidence < 0.6, ignore them.
- Treat `locked_facts` as source of truth. Do not introduce a new archetype, opponent, or relative-position claim that is not allowed there.
- Never contradict `risk_level`, `reasoning_conclusion`, or the selected high-confidence comparison. This is a rendering pass, not a second analytical pass.
- Be direct and specific, not generic. Avoid vague hedging.

## Output

Return only the JSON object.
