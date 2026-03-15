---
name: digest/b2b_exploratory_overview
domain: digest
description: Focused exploratory follow-up for B2B churn intelligence timeline risk and narrative context
tags: [digest, b2b, churn, intelligence, exploratory]
version: 1
---

# B2B Exploratory Overview

You are generating a small supplemental JSON object for the exploratory B2B churn report.

## Purpose

The core report tables are already built deterministically. Your job is only to provide:

1. `exploratory_summary`: a short analyst narrative that highlights the most interesting non-headline pattern in the payload.
2. `timeline_hot_list`: upcoming renewal or evaluation checkpoints that are still future-dated relative to `date`.

## Input

You will receive the same trimmed intelligence payload used by the B2B churn pipeline, including:

- `date`
- `data_context`
- `high_intent_companies`
- `timeline_signals`
- `competitive_displacement`
- `quotable_evidence`
- supporting vendor and category context
- `known_companies`
- `stratified_intelligence` (if present): per-vendor reasoning conclusions with:
  - `vendor`: vendor name
  - `archetype`: reasoning archetype (e.g., "pricing_shock", "feature_gap", "acquisition_decay")
  - `confidence`: archetype confidence (0-1)
  - `risk_level`: "low", "medium", "high", or "critical"
  - `executive_summary`: 2-3 sentence reasoning assessment
  - `key_signals`: list of key evidence signals

## Output Schema

```json
{
  "exploratory_summary": "string",
  "timeline_hot_list": [
    {
      "company": "exact value from known_companies",
      "vendor": "string",
      "contract_end": "YYYY-MM-DD or null",
      "urgency": 0,
      "action": "buyer-facing recommendation",
      "buyer_role": "string",
      "budget_authority": false
    }
  ]
}
```

## Rules

- Respond with ONLY a valid JSON object.
- Do not include markdown fences.
- `exploratory_summary` must be 120 words or fewer.
- If `stratified_intelligence` is present, synthesize archetype clusters into the narrative (e.g., "3 vendors show pricing_shock patterns"). Highlight any vendors with critical risk_level.
- Use only evidence present in the payload.
- `timeline_hot_list` must only include future dates relative to `date`.
- Omit expired, stale, or ambiguous timelines.
- `company` values must exactly match `known_companies`.
- If there are no valid upcoming timeline entries, return an empty array.
- `action` must be buyer-facing.
- Do not repeat the full weekly feed, vendor scorecards, displacement map, or category overview.

## Output

Return only the JSON object.