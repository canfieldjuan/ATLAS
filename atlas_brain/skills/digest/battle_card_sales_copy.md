---
name: digest/battle_card_sales_copy
domain: digest
description: Generate sales-oriented objection handlers and recommended plays from battle card data
tags: [digest, b2b, churn, battle_card, sales]
version: 1
---

# Battle Card Sales Copy

You are generating sales messaging for a B2B competitive battle card.

## Purpose

The battle card's data sections (weaknesses, pain quotes, competitor differentiators, objection data) are already built deterministically. Your job is to generate two sales-ready sections:

1. `objection_handlers`: Data-backed responses to common buyer objections about this vendor.
2. `recommended_plays`: Tactical sales plays for targeting this vendor's dissatisfied customers.

## Input

You will receive a JSON object with:

- `vendor`: the target vendor name
- `vendor_weaknesses`: top weaknesses with evidence counts
- `customer_pain_quotes`: verbatim customer quotes with urgency scores
- `competitor_differentiators`: top competitors this vendor loses to
- `objection_data`: raw metrics including:
  - `price_complaint_rate` (0-1 fraction)
  - `dm_churn_rate` (0-1 fraction)
  - `sentiment_direction` (improving/stable/declining/insufficient_data)
  - `top_feature_gaps` (list of {feature, mentions})
  - `total_reviews`, `churn_signal_density`, `avg_urgency`
  - `budget_context` (seat counts, contract signals)

## Output Schema

```json
{
  "objection_handlers": [
    {
      "objection": "What the prospect might say in favor of the vendor",
      "response": "Data-backed counter-argument (1-2 sentences)",
      "data_backing": "Specific metric or evidence cited"
    }
  ],
  "recommended_plays": [
    {
      "play": "Tactical action for the sales team (1 sentence)",
      "target_segment": "Who to target (role, company size, industry)",
      "key_message": "Core value proposition to lead with (1 sentence)"
    }
  ]
}
```

## Rules

- Respond with ONLY a valid JSON object. No markdown fences.
- Generate 2-4 objection handlers based on available data. Skip objection types that lack supporting metrics.
- Generate 2-3 recommended plays.
- Every number in `response` or `data_backing` MUST come from the input data. Do not fabricate statistics.
- Reference actual metrics: use `price_complaint_rate`, `dm_churn_rate`, `churn_signal_density`, feature gap mention counts, etc.
- Keep `response` to 1-2 sentences max. Be direct and specific.
- `objection` should be phrased as something a prospect would actually say (e.g., "They're cheaper than your solution").
- `recommended_plays` should be actionable by a sales rep today.
- `target_segment` should reference data from the input (company sizes, industries, buyer roles from quotes).
- Do not reference vendor weaknesses that have zero evidence.
- If `price_complaint_rate` >= 0.15, include a pricing objection handler.
- If `top_feature_gaps` has 2+ entries, include a features objection handler.
- If `dm_churn_rate` >= 0.25, reference decision-maker dissatisfaction in a play.

## Output

Return only the JSON object.
