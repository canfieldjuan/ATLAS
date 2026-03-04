---
name: b2b/product_profile_synthesis
domain: b2b
description: Synthesize a product profile knowledge card from aggregated B2B review data
tags: [b2b, churn, product, profile, autonomous]
version: 1
---

# B2B Product Profile Synthesis

You are analyzing a B2B software product based on aggregated review data. Your goal is to produce a concise, evidence-based profile summary and validate pain-addressed scores.

## Input

You will receive a JSON object with these fields:

- **vendor_name**: The product/vendor name
- **product_category**: Product category (may be null)
- **total_reviews**: Number of reviews analyzed
- **avg_rating**: Average review rating (1-5 scale)
- **strengths**: Array of `{"area": str, "score": float, "evidence_count": int}` — features/areas where users are satisfied (higher score = more positive)
- **weaknesses**: Array of `{"area": str, "score": float, "evidence_count": int}` — features/areas where users complain (lower score = worse)
- **use_cases**: Array of `{"use_case": str, "count": int}` — primary workflows
- **integrations**: Array of integration names
- **competitive_data**: Object with `commonly_compared_to` and `commonly_switched_from` arrays
- **pain_categories**: Array of all known pain category names to score against

## Tasks

1. **Profile Summary** (2-3 sentences): Describe what this product does well and where it falls short. Be specific and evidence-based. Reference actual strength/weakness areas and scores. Do NOT use marketing language.

2. **Pain Addressed Scores**: For each pain category provided, score 0.0-1.0 how well this product ADDRESSES that pain:
   - 1.0 = this vendor completely solves this pain (it appears as a top strength)
   - 0.5 = neutral (no strong signal either way)
   - 0.0 = this vendor makes it worse (it appears as a top weakness)
   - Base scores on the strengths/weaknesses data. If a strength directly maps to a pain category, score high. If a weakness maps to it, score low.

## Output

Respond with a single JSON object (no markdown fences, no extra text):

```json
{
  "summary": "2-3 sentence profile summary...",
  "pain_addressed": {
    "integration_complexity": 0.85,
    "poor_support": 0.3,
    "pricing_concerns": 0.6
  }
}
```

Only include pain categories from the input list. Scores must be between 0.0 and 1.0.
