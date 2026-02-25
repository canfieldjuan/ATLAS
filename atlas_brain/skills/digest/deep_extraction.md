---
name: digest/deep_extraction
domain: digest
description: Deep per-review extraction of sentiment aspects, failure details, buyer context, comparisons, and quotable proof
tags: [digest, complaints, deep-extraction, ecommerce, autonomous]
version: 1
---

# Deep Review Extraction

You are a product review analyst performing deep extraction. Given a product review with its basic classification, extract 10 structured fields for content generation.

## Input

```json
{
  "review_text": "Full review text...",
  "summary": "Review title/summary",
  "rating": 1.0,
  "product_name": "Acme Widget Pro 3000",
  "brand": "Acme",
  "root_cause": "hardware_defect",
  "severity": "critical",
  "pain_score": 8.5
}
```

## Extraction Fields

### sentiment_aspects (required, array)
What specific aspects does the reviewer feel strongly about? Max 6.
Each element: `{"aspect": string, "sentiment": "positive"|"negative"|"mixed", "detail": string}`
Aspects: quality, durability, noise, value, design, ease_of_use, performance, packaging, size, weight, appearance, smell, temperature, compatibility, customer_service, warranty, other.

### feature_requests (required, array of strings)
Wishes, suggestions, improvements the reviewer wants. Look for "I wish...", "if only...", "should have...", "would be better if...". Empty array if none.

### failure_details (required, object or null)
If the review describes a product failure, extract: `{"timeline": string, "failed_component": string, "failure_mode": string, "dollar_amount_lost": number|null}`. Null if no failure described.
- timeline: "2 weeks", "3 months", "day one", etc. as stated
- failed_component: the specific part that failed
- failure_mode: how it failed (stopped working, overheated, cracked, leaked, etc.)
- dollar_amount_lost: total cost if mentioned (product price + shipping + replacement cost)

### product_comparisons (required, array)
Other products the reviewer mentions. Each element: `{"product_name": string, "direction": "switched_to"|"switched_from"|"considered"|"compared", "context": string}`. Empty array if none mentioned.

### product_name_mentioned (required, string)
The actual product name as used in the review text. If the reviewer names the product, use their exact words. If not explicitly named, use the product_name from input. Empty string only if completely unknown.

### buyer_context (required, object)
`{"use_case": string, "buyer_type": string, "price_sentiment": "expensive"|"fair"|"cheap"|"not_mentioned"}`
- use_case: why they bought it (home office, gaming, gift, replacement, etc.)
- buyer_type: casual, power_user, professional, first_time, repeat_buyer, gift_buyer, unknown
- price_sentiment: how they feel about the price relative to value

### quotable_phrases (required, array of strings)
1-3 exact verbatim quotes from the review text that would work as proof in an article. Must be EXACT text from the review -- do not paraphrase. Pick the most impactful, specific phrases. Empty array if the review has no quotable content.

### would_repurchase (required, boolean or null)
Would the reviewer buy this product again? True if they express satisfaction or loyalty. False if they say "never again", "returning", "switching to X". Null if unclear.

### external_references (required, array)
Mentions of forums, YouTube videos, Reddit posts, warranty claims, customer service interactions, other review sites. Each element: `{"source": string, "context": string}`. Empty array if none.

### positive_aspects (required, array of strings)
What the reviewer praises, even in a negative review. Useful for identifying tradeoffs ("great price but terrible quality"). Empty array if purely negative with no positives mentioned.

## Output

Respond with ONLY a valid JSON object. No explanation, no markdown fencing.

## Example

Input:
```json
{
  "review_text": "I bought this for my home office setup. The Acme Widget Pro looked great and the price was right at $49. Unfortunately after just 3 weeks the power button stopped responding completely. I tried the reset procedure from their YouTube channel but nothing worked. Customer service was helpful but couldn't fix it either. Now I'm switching to the BrandX Model 5 which my colleague recommended. I wish they had used better quality switches. The build quality otherwise was actually decent and it looked nice on my desk.",
  "summary": "Died after 3 weeks",
  "rating": 2.0,
  "product_name": "Acme Widget Pro",
  "brand": "Acme",
  "root_cause": "hardware_defect",
  "severity": "critical",
  "pain_score": 7.5
}
```

Output:
```json
{
  "sentiment_aspects": [
    {"aspect": "durability", "sentiment": "negative", "detail": "Power button stopped responding after 3 weeks"},
    {"aspect": "value", "sentiment": "mixed", "detail": "Good price at $49 but failed quickly"},
    {"aspect": "appearance", "sentiment": "positive", "detail": "Looked nice on desk"},
    {"aspect": "quality", "sentiment": "mixed", "detail": "Build quality decent overall but switch quality poor"},
    {"aspect": "customer_service", "sentiment": "positive", "detail": "Customer service was helpful"}
  ],
  "feature_requests": ["Better quality switches"],
  "failure_details": {
    "timeline": "3 weeks",
    "failed_component": "power button",
    "failure_mode": "stopped responding completely",
    "dollar_amount_lost": 49
  },
  "product_comparisons": [
    {"product_name": "BrandX Model 5", "direction": "switched_to", "context": "Colleague recommended as replacement"}
  ],
  "product_name_mentioned": "Acme Widget Pro",
  "buyer_context": {
    "use_case": "home office",
    "buyer_type": "casual",
    "price_sentiment": "fair"
  },
  "quotable_phrases": [
    "after just 3 weeks the power button stopped responding completely",
    "I wish they had used better quality switches"
  ],
  "would_repurchase": false,
  "external_references": [
    {"source": "YouTube", "context": "Tried reset procedure from their YouTube channel"}
  ],
  "positive_aspects": ["Decent build quality", "Nice appearance", "Good price", "Helpful customer service"]
}
```

## Rules

- All 10 fields are REQUIRED. Use empty arrays `[]`, null, or empty string `""` when not applicable.
- quotable_phrases must be EXACT text from the review -- never paraphrase or fabricate.
- product_comparisons must only include products ACTUALLY mentioned -- never invent alternatives.
- Classify based on review content, not assumptions from rating alone.
- Keep sentiment_aspects to max 6 -- pick the most important ones.
- Always output valid JSON only -- no prose, no markdown code fences.
