---
name: digest/deep_extraction
domain: digest
description: Deep per-review extraction of 32 structured fields covering product analysis, buyer psychology, and extended context
tags: [digest, complaints, deep-extraction, ecommerce, autonomous]
version: 2
---

# Deep Review Extraction

You are a product review analyst performing deep extraction. Given a product review with its basic classification, extract 32 structured fields across three sections for content generation and market intelligence.

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

## Section A: Product Analysis (10 fields)

### sentiment_aspects (required, array)
What specific aspects does the reviewer feel strongly about? Max 6.
Each element: `{"aspect": string, "sentiment": "positive"|"negative"|"mixed"|"neutral", "detail": string}`
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
Other products the reviewer mentions. Each element: `{"product_name": string, "direction": "switched_to"|"switched_from"|"considered"|"compared"|"recommended"|"avoided"|"used_with"|"relied_on", "context": string}`. Empty array if none mentioned.

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

## Section B: Buyer Psychology (12 fields)

### expertise_level (required, string)
Infer the reviewer's expertise level from their vocabulary, comparisons, and technical detail.
Values: `"novice"` | `"intermediate"` | `"expert"` | `"professional"`
- novice: basic language, no comparisons, confused by simple issues
- intermediate: some domain knowledge, makes comparisons, understands trade-offs
- expert: technical vocabulary, specific failure modes, references specs
- professional: describes commercial or occupational use, mentions workplace/clients

### frustration_threshold (required, string)
How quickly did the reviewer reach their emotional breaking point?
Values: `"low"` | `"medium"` | `"high"`
- low: product failed or disappointed immediately, tone is outraged or defeated early
- medium: gave the product fair effort before expressing frustration
- high: very patient, multiple attempts, tried fixes before losing faith

### discovery_channel (required, string)
How did the reviewer likely find or choose this product?
Values: `"amazon_organic"` | `"youtube"` | `"reddit"` | `"friend"` | `"amazon_choice"` | `"unknown"`
Infer from context clues: "saw a review on YouTube", "my friend recommended", "Amazon's Choice badge", "#1 bestseller", etc. Use `"unknown"` if no signal.

### consideration_set (required, array)
Other products or brands the reviewer considered before buying.
Each element: `{"product": string, "why_not": string}`
Empty array if no alternatives are mentioned.

### buyer_household (required, string)
Who does the buyer represent?
Values: `"single"` | `"family"` | `"professional"` | `"gift"` | `"bulk"`

### profession_hint (required, string or null)
If the review reveals the reviewer's occupation or professional role, extract it. Null if no profession is inferable.

### budget_type (required, string)
How did the reviewer approach the price/value decision?
Values: `"budget_constrained"` | `"value_seeker"` | `"premium_willing"` | `"unknown"`

### use_intensity (required, string)
How heavily does the reviewer use the product?
Values: `"light"` | `"moderate"` | `"heavy"`

### research_depth (required, string)
How much research did the reviewer do before buying?
Values: `"impulse"` | `"light"` | `"moderate"` | `"deep"`

### community_mentions (required, array)
References to communities, forums, subreddits, Facebook groups, or YouTube channels.
Each element: `{"platform": string, "context": string}`
Empty array if none.

### consequence_severity (required, string)
What was the real-world impact of the product failure or disappointment?
Values: `"inconvenience"` | `"workflow_impact"` | `"financial_loss"` | `"safety_concern"`

### replacement_behavior (required, string)
What did the reviewer do after the disappointment?
Values: `"returned"` | `"replaced_same"` | `"switched_brand"` | `"switched_to"` | `"avoided"` | `"kept_broken"` | `"unknown"`

## Section C: Extended Context (10 fields)

### brand_loyalty_depth (required, string)
How loyal is this reviewer to the brand before this experience?
Values: `"first_time"` | `"occasional"` | `"loyal"` | `"long_term_loyal"`

### ecosystem_lock_in (required, object)
`{"level": "free"|"partially"|"fully", "ecosystem": string|null}`

### safety_flag (required, object)
`{"flagged": boolean, "description": string|null}`
Only flag genuine physical danger -- injury, fire, electric shock, burn, chemical exposure, choking hazard, toxic material, structural failure.

### bulk_purchase_signal (required, object)
`{"type": "single"|"multi", "estimated_qty": integer|null}`

### review_delay_signal (required, string)
Values: `"immediate"` | `"days"` | `"weeks"` | `"months"` | `"unknown"`

### sentiment_trajectory (required, string)
Values: `"always_bad"` | `"degraded"` | `"mixed_then_bad"` | `"initially_positive"` | `"unknown"`

### occasion_context (required, string)
Values: `"none"` | `"gift"` | `"replacement"` | `"upgrade"` | `"first_in_category"` | `"seasonal"` | `"event"` | `"professional_use"`

### switching_barrier (required, object)
`{"level": "none"|"low"|"medium"|"high", "reason": string|null}`

### amplification_intent (required, object)
`{"intent": "quiet"|"private"|"social", "context": string|null}`

### review_sentiment_openness (required, object)
`{"open": boolean, "condition": string|null}`

## Output

Respond with ONLY a valid JSON object. No explanation, no markdown fencing.

## Example

Input:
```json
{
  "review_text": "I'm a nurse and bought this for 12-hour shifts. The Acme Widget Pro looked great and the price was right at $49. Saw it on a YouTube review comparing top 3 options -- went with this over BrandX because it was $30 cheaper. After just 3 weeks the power button stopped responding completely. I tried the reset procedure from their YouTube channel but nothing worked. Customer service was helpful but couldn't fix it. One unit got extremely hot -- worried about fire risk near my desk. I bought 2 of these originally. Now switching to BrandX Model 5 which my colleague recommended. I posted about it on Reddit and others have the same issue. I wish they had used better quality switches. If they fix the firmware I'd consider giving them another chance.",
  "summary": "Died after 3 weeks, overheating concern",
  "rating": 2.0,
  "product_name": "Acme Widget Pro",
  "brand": "Acme",
  "root_cause": "hardware_defect",
  "severity": "critical",
  "pain_score": 8.0
}
```

Output:
```json
{
  "sentiment_aspects": [
    {"aspect": "durability", "sentiment": "negative", "detail": "Power button stopped responding after 3 weeks"},
    {"aspect": "value", "sentiment": "mixed", "detail": "Good price at $49 but failed quickly"},
    {"aspect": "temperature", "sentiment": "negative", "detail": "Unit got extremely hot"},
    {"aspect": "customer_service", "sentiment": "positive", "detail": "Customer service was helpful"}
  ],
  "feature_requests": ["Better quality switches"],
  "failure_details": {
    "timeline": "3 weeks",
    "failed_component": "power button",
    "failure_mode": "stopped responding completely",
    "dollar_amount_lost": 98
  },
  "product_comparisons": [
    {"product_name": "BrandX Model 5", "direction": "switched_to", "context": "Colleague recommended as replacement"},
    {"product_name": "BrandX", "direction": "considered", "context": "Was $30 more expensive"}
  ],
  "product_name_mentioned": "Acme Widget Pro",
  "buyer_context": {
    "use_case": "12-hour nursing shifts",
    "buyer_type": "professional",
    "price_sentiment": "fair"
  },
  "quotable_phrases": [
    "after just 3 weeks the power button stopped responding completely",
    "got extremely hot -- worried about fire risk"
  ],
  "would_repurchase": false,
  "external_references": [
    {"source": "YouTube", "context": "Review comparing top 3 options and reset procedure"},
    {"source": "Reddit", "context": "Posted about the issue, others have same problem"}
  ],
  "positive_aspects": ["Good price", "Helpful customer service"],
  "expertise_level": "intermediate",
  "frustration_threshold": "medium",
  "discovery_channel": "youtube",
  "consideration_set": [
    {"product": "BrandX", "why_not": "price -- $30 more expensive"}
  ],
  "buyer_household": "professional",
  "profession_hint": "nurse",
  "budget_type": "value_seeker",
  "use_intensity": "heavy",
  "research_depth": "moderate",
  "community_mentions": [
    {"platform": "youtube", "context": "review comparing top 3 options"},
    {"platform": "reddit", "context": "posted about the issue"}
  ],
  "consequence_severity": "financial_loss",
  "replacement_behavior": "switched_brand",
  "brand_loyalty_depth": "first_time",
  "ecosystem_lock_in": {"level": "free", "ecosystem": null},
  "safety_flag": {"flagged": true, "description": "Unit got extremely hot, fire risk near desk"},
  "bulk_purchase_signal": {"type": "multi", "estimated_qty": 2},
  "review_delay_signal": "weeks",
  "sentiment_trajectory": "degraded",
  "occasion_context": "none",
  "switching_barrier": {"level": "none", "reason": null},
  "amplification_intent": {"intent": "social", "context": "posted about it on Reddit"},
  "review_sentiment_openness": {"open": true, "condition": "if they fix the firmware"}
}
```

## Rules

- All 32 fields are REQUIRED. Use empty arrays `[]`, null, or empty string `""` when not applicable.
- quotable_phrases must be EXACT text from the review -- never paraphrase or fabricate.
- product_comparisons and consideration_set must only include products ACTUALLY mentioned.
- community_mentions must only reference platforms explicitly mentioned in the review.
- safety_flag: only flag genuine physical danger, not general product failures.
- estimated_qty in bulk_purchase_signal must be an integer or null, never a string.
- Keep sentiment_aspects to max 6 -- pick the most important ones.
- Always output valid JSON only -- no prose, no markdown code fences.
