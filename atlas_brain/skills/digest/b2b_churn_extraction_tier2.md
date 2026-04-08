---
name: digest/b2b_churn_extraction_tier2
domain: digest
description: Tier 2 extraction -- bounded classification + verbatim phrase extraction via local vLLM
tags: [digest, b2b, churn, saas, autonomous, hybrid, tier2]
version: 3
---

# B2B Churn Signal Extraction -- Tier 2 (Classify + Extract)

Given one software review plus Tier 1 extraction, classify bounded enum fields and extract verbatim phrases. A separate system handles scoring, inference, and conclusions. Do only the fields below.

## Core Rules

- CLASSIFY fields: use only the listed enum values.
- EXTRACT fields: copy verbatim text from the review. Do not summarize or paraphrase.
- INDICATOR fields: set booleans only when the language pattern is explicit.
- When uncertain, use `unknown`, `null`, `[]`, or `false` as appropriate. Never guess.
- Prefer the most commercially specific verbatim evidence: money, dates, deadlines, renewal language, named orgs, switch/evaluation language, bundle pressure, workflow substitution, or productivity claims.

## Input

The input is one JSON object from the pipeline. Fields may be absent when blank. Typical fields include:

- review context: `vendor_name`, `product_name`, `product_category`, `source_name`, `source_weight`, `source_type`, `content_type`, `rating`, `rating_max`
- review content: `summary`, `review_text`, `pros`, `cons`
- reviewer metadata: `reviewer_title`, `reviewer_company`, `company_size_raw`, `reviewer_industry`
- Tier 1 context: `tier1_specific_complaints`, `tier1_quotable_phrases`

`content_type` is one of `review`, `community_discussion`, `comment`, `insider_account`.

Use `tier1_specific_complaints` and `tier1_quotable_phrases` as the primary evidence for pain classification. Use the full review text for all other extraction and classification.

## Required Output Keys

Return one valid JSON object with exactly these top-level keys:

```json
{
  "competitors_mentioned": [],
  "pain_categories": [],
  "sentiment_trajectory": {"tenure": null},
  "buyer_authority": {
    "role_type": "unknown",
    "executive_sponsor_mentioned": false,
    "buying_stage": "unknown"
  },
  "timeline": {
    "contract_end": null,
    "evaluation_deadline": null,
    "decision_timeline": "unknown"
  },
  "contract_context": {
    "contract_value_signal": "unknown",
    "usage_duration": null
  },
  "positive_aspects": [],
  "feature_gaps": [],
  "recommendation_language": [],
  "pricing_phrases": [],
  "event_mentions": [],
  "urgency_indicators": {
    "explicit_cancel_language": false,
    "active_migration_language": false,
    "active_evaluation_language": false,
    "completed_switch_language": false,
    "comparison_shopping_language": false,
    "named_alternative_with_reason": false,
    "frustration_without_alternative": false,
    "dollar_amount_mentioned": false,
    "timeline_mentioned": false,
    "decision_maker_language": false
  },
  "insider_signals": null
}
```

## Field Rules

### competitors_mentioned -- CLASSIFY per competitor

- Preserve the exact competitor names found in the review or Tier 1 output.
- `evidence_type`: `explicit_switch`, `active_evaluation`, `implied_preference`, `reverse_flow`, `neutral_mention`
- `displacement_confidence`: `high`, `medium`, `low`, `none`
- `reason_category`: `pricing`, `features`, `reliability`, `ux`, `support`, `integration`, or `null`
- Only use `explicit_switch` or `active_evaluation` when direction is explicit.
- Only populate `reason_category` when the review explicitly states the reason.

### pain_categories -- CLASSIFY from complaints

Use `tier1_specific_complaints` and `tier1_quotable_phrases` as primary evidence.

- `category`: `pricing`, `features`, `reliability`, `support`, `integration`, `performance`, `security`, `ux`, `onboarding`, `technical_debt`, `contract_lock_in`, `data_migration`, `api_limitations`, `outcome_gap`, `admin_burden`, `ai_hallucination`, `integration_debt`, `other`
- `severity`: `primary`, `secondary`, `minor`
- First entry must be `primary`.
- Use `other` only when no named category fits.
- If there are no real complaints, return `[]`.
- `outcome_gap`: promised outcome or ROI does not materialize.
- `admin_burden`: ongoing operational cost or heavy maintenance burden.
- `ai_hallucination`: AI output is inaccurate or fabricated.
- `integration_debt`: integrations exist but are brittle or expensive to maintain.
- Switching or evaluation-linked pain wins ties.
- Pricing wins only when price pressure is explicit, such as dollar amounts or "too expensive".
- Do not collapse workflow substitution, bundled-suite pressure, or internal-tooling replacement into generic UX when the review is explicit.

### sentiment_trajectory -- EXTRACT only tenure

- `tenure`: verbatim usage duration, or `null`

### buyer_authority -- CLASSIFY

- `role_type`: `economic_buyer`, `champion`, `evaluator`, `end_user`, `unknown`
- `executive_sponsor_mentioned`: true only when an executive is explicitly referenced
- `buying_stage`: `active_purchase`, `evaluation`, `renewal_decision`, `post_purchase`, `unknown`
- Infer `role_type` from title when present; otherwise use explicit review language

### timeline -- EXTRACT dates, CLASSIFY timeline

- `contract_end`: verbatim, or `null`
- `evaluation_deadline`: verbatim, or `null`
- `decision_timeline`: `immediate`, `within_quarter`, `within_year`, `unknown`

### contract_context -- CLASSIFY + EXTRACT

- `contract_value_signal`: `enterprise_high`, `enterprise_mid`, `mid_market`, `smb`, `unknown`
- `usage_duration`: verbatim, or `null`

### positive_aspects

- Extract verbatim positive aspects. Return `[]` if none.

### feature_gaps

- Extract verbatim missing-feature requests. Return `[]` if none.

### recommendation_language

- Extract verbatim recommendation or anti-recommendation phrases. Return `[]` if none.

### pricing_phrases

- Extract verbatim pricing, billing, cost, or value complaints. Keep specific amounts when present. Return `[]` if none.

### event_mentions

- Each item is `{"event": <verbatim>, "timeframe": <verbatim or null>}`.
- Extract only concrete events that affected the experience. Return `[]` if none.

### urgency_indicators -- DETECT

Set booleans only when explicit:

- `explicit_cancel_language`: canceling, not renewing, terminating, ending contract
- `active_migration_language`: migration or switch is in progress
- `active_evaluation_language`: evaluating alternatives, POC, shortlist, comparisons
- `completed_switch_language`: completed past switch
- `comparison_shopping_language`: asking which option to choose or requesting alternatives
- `named_alternative_with_reason`: competitor named and explicit reason provided
- `frustration_without_alternative`: dissatisfaction with no named alternative
- `dollar_amount_mentioned`: specific dollar amount appears
- `timeline_mentioned`: contract date, renewal date, or decision deadline appears
- `decision_maker_language`: explicit approval or signoff language appears

### insider_signals -- CLASSIFY + EXTRACT (only for insider_account)

Only for `content_type = "insider_account"`. All other types must return `null`.

- `role_at_company`: verbatim, or `null`
- `departure_type`: `voluntary`, `involuntary`, `still_employed`, `unknown`
- `departures_mentioned`: boolean
- `layoff_fear`: boolean
- `morale`: `high`, `medium`, `low`, `unknown`
- `bureaucracy_level`: `high`, `medium`, `low`, `unknown`
- `leadership_quality`: `poor`, `mixed`, `good`, `unknown`
- `innovation_climate`: `stagnant`, `declining`, `healthy`, `unknown`
- `culture_indicators`: verbatim list
- `morale_language`: verbatim list

## Output

Respond with only one valid JSON object. No markdown fencing. No commentary. No reasoning.
