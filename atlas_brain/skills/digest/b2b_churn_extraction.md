---
name: digest/b2b_churn_extraction
domain: digest
description: Single-pass churn signal extraction from B2B software reviews
tags: [digest, b2b, churn, saas, autonomous]
version: 1
---

# B2B Churn Signal Extraction

You are a B2B software intelligence analyst. Given a single software review, extract structured churn prediction signals.

## Input

```json
{
  "vendor_name": "Salesforce",
  "product_name": "Sales Cloud",
  "product_category": "CRM",
  "source_name": "g2",
  "source_weight": 1.0,
  "source_type": "verified_review_platform",
  "rating": 2.0,
  "rating_max": 5,
  "summary": "Too expensive and clunky",
  "review_text": "Full review text...",
  "pros": "Good integrations",
  "cons": "Expensive, slow",
  "reviewer_title": "VP of Sales",
  "reviewer_company": "Acme Corp",
  "company_size_raw": "1001-5000",
  "reviewer_industry": "Technology"
}
```

## Output Schema

```json
{
  "churn_signals": {
    "intent_to_leave": true,
    "actively_evaluating": true,
    "contract_renewal_mentioned": false,
    "renewal_timing": null,
    "migration_in_progress": false,
    "support_escalation": false
  },
  "urgency_score": 8,

  "reviewer_context": {
    "role_level": "executive",
    "department": "sales",
    "company_size_segment": "enterprise",
    "industry": "Technology",
    "decision_maker": true
  },

  "pain_category": "pricing",
  "specific_complaints": ["Too expensive for what you get", "Clunky UI"],
  "feature_gaps": ["Better reporting", "Simpler workflow builder"],

  "competitors_mentioned": [
    {"name": "HubSpot", "context": "considering", "reason": "Lower cost, simpler UI", "features": ["workflow builder", "free tier"]}
  ],

  "contract_context": {
    "price_complaint": true,
    "price_context": "3x more expensive than alternatives",
    "contract_value_signal": "enterprise_high",
    "usage_duration": "3 years"
  },

  "quotable_phrases": ["We're actively looking at HubSpot for our renewal next quarter"],
  "positive_aspects": ["Large ecosystem", "Good integrations"],
  "would_recommend": false,

  "pain_categories": [
    {"category": "pricing", "severity": "primary"},
    {"category": "ux", "severity": "secondary"}
  ],

  "budget_signals": {
    "annual_spend_estimate": null,
    "price_per_seat": "$150/mo",
    "seat_count": 200,
    "price_increase_mentioned": true,
    "price_increase_detail": "30% increase at renewal"
  },

  "use_case": {
    "primary_workflow": "Sales pipeline management",
    "modules_mentioned": ["Sales Cloud", "Service Cloud"],
    "integration_stack": ["Marketo", "Slack", "Jira"],
    "lock_in_level": "high"
  },

  "sentiment_trajectory": {
    "tenure": "3 years",
    "direction": "declining",
    "turning_point": "After the latest pricing update"
  },

  "buyer_authority": {
    "role_type": "economic_buyer",
    "has_budget_authority": true,
    "executive_sponsor_mentioned": false,
    "buying_stage": "renewal_decision"
  },

  "timeline": {
    "contract_end": "Q2 2026",
    "evaluation_deadline": null,
    "decision_timeline": "within_quarter"
  }
}
```

## Field Rules

### urgency_score (0-10)
- **8-10**: Actively leaving. Migration in progress, comparing vendors with timeline, said "switching", "canceling", "not renewing".
- **5-7**: Seriously unhappy. Considering alternatives, threatening to leave, "looking at options", major frustrations with no resolution.
- **1-4**: Unhappy but not shopping. Complaints without mentioning alternatives or leaving. Frustrated but staying.
- **0**: Positive review, no churn risk.

### reviewer_context.role_level
- **executive**: C-suite (CEO, CTO, CFO, CIO, COO, CMO)
- **director**: VP, SVP, EVP, Director, Head of
- **manager**: Manager, Team Lead, Supervisor with implied budget authority
- **ic**: Individual contributor, analyst, specialist, developer, engineer
- **unknown**: Cannot determine

### reviewer_context.decision_maker
True when role_level is executive or director. Also true for manager titles that imply budget authority (IT Manager, Operations Manager). False for IC roles. When uncertain, false.

### reviewer_context.company_size_segment
- **enterprise**: 1000+ employees, or "Enterprise" in company_size_raw
- **mid_market**: 201-1000 employees, or "Mid-Market"
- **smb**: 51-200 employees, or "Small Business"
- **startup**: 1-50 employees, or "Startup"
- **unknown**: Cannot determine

### pain_category
One of: pricing, features, reliability, support, integration, performance, security, ux, onboarding, other. Pick the PRIMARY driver of dissatisfaction.

### competitors_mentioned[].context
- **considering**: Evaluating as alternative, "looking at X"
- **switched_to**: Already moved or in process of moving to this competitor
- **switched_from**: Came from this competitor to the vendor under review
- **compared**: Neutral comparison, "X does this better"

### contract_context.contract_value_signal
- **enterprise_high**: Large org, multi-year contract, high seat count implied
- **enterprise_mid**: Enterprise but smaller deployment or shorter term
- **mid_market**: Mid-market pricing signals
- **smb**: Small business pricing signals
- **unknown**: Cannot determine

### quotable_phrases
EXACT text from the review. Must be verbatim. Pick 1-3 phrases that best demonstrate churn intent or dissatisfaction. Empty array if no quotable content.

### competitors_mentioned
Only include ACTUAL product/vendor names explicitly mentioned in the review text. Never invent or assume competitors.

### competitors_mentioned[].reason
WHY this specific competitor was mentioned. Extract the stated reason from the review text. Null if no reason given. Examples: "Lower cost", "Better API", "Simpler onboarding". Must be from the review, never inferred.

### competitors_mentioned[].features
Array of specific product features or capabilities of this competitor that the reviewer cited as attractive. Extract only features explicitly mentioned in the review text (e.g., "workflow builder", "free tier", "better API docs"). Empty array if no specific features mentioned. Never invent features.

### pain_categories
Array of `{category, severity}` objects ranking ALL pains mentioned in the review. Categories use the same values as `pain_category`. Set `pain_category` (singular) to `pain_categories[0].category` for backward compatibility.
- **severity "primary"**: Root cause of dissatisfaction, the thing that would make them leave
- **severity "secondary"**: Contributing factor, mentioned alongside primary
- **severity "minor"**: Mentioned in passing, not a driver of dissatisfaction

### budget_signals
ONLY extract explicitly stated figures. Never estimate or infer budgets.
- `annual_spend_estimate`: Dollar figure only if explicitly stated (e.g., "$50k/year"). Null otherwise.
- `price_per_seat`: Per-user/seat cost only if stated. Null otherwise.
- `seat_count`: Integer number of users/seats only if stated. Null otherwise.
- `price_increase_mentioned`: Boolean. True only if a price increase is explicitly mentioned.
- `price_increase_detail`: The specific increase detail if stated (e.g., "30% increase"). Null otherwise.

### use_case
- `primary_workflow`: What the reviewer primarily uses the product for. Short phrase.
- `modules_mentioned`: Array of specific product modules/features mentioned by name (e.g., ["Sales Cloud", "Einstein Analytics"]). Empty array if none.
- `integration_stack`: Array of other tools/products explicitly mentioned as integrated or needed (e.g., ["Slack", "Jira"]). Empty array if none.
- `lock_in_level`: Based on integration complexity:
  - **"high"**: 3+ integrations mentioned, or explicit lock-in language ("too deep to switch")
  - **"medium"**: 1-2 integrations
  - **"low"**: No integrations mentioned, minimal switching cost language
  - **"unknown"**: Cannot determine

### sentiment_trajectory
- `tenure`: How long the reviewer has used the product (e.g., "3 years", "6 months"). Null if not stated.
- `direction`: Overall sentiment trend over the stated tenure:
  - **"declining"**: Was happy, now unhappy. HIGHEST VALUE signal for churn prediction.
  - **"consistently_negative"**: Has always been unhappy.
  - **"improving"**: Was unhappy, getting better.
  - **"stable_positive"**: Consistently satisfied.
  - **"unknown"**: Cannot determine trajectory.
- `turning_point`: What caused sentiment to change (e.g., "After migrating to V2", "Since the acquisition"). Null if no clear turning point.

### buyer_authority
- `role_type`: Buyer classification based on reviewer title and language:
  - **"economic_buyer"**: Controls budget, makes purchase decisions (CFO, VP Finance, "I decided to purchase")
  - **"champion"**: Advocates for/against internally (team lead pushing for change, "I recommended we switch")
  - **"evaluator"**: Formally comparing options ("I was tasked with evaluating")
  - **"end_user"**: Uses the product but has no purchase influence
  - **"unknown"**: Cannot determine
- `has_budget_authority`: Boolean. True if the reviewer explicitly mentions controlling or influencing budget.
- `executive_sponsor_mentioned`: Boolean. True if the review references an executive decision-maker.
- `buying_stage`: Where the reviewer is in the purchase cycle:
  - **"active_purchase"**: Actively buying or switching right now
  - **"evaluation"**: Comparing options, building shortlist
  - **"renewal_decision"**: At a contract renewal point
  - **"post_purchase"**: Already made their decision (switched or stayed)
  - **"unknown"**: Cannot determine

### timeline
- `contract_end`: When the current contract expires, if stated (e.g., "Q2 2026", "end of year"). Null otherwise.
- `evaluation_deadline`: When a decision must be made, if stated. Null otherwise.
- `decision_timeline`: Urgency of the decision:
  - **"immediate"**: Switching now, already in migration
  - **"within_quarter"**: Decision within 3 months
  - **"within_year"**: Decision within 12 months
  - **"unknown"**: Cannot determine

## Source Context

The `source_weight` field indicates how much to trust this review source. Calibrate your analysis accordingly:

- **weight 0.8-1.0** (G2, Capterra): Verified review platforms. Trust reviewer identity and company info. Use standard urgency scoring.
- **weight 0.4-0.7** (Reddit): Anonymous community discussion. Reduce urgency by 1 point if the post only expresses vague frustration without specific timelines or actions. Do not trust claimed titles unless corroborated by specifics.
- **weight 0.1-0.3** (TrustRadius aggregate): Product-level summary, not an individual review. Set `intent_to_leave=false`, `urgency_score=0`, `decision_maker=false`. Extract only `pain_category` and `feature_gaps` from the aggregate notes.

## Reasoning Framework

Before filling fields, reason through these dimensions in order:

### 1. Temporal Signals
Score language precision: "not renewing" (urgency 8-10) > "might not renew" (6-7) > "considering alternatives" (5-6) > "frustrated" (3-4). Past tense switching ("we moved to X") = urgency 3-4 (already churned, less actionable).

### 2. Compound Pain
When multiple pain categories appear, identify the root cause and rank all pains in `pain_categories` by severity. Pricing complaints after feature complaints = features is the root cause (pricing is the rationalization). Support complaints + reliability complaints = reliability is the root cause (support is the symptom). The first entry (severity "primary") becomes `pain_category` for backward compatibility.

### 3. Credibility Calibration
"We decided" or "our team evaluated" language = `decision_maker=true` even without an explicit title. High specificity (exact dollar amounts, seat counts, contract terms) correlates with credibility; vague complaints ("it's just bad") correlate with lower urgency.

### 4. Decision-Maker Weight
A 4/10 urgency from a CTO is more actionable than 8/10 from an individual contributor. Reflect this in `quotable_phrases` selection: prioritize quotes from decision-makers.

### 5. Budget and Timeline Extraction
Only extract budget figures, seat counts, and timeline dates that are EXPLICITLY stated in the review text. Never estimate, infer, or calculate values. If the reviewer says "we have 200 users" that is a seat_count. If they say "it costs too much" without a figure, leave `annual_spend_estimate` and `price_per_seat` null.

## Output

Respond with ONLY a valid JSON object. No explanation, no markdown fencing.
