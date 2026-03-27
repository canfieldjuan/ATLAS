---
name: digest/b2b_churn_extraction
domain: digest
description: Single-pass churn signal extraction from B2B software reviews
tags: [digest, b2b, churn, saas, autonomous]
version: 3
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
  "content_type": "review",
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

`content_type` values: `review`, `community_discussion`, `comment`, `insider_account`.

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
    "decision_maker": true,
    "company_name": null
  },

  "pain_category": "pricing",
  "specific_complaints": ["Too expensive for what you get", "Clunky UI"],
  "feature_gaps": ["Better reporting", "Simpler workflow builder"],

  "competitors_mentioned": [
    {
      "name": "HubSpot",
      "evidence_type": "active_evaluation",
      "displacement_confidence": "medium",
      "reason_category": "pricing",
      "reason_detail": "3x more expensive than HubSpot",
      "features": ["workflow builder", "free tier"],
      "context": "considering",
      "reason": "pricing: 3x more expensive than HubSpot"
    }
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
  },

  "content_classification": "review",

  "recommendation_language": ["I would not recommend this to anyone"],
  "pricing_phrases": ["3x more expensive than HubSpot", "30% increase at renewal"],
  "event_mentions": [
    {"event": "latest pricing update", "timeframe": "Q1 2026"}
  ],
  "urgency_indicators": {
    "explicit_cancel_language": false,
    "active_migration_language": false,
    "active_evaluation_language": true,
    "completed_switch_language": false,
    "comparison_shopping_language": false,
    "named_alternative_with_reason": true,
    "frustration_without_alternative": false,
    "dollar_amount_mentioned": false,
    "timeline_mentioned": true,
    "decision_maker_language": false
  },

  "insider_signals": null
}
```

**Pipeline-computed fields:** `urgency_score`, `would_recommend`, `pain_category`, `sentiment_trajectory.direction`, `sentiment_trajectory.turning_point`, `buyer_authority.has_budget_authority`, `contract_context.price_complaint`, and `contract_context.price_context` are computed deterministically by the pipeline after extraction. You may still return them as hints but the pipeline will override with its own calculation.

For `content_type = "insider_account"`, populate `insider_signals`:

```json
{
  "insider_signals": {
    "role_at_company": "Senior Engineer",
    "departure_type": "voluntary",
    "departures_mentioned": true,
    "layoff_fear": false,
    "morale": "low",
    "bureaucracy_level": "high",
    "leadership_quality": "poor",
    "innovation_climate": "stagnant",
    "culture_indicators": ["micromanagement", "no autonomy", "reorg every 6 months"],
    "morale_language": ["team morale is at an all-time low", "people are leaving in droves"]
  }
}
```

For all other `content_type` values, set `"insider_signals": null`.

## Field Rules

### urgency_score (0-10)

IMPORTANT: This score measures INTELLIGENCE VALUE for B2B churn prediction, not just the reviewer's explicit intent to cancel. A review that reveals competitive displacement, active evaluation, or systemic vendor weakness is high-urgency intelligence even if the reviewer does not say "I am leaving."

- **9-10**: Confirmed active churn. Explicit "we are switching/canceling/not renewing" with timeline or migration details. Example: "We're migrating to HubSpot next quarter, already in POC."
- **7-8**: Strong churn signal. Actively evaluating replacements, completed a recent switch (past tense), or migration report with competitive comparison. Examples: "We switched from Jira to Height last year", "I just did a Tableau to Power BI migration for a client", "We're evaluating three CRM alternatives for renewal."
- **5-6**: Moderate churn signal. Comparing vendors (even casually), expressing frustration with named alternatives in mind, asking the community for recommendations, or describing feature gaps that competitors solve. Examples: "Should I use Shopify or WooCommerce?", "X does this much better than Y", "Seriously considering alternatives after the latest price hike."
- **3-4**: Weak but present signal. Complaints without naming alternatives, general dissatisfaction, poor experience report. The reviewer is unhappy but not shopping. Examples: "Support has been terrible for months", "The UI got worse after the update."
- **1-2**: Minimal signal. Minor gripes in an otherwise neutral/positive review. Example: "Great tool but wish it had better reporting."
- **0**: ONLY for purely positive reviews with zero complaints. Example: "Love this product, works perfectly for our team."

Past tense switching ("we moved to X", "we ditched Y") scores 7-8, NOT 0. A completed switch is a confirmed displacement data point -- high-value intelligence for competitive flow analysis.

### reviewer_context.role_level
- **executive**: C-suite (CEO, CTO, CFO, CIO, COO, CMO)
- **director**: VP, SVP, EVP, Director, Head of
- **manager**: Manager, Team Lead, Supervisor with implied budget authority
- **ic**: Individual contributor, analyst, specialist, developer, engineer
- **unknown**: Cannot determine

Infer from `reviewer_title` when present. If `reviewer_title` is blank, use first-person self-identification or explicit operating language in the review text (for example "as an engineer", "I'm the VP of IT"). Do not require a title field when the review text itself clearly establishes the role.

### reviewer_context.decision_maker
True when role_level is executive or director. Also true for manager titles that imply budget authority (IT Manager, Operations Manager). True when the reviewer explicitly says they approved, signed off on, or made the purchase/renewal decision. False for IC roles. When uncertain, false.

### reviewer_context.company_size_segment
- **enterprise**: 1000+ employees, or "Enterprise" in company_size_raw
- **mid_market**: 201-1000 employees, or "Mid-Market"
- **smb**: 51-200 employees, or "Small Business"
- **startup**: 1-50 employees, or "Startup"
- **unknown**: Cannot determine

### reviewer_context.company_name
Extract the reviewer's company name ONLY when explicitly stated in the review text, pros, cons, summary, or reviewer_company field. Examples: "We use this at Acme Corp", "As a Google employee", "Our team at [Company] switched to..."
- Only extract proper company names explicitly mentioned in the text
- Never infer from industry, company size, role, or contextual clues
- Set to null when no company name is explicitly stated
- Do not extract generic references like "my company" or "our organization"
- If reviewer_company is already provided in the input and is not empty, use that value

### pain_category
One of: pricing, features, reliability, support, integration, performance, security, ux, onboarding, technical_debt, contract_lock_in, data_migration, api_limitations, other. Pick the PRIMARY driver of dissatisfaction -- the root cause that, if fixed, would retain the customer. When multiple pains co-occur, apply these tiebreakers: (1) the pain explicitly linked to switching/evaluation language wins; (2) pricing beats other categories only when dollar amounts or "too expensive" are stated; (3) "other" is a last resort -- prefer a specific category even if the fit is imperfect. For comparison/evaluation posts where no pain is expressed, use "features" (the reviewer is comparing capabilities).

### competitors_mentioned
Only include ACTUAL product/vendor names explicitly mentioned in the review text. Never invent or assume competitors.

### competitors_mentioned[].evidence_type
Classify the EVIDENCE for competitive displacement in THIS review. This is the most important field for data quality — be conservative:
- **"explicit_switch"**: Reviewer explicitly states they switched, migrated, or moved to this competitor. Evidence: "We moved to X", "We switched to X", "We migrated to X last quarter", "We replaced [vendor] with X".
- **"active_evaluation"**: Reviewer is actively evaluating this competitor as a replacement. Evidence: "We're evaluating X", "X is on our shortlist", "We're doing a POC with X", "Looking at X for renewal".
- **"implied_preference"**: Reviewer implies this competitor is better but does NOT state switching or active evaluation. Evidence: "X does this much better", "I wish we had X's features", "X is cheaper".
- **"reverse_flow"**: Reviewer came FROM this competitor TO the vendor under review. Evidence: "We switched from X to [vendor]", "After leaving X...", "We moved off X".
- **"neutral_mention"**: Competitor named but no directional signal. Evidence: "X and [vendor] are both in this space", "I also use X", "Competitors like X exist".

CRITICAL: Only classify as "explicit_switch" or "active_evaluation" when the review text contains clear directional language. "X is better" is NOT a switch — it is "implied_preference". "I've heard good things about X" is "neutral_mention", not "active_evaluation".

### competitors_mentioned[].displacement_confidence
How confident are you that this review represents REAL competitive displacement toward this competitor?
- **"high"**: explicit_switch with specific details (timeline, team size, migration status, named product)
- **"medium"**: active_evaluation with named product and specifics, OR explicit_switch without details
- **"low"**: implied_preference, or vague evaluation language without specifics
- **"none"**: reverse_flow or neutral_mention (not displacement)

### competitors_mentioned[].reason_category
Why this competitor is being considered or chosen. MUST be one of:
- **"pricing"**: Cost, pricing model, value for money, price increase
- **"features"**: Missing capabilities, feature gaps, roadmap disappointment
- **"reliability"**: Uptime, stability, performance, bugs, outages
- **"ux"**: User experience, ease of use, learning curve, admin burden
- **"support"**: Customer support quality, response time, account management
- **"integration"**: API quality, integrations, ecosystem compatibility
- **"technical_debt"**: Code quality, legacy stack, maintenance burden
- **"contract_lock_in"**: Renewal terms, exit costs, rigid contracts
- **"data_migration"**: Ease/difficulty of moving data, portability
- **"api_limitations"**: Rate limits, missing endpoints, poor documentation
- **null**: No reason stated in the review text. Do NOT infer a reason.

### competitors_mentioned[].reason_detail
Verbatim phrase from the review explaining WHY (e.g., "30% cheaper", "better API docs", "simpler onboarding"). Null if no specific reason stated. Must be extracted from the text, never inferred.

### competitors_mentioned[].context
Backward-compatible alias. Set automatically from evidence_type:
- explicit_switch → "switched_to"
- active_evaluation → "considering"
- implied_preference → "compared"
- reverse_flow → "switched_from"
- neutral_mention → "compared"

### competitors_mentioned[].reason
Backward-compatible alias. Set automatically: if reason_category and reason_detail both exist, format as "{reason_category}: {reason_detail}". If only reason_category, use that. If only reason_detail, use that. Null if neither.

### contract_context.contract_value_signal
- **enterprise_high**: Large org, multi-year contract, high seat count implied
- **enterprise_mid**: Enterprise but smaller deployment or shorter term
- **mid_market**: Mid-market pricing signals
- **smb**: Small business pricing signals
- **unknown**: Cannot determine

### quotable_phrases
EXACT text from the review. Must be verbatim. Pick 1-3 phrases that best demonstrate churn intent or dissatisfaction. Empty array if no quotable content.

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

When `reviewer_title` is blank, infer `role_type` from the review text itself. Strong purchase/renewal approval language indicates `economic_buyer`; formal comparison/POC/shortlist language indicates `evaluator`; clear day-to-day product use language without purchase authority indicates `end_user`.
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

### content_classification
Set to the `content_type` value from the input (pass-through for downstream filtering):
- `"review"`: Structured review from a verified platform
- `"community_discussion"`: Reddit/HN post expressing user experience
- `"comment"`: Reply within a thread (lower signal weight)
- `"insider_account"`: Employee or ex-employee perspective

### insider_signals
Only populated when `content_type = "insider_account"`. All other types must set this to `null`.

- `role_at_company` (EXTRACT): The role the author held at the vendor. E.g., "Senior Engineer", "Product Manager". Null if not stated.
- `departure_type` (CLASSIFY): `"voluntary"`, `"involuntary"`, `"still_employed"`, or `"unknown"`.
- `departures_mentioned` (boolean): True if the text mentions people leaving, high turnover, or difficulty retaining talent.
- `layoff_fear` (boolean): True if the text mentions fear of layoffs, RIFs, or job security concerns.
- `morale` (CLASSIFY): Overall morale signal: `"high"`, `"medium"`, `"low"`, or `"unknown"`.
- `bureaucracy_level` (CLASSIFY): How bureaucratic/slow the company is: `"high"`, `"medium"`, `"low"`, `"unknown"`.
- `leadership_quality` (CLASSIFY): Perception of leadership effectiveness: `"poor"`, `"mixed"`, `"good"`, `"unknown"`.
- `innovation_climate` (CLASSIFY): Product/engineering culture: `"stagnant"`, `"declining"`, `"healthy"`, `"unknown"`.
- `culture_indicators` (EXTRACT): Array of specific culture descriptors from the text (e.g., `["micromanagement", "no autonomy"]`). Empty array if none.
- `morale_language` (EXTRACT): Verbatim phrases about morale, culture, or team health. Empty array if none.

**Why insider signals matter for churn prediction**: A vendor with deteriorating engineering culture, talent exodus, and poor leadership quality will ship a worse product over time — directly increasing customer churn risk. Treat insider accounts as forward-looking churn indicators.

## Source Context

The `source_weight` field indicates how much to trust this review source. Calibrate your analysis accordingly:

- **weight 0.8-1.0** (G2, Capterra): Verified review platforms. Trust reviewer identity and company info. Use standard urgency scoring.
- **weight 0.4-0.7** (Reddit): Anonymous community discussion. Do not trust claimed titles unless corroborated by specifics. Apply standard urgency scoring -- do NOT discount Reddit posts simply for being anonymous. Reddit posts asking "which tool should I use?" or comparing alternatives are active evaluation signals (urgency 5-6+). Reddit posts describing completed switches are displacement data (urgency 7-8).
- **weight 0.1-0.3** (TrustRadius aggregate): Product-level summary, not an individual review. Set `intent_to_leave=false`, `urgency_score=0`, `decision_maker=false`. Extract only `pain_category` and `feature_gaps` from the aggregate notes.
- **content_type = "insider_account"**: Employee/ex-employee perspective. Prioritize `insider_signals` extraction. The standard churn fields (`urgency_score`, `churn_signals`) still apply -- an insider post predicting product stagnation IS a churn signal. Map insider signals to urgency: mass departures or talent drain = urgency 7-8 (customers will see quality decline); leadership dysfunction or repeated reorgs = urgency 5-6 (product direction uncertainty); minor culture gripes = urgency 3-4. Set `intent_to_leave=true` when the insider describes conditions that will drive customer churn (product quality collapse, support degradation).
- **content_type = "comment"**: Short reply in a thread. Lower signal confidence. Reduce urgency by 1 point. Focus on extracting `quotable_phrases` and `competitors_mentioned`.

### recommendation_language (NEW)
Extract ALL verbatim phrases where the reviewer expresses a recommendation or anti-recommendation. Include the full phrase. Examples: "I would highly recommend this", "stay away from this product", "not worth the price". Empty array if none.

### pricing_phrases (NEW)
Extract ALL verbatim phrases where the reviewer complains about pricing, cost, or value. Include specific amounts when stated. Examples: "3x more expensive than HubSpot", "$150/user/month is outrageous". Empty array if none.

### event_mentions (NEW)
Extract mentions of specific events that affected the reviewer's experience. Each entry: `{"event": "verbatim description", "timeframe": "when it happened or null"}`. Examples: `{"event": "latest pricing update", "timeframe": "Q1 2026"}`. Empty array if none.

### urgency_indicators (NEW)
Set each boolean based on whether the language pattern is explicitly present in the review:
- `explicit_cancel_language`: "canceling", "not renewing", "terminating contract"
- `active_migration_language`: describes ongoing migration or switch in progress
- `active_evaluation_language`: describes evaluating alternatives, running POC, comparing
- `completed_switch_language`: past-tense switch: "we moved to X", "we switched last year"
- `comparison_shopping_language`: "which should I choose", "X vs Y", "looking for alternatives"
- `named_alternative_with_reason`: competitor named AND specific reason given
- `frustration_without_alternative`: complaints but no competitor named
- `dollar_amount_mentioned`: specific dollar amount appears
- `timeline_mentioned`: contract end date, renewal date, or deadline mentioned
- `decision_maker_language`: "I decided", "our team approved", "we signed off on"

## Reasoning Framework

Before filling fields, reason through these dimensions in order:

### 1. Temporal Signals
Score language precision: "not renewing" (9-10) > "evaluating replacements" (7-8) > "we switched to X last year" (7-8) > "might not renew" (6-7) > "considering alternatives" (5-6) > "frustrated" (3-4). Past tense switching ("we moved to X") = urgency 7-8 (confirmed displacement -- high intelligence value). Pre-purchase evaluation ("which should I choose, X or Y?") = urgency 5-6 (reveals competitive positioning).

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
