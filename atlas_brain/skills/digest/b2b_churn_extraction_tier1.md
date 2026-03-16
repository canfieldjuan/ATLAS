---
name: digest/b2b_churn_extraction_tier1
domain: digest
description: Tier 1 deterministic extraction -- 26 fields via local vLLM (NER, booleans, enums, verbatim text)
tags: [digest, b2b, churn, saas, autonomous, hybrid, tier1]
version: 1
---

# B2B Churn Signal Extraction -- Tier 1 (Deterministic)

You are a B2B software intelligence analyst performing DETERMINISTIC extraction from a software review. Extract only factual, pattern-matchable fields. Do NOT interpret, reason about, or score anything.

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

## Output Schema (26 fields only)

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

  "reviewer_context": {
    "role_level": "executive",
    "department": "sales",
    "company_size_segment": "enterprise",
    "industry": "Technology",
    "decision_maker": true,
    "company_name": "Acme Corp"
  },

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

  "content_classification": "review",

  "competitors_mentioned": [
    {
      "name": "HubSpot",
      "features": ["workflow builder", "free tier"],
      "reason_detail": "3x more expensive than HubSpot"
    }
  ],

  "specific_complaints": ["Too expensive for what you get", "Clunky UI"],
  "quotable_phrases": ["We're actively looking at HubSpot for our renewal next quarter"]
}
```

## Field Rules

### churn_signals (6 fields -- boolean/string detection)
- `intent_to_leave`: True if reviewer uses language like "switching", "leaving", "canceling", "not renewing", "moving to". False otherwise.
- `actively_evaluating`: True if reviewer mentions evaluating, comparing, shortlisting, or doing POC with alternatives. False otherwise.
- `contract_renewal_mentioned`: True if "renewal", "renew", "contract end", or similar appears. False otherwise.
- `renewal_timing`: Extract the exact timing phrase if stated (e.g., "Q2 2026", "next quarter"). Null if not stated.
- `migration_in_progress`: True if reviewer describes an ongoing migration or switch. False otherwise.
- `support_escalation`: True if reviewer mentions escalating support tickets, contacting management, or filing formal complaints. False otherwise.

### reviewer_context (6 fields -- NER + enum mapping)
- `role_level`: Map reviewer_title to one of: `executive` (C-suite), `director` (VP/Director/Head of), `manager`, `ic` (individual contributor), `unknown`.
- `department`: Extract department from title (e.g., "VP of Sales" -> "sales", "Engineering Manager" -> "engineering"). Null if unclear.
- `company_size_segment`: Map company_size_raw: 1000+ = `enterprise`, 201-1000 = `mid_market`, 51-200 = `smb`, 1-50 = `startup`, else `unknown`.
- `industry`: Pass through from reviewer_industry. Null if empty.
- `decision_maker`: True for executive/director roles. True for manager titles implying budget authority. False otherwise.
- `company_name`: Use reviewer_company if provided. Otherwise extract from text ONLY if explicitly stated. Null if not found.

### budget_signals (5 fields -- number/string extraction)
- Extract ONLY explicitly stated figures. Never estimate.
- `annual_spend_estimate`: Dollar figure only if explicitly stated. Null otherwise.
- `price_per_seat`: Per-user cost only if stated. Null otherwise.
- `seat_count`: Integer only if stated. Null otherwise.
- `price_increase_mentioned`: True only if a price increase is explicitly mentioned.
- `price_increase_detail`: The specific increase detail if stated. Null otherwise.

### use_case (4 fields -- keyword extraction)
- `primary_workflow`: Short phrase describing what the reviewer uses the product for.
- `modules_mentioned`: Array of specific product modules/features mentioned by name. Empty array if none.
- `integration_stack`: Array of other tools explicitly mentioned as integrated. Empty array if none.
- `lock_in_level`: `high` (3+ integrations or explicit lock-in language), `medium` (1-2 integrations), `low` (none), `unknown`.

### content_classification (pass-through)
Set to the `content_type` value from the input.

### competitors_mentioned (Tier 1 subset: name + features + reason_detail only)
- `name`: Exact product/vendor name as mentioned in text. Only real names, never invented.
- `features`: Array of specific features of this competitor cited as attractive. Empty array if none.
- `reason_detail`: Verbatim phrase from the review explaining why this competitor was mentioned. Null if no specific reason stated.

### specific_complaints (verbatim extraction)
Array of specific complaint phrases extracted from the review text. Empty array if none.

### quotable_phrases (verbatim extraction)
Array of 1-3 exact verbatim phrases from the review that demonstrate dissatisfaction or churn intent. Empty array if none.

## Output

Respond with ONLY a valid JSON object. No explanation, no markdown fencing. Temperature 0 -- always produce the same output for the same input.
