---
name: digest/b2b_churn_extraction_tier1
domain: digest
description: Single-pass deterministic extraction via local vLLM
tags: [digest, b2b, churn, saas, autonomous, tier1]
version: 3
---

# B2B Churn Signal Extraction -- Single Pass

Perform deterministic extraction from one software review.

Extract only factual, pattern-matchable fields:

- explicit booleans
- verbatim phrases
- named entities
- numbers
- bounded extract-only fields

Do not score, rank, infer, summarize, or recommend. Do not classify competitive evidence, urgency, pain priority, contract value, or buying stage. Those are handled later in Python.

## Input

The input is one JSON object from the pipeline. Fields may be absent when blank. Typical fields include:

- `vendor_name`, `product_name`, `product_category`
- `source_name`, `source_weight`, `source_type`, `content_type`
- `rating`, `rating_max`
- `summary`, `review_text`, `pros`, `cons`
- `reviewer_title`, `reviewer_company`, `company_size_raw`, `reviewer_industry`

## Required Output Keys

Return one valid JSON object with exactly these top-level keys:

```json
{
  "churn_signals": {
    "intent_to_leave": false,
    "actively_evaluating": false,
    "contract_renewal_mentioned": false,
    "renewal_timing": null,
    "migration_in_progress": false,
    "support_escalation": false
  },
  "reviewer_context": {
    "role_level": "unknown",
    "department": null,
    "company_size_segment": "unknown",
    "industry": null,
    "decision_maker": false,
    "company_name": null
  },
  "budget_signals": {
    "annual_spend_estimate": null,
    "price_per_seat": null,
    "seat_count": null,
    "price_increase_mentioned": false,
    "price_increase_detail": null
  },
  "use_case": {
    "primary_workflow": null,
    "modules_mentioned": [],
    "integration_stack": [],
    "lock_in_level": "unknown"
  },
  "content_classification": "review",
  "competitors_mentioned": [],
  "specific_complaints": [],
  "quotable_phrases": [],
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
  "sentiment_trajectory": {"tenure": null},
  "buyer_authority": {"executive_sponsor_mentioned": false},
  "timeline": {"contract_end": null, "evaluation_deadline": null},
  "contract_context": {"usage_duration": null},
  "insider_signals": null
}
```

## Field Rules

- `churn_signals`: detect only explicit language. Never infer.
- `reviewer_context.role_level`: `executive`, `director`, `manager`, `ic`, `unknown`.
- `reviewer_context.department`: short department label or `null`.
- `reviewer_context.company_size_segment`: `enterprise`, `mid_market`, `smb`, `startup`, `unknown`.
- `reviewer_context.decision_maker`: true only when approval or decision ownership is explicit.
- `budget_signals`: extract only explicit figures. Never estimate.
- `use_case.lock_in_level`: `high` for 3+ integrations or explicit lock-in language, `medium` for 1-2 integrations, `low` for none, `unknown` if unclear.
- `competitors_mentioned`: extract only explicitly named competitors. If the text says `switched to`, `moved to`, `replaced with`, `evaluating`, `looking at`, or `considering`, extract that competitor name.
- `specific_complaints`, `quotable_phrases`, `positive_aspects`, `feature_gaps`, `recommendation_language`, `pricing_phrases`: verbatim phrases only. No paraphrase.
- Prefer commercially relevant evidence over catchy wording.
- Billing, cancellation, and contract issues belong in `specific_complaints` when explicit.
- Price and billing language belongs in `pricing_phrases` when explicit.
- Preserve explicit language about docs, async workflows, bundled suites, internal tooling, or productivity shifts instead of flattening it into generic UX.
- `event_mentions[*].event` and `event_mentions[*].timeframe`: verbatim. Use `null` timeframe if absent.
- `urgency_indicators`: set booleans only when the pattern is explicit.
- `sentiment_trajectory.tenure`, `timeline.contract_end`, `timeline.evaluation_deadline`, `contract_context.usage_duration`: verbatim only.
- `buyer_authority.executive_sponsor_mentioned`: true only when an executive sponsor is explicitly mentioned.
- `insider_signals`: only populate for `content_type = insider_account`; otherwise `null`.

## Output

Respond with only one valid JSON object. No markdown fencing. No commentary.
