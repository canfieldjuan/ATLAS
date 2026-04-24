---
name: digest/b2b_churn_extraction_tier1
domain: digest
description: Single-pass deterministic extraction via local vLLM
tags: [digest, b2b, churn, saas, autonomous, tier1]
version: 4
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
  "enrichment_schema_version": 4,
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
  "insider_signals": null,
  "phrase_metadata": []
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

## Phrase Metadata (`phrase_metadata`)

Parallel to the six verbatim phrase arrays (`specific_complaints`, `pricing_phrases`, `feature_gaps`, `quotable_phrases`, `recommendation_language`, `positive_aspects`). For every phrase you placed in one of those arrays, add exactly one corresponding entry in `phrase_metadata`.

Each entry must have:

- `field`: the array name (exact string, one of the six listed above)
- `index`: the phrase's 0-based position in that array
- `text`: MUST equal the string at `[field][index]` exactly -- do not paraphrase, do not summarize
- `subject`: strict enum -- `subject_vendor`, `alternative`, `self`, `third_party`, `unclear`
- `polarity`: strict enum -- `negative`, `positive`, `mixed`, `unclear`
- `role`: strict enum -- `primary_driver`, `supporting_context`, `passing_mention`, `unclear`
- `verbatim`: boolean -- `true` if the phrase text appears word-for-word in the review; `false` otherwise
- `category_hint`: optional string (pain category like `pricing`, `support`, `features`) for `specific_complaints` and `quotable_phrases` only; omit or `null` otherwise

Rules:

- Use `unclear` whenever you cannot tell. Do not guess.
- Legacy arrays stay verbatim string arrays. Do not put objects into `specific_complaints` etc.
- Never invent phrases. Every metadata row corresponds to a real extraction.
- Never skip phrases. Every non-empty legacy-array element gets exactly one metadata row.

### Subject -- critical disambiguation

The target of this review is `subject_vendor`. Everything else is something else.

- "I pay $X for [vendor]" -- the phrase is about [vendor]; `subject=subject_vendor`.
- "I pay $X" (with no vendor reference nearby) -- about the reviewer's own spending; `subject=self`.
- "I built my own" / "we use in-house tools" -- `subject=self`.
- "switched to [competitor]" / "considering [competitor]" / "[competitor] is cheaper" -- the phrase is about the competitor; `subject=alternative`.
- "Salesforce and HubSpot users both..." (general industry observation) -- `subject=third_party`.

### Polarity -- what the phrase expresses

- Reviewer voicing dissatisfaction or a complaint -- `negative`.
- Reviewer voicing praise or satisfaction -- `positive`.
- Both in one phrase -- `mixed`.
- Neutral fact statement with no valence -- `unclear`.

### Role -- how central the phrase is

- Cited as a reason for sentiment, churn, or decision -- `primary_driver`.
- Provides background relevant to the main point -- `supporting_context`.
- Drive-by mention, not part of the core argument -- `passing_mention`.

### Examples

1. Review says "I used to pay $1,500 a month for Monday, but I built my own setup for $30."

   `pricing_phrases`: `["pay $1,500 a month for Monday", "built my own setup for $30"]`
   `phrase_metadata`:
   - `{field: "pricing_phrases", index: 0, text: "pay $1,500 a month for Monday", subject: "subject_vendor", polarity: "negative", role: "supporting_context", verbatim: true}`
   - `{field: "pricing_phrases", index: 1, text: "built my own setup for $30", subject: "self", polarity: "positive", role: "supporting_context", verbatim: true}`

   Critical: the $1,500 phrase is about Monday's pricing; the $30 phrase is about the reviewer's own alternative. Do NOT tag both as `subject_vendor`.

2. Review says "Good value at $500 per seat. Salesforce charges twice that for less."

   `pricing_phrases`: `["Good value at $500 per seat", "Salesforce charges twice that for less"]`
   `phrase_metadata`:
   - `{field: "pricing_phrases", index: 0, text: "Good value at $500 per seat", subject: "subject_vendor", polarity: "positive", role: "primary_driver", verbatim: true}`
   - `{field: "pricing_phrases", index: 1, text: "Salesforce charges twice that for less", subject: "alternative", polarity: "negative", role: "supporting_context", verbatim: true}`

3. Review says "Their pricing killed our budget. We're moving to HubSpot next quarter."

   `pricing_phrases`: `["Their pricing killed our budget"]`
   `recommendation_language`: `["We're moving to HubSpot next quarter"]`
   `phrase_metadata`:
   - `{field: "pricing_phrases", index: 0, text: "Their pricing killed our budget", subject: "subject_vendor", polarity: "negative", role: "primary_driver", verbatim: true, category_hint: "pricing"}`
   - `{field: "recommendation_language", index: 0, text: "We're moving to HubSpot next quarter", subject: "alternative", polarity: "negative", role: "primary_driver", verbatim: true}`

4. Review says "Support was helpful."

   `positive_aspects`: `["Support was helpful"]`
   `phrase_metadata`:
   - `{field: "positive_aspects", index: 0, text: "Support was helpful", subject: "subject_vendor", polarity: "positive", role: "passing_mention", verbatim: true}`

## Output

Respond with only one valid JSON object. No markdown fencing. No commentary.
