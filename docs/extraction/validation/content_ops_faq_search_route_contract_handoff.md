# Content Ops FAQ Search Route Contract Handoff

This document is the consumer-facing contract for the FAQ deflection search
route used by the searchable demo and frontend mappers. The machine checker for
this shape is `scripts/check_content_ops_faq_search_route_contract.py`.

## Route

Search:

```http
GET /api/v1/content-ops/faq-deflection-search?q=<query>&limit=5
Authorization: Bearer <token>
```

Detail:

```http
GET /api/v1/content-ops/faq-deflection-search/{faq_id}
Authorization: Bearer <token>
```

Optional search filters:

- `corpus_id`: restricts matches to one seeded corpus.
- `status`: defaults to approved results unless explicitly set blank.
- `limit`: positive integer, capped by the route config.

## Search Envelope

Successful search responses use this envelope:

```json
{"query": "<query>", "results": [], "count": 0}
```

No-match is not an error. A no-match response is exactly:

```json
{"query": "<query>", "results": [], "count": 0}
```

When matches exist, `results[0]` is the lean card/list shape:

| Field | Type | Meaning |
|---|---|---|
| `faq_id` | string | ID to hydrate through the detail route. |
| `question` | string | Customer-facing FAQ question. |
| `answer_summary` | string | Short display summary only. Use detail for steps/body. |
| `topic` | string | Deterministic FAQ topic label. |
| `source_ids` | string list | Source row IDs behind the item. |
| `ticket_count` | integer | Ticket/source volume represented by the FAQ item. |
| `score` | integer | Deterministic text relevance score for this query. It is not a percentage and is not the FAQ opportunity score. |

`count is the number of returned rows` in this response, not a total-result
count across all pages.

## Score Semantics

`score` is query relevance from the search projection. It should drive match
ordering or a generic relevance indicator only.

Do not render `score` as:

- a `0-100%` match bar,
- customer impact,
- cost savings,
- opportunity score,
- confidence that an answer is correct.

Use `detail.items[].opportunity_score` when the UI needs the FAQ opportunity
ranking that combines frequency and failure-risk signals.

## Detail Payload

Hydrate the full generated FAQ with `results[0].faq_id`. The detail route
returns the persisted FAQ draft:

| Field | Type | Meaning |
|---|---|---|
| `account_id` | string | Authenticated account scope. |
| `id` | string | FAQ draft ID; matches `results[0].faq_id`. |
| `target_id` | string | Persisted target/corpus identifier. |
| `target_mode` | string | Target type such as `support_account`. |
| `title` | string | FAQ report title. |
| `markdown` | string | Full generated FAQ Markdown. |
| `items` | list | Full generated FAQ items. |
| `source_count` | integer | Source rows considered. |
| `ticket_source_count` | integer | Ticket-like sources represented. |
| `output_checks` | object | Generator output-check status. |
| `warnings` | list | Generator warnings. |
| `metadata` | object | Persisted metadata such as corpus ID. |
| `status` | string | Draft status, usually `approved` for search. |

Each detail `items[]` entry carries the full generated FAQ shape:

- strings: `topic`, `question`, `question_source`, `summary`, `answer`,
  `answer_evidence_status`, `when_to_contact_support`
- integers: `frequency`, `weighted_frequency`, `ticket_count`,
  `opportunity_score`, `failure_risk_score`, `resolution_source_count`,
  `evidence_count`, `displayed_evidence_count`
- string lists: `failure_risk_signals`, `steps`, `action_items`,
  `evidence_quotes`, `source_ids`, `source_labels`
- count maps: `source_type_counts`, `weighted_source_volume_by_type`
- term mappings: `term_mappings`, rendered as `term_mappings[]`

Use detail `items[].steps` or `items[].action_items` for stepwise answer body
rendering. Search `answer_summary` is intentionally not the full answer.

## Term Mapping Shape

Each `term_mappings[]` entry uses:

- strings: `customer_term`, `documentation_term`, `suggestion`,
  `first_source_id`
- integers: `source_id_count`, `zero_result_source_count`,
  `failure_risk_score`, `opportunity_score`
- string list: `failure_risk_signals`

## Go-Live Gate

Before pointing a public demo at a host, run the checker with result and detail
requirements enabled:

```bash
python scripts/check_content_ops_faq_search_route_contract.py \
  --base-url "$ATLAS_API_BASE_URL" \
  --token "${ATLAS_B2B_JWT:-$ATLAS_TOKEN}" \
  --query "$ATLAS_FAQ_SEARCH_QUERY" \
  --require-results \
  --require-detail \
  --max-search-ms 1500 \
  --max-detail-ms 2500 \
  --max-total-ms 3500 \
  --output-result /tmp/faq-search-route-contract-result.json
```

The SaaS demo one-command smoke should remain the stronger proof because it
seeds known data, checks search and detail, and cleans up afterward.
