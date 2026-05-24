# PR-Content-Ops-FAQ-Search-Projection-V1

## Why this slice exists

FAQ generation and ingestion scale tests prove batch creation, but demo retrieval
has a different failure mode: concurrent users should search compact generated
FAQ documents, scoped by tenant and corpus, instead of querying raw uploaded
ticket rows or full Markdown blobs.

This slice creates the thinnest end-to-end retrieval contract:
`TicketFAQDraft` -> search projection documents -> deterministic query envelope.
It answers the current architecture question by establishing a separate search
projection seam before a hosted route or database index exists.

Review follow-up tightened two contract boundaries: `search_text` stays internal
to the projection and blank tenant searches fail closed.

## Scope (this PR)

Ownership lane: content-ops/faq-search

Slice phase: Vertical slice.

1. Add a small FAQ search projection helper over generated `TicketFAQDraft`
   items.
2. Add deterministic in-memory search that returns the future route envelope:
   `{query, results, count}`.
3. Preserve tenant/corpus isolation in the projection and search filters.
4. Register the new owned module in the extracted package manifest.
5. Add focused tests for projection fields, query ranking, envelope shape, and
   tenant/corpus isolation.
6. Keep internal index text out of serialized document/result payloads and
   reject blank tenant searches.

### Files touched

- `plans/PR-Content-Ops-FAQ-Search-Projection-V1.md`
- `extracted_content_pipeline/ticket_faq_search.py`
- `extracted_content_pipeline/manifest.json`
- `tests/test_extracted_ticket_faq_search.py`

## Mechanism

`ticket_faq_search.py` will build one compact `TicketFAQSearchDocument` per FAQ
item. Each document carries the scoped identifiers needed by the eventual route:
`account_id`, `corpus_id`, `faq_id`, `target_mode`, `status`, item rank, topic,
question, answer summary, source ids, ticket count, and normalized `search_text`.
`search_text` is used for scoring only and is intentionally excluded from the
serialized document/result payload.

`search_ticket_faq_documents(...)` will filter by `account_id`, optional
`corpus_id`, and optional `status`, score deterministic token matches over the
compact search text, and return:

```python
{
    "query": query,
    "results": [result.as_dict(), ...],
    "count": len(results),
}
```

This is not the final storage layer. It locks the API-shaped result contract and
the isolation/ranking semantics so the next slice can swap the backing store to
Postgres without changing the consumer envelope.

## Intentional

- No hosted FastAPI route lands in this slice. The route should sit on top of
  the projection contract after the data shape is proven.
- No Postgres migration or full-text index lands here. This PR defines the
  projection boundary and deterministic behavior first; database indexing is the
  next natural slice.
- No embeddings/vector search are added. Exact/token scoring is enough for the
  first production-safe demo seam and is easier to reason about under tests.

## Deferred

- `PR-Content-Ops-FAQ-Search-Postgres-Projection`: write projected FAQ item rows
  to a dedicated table with tenant/corpus/status indexes and Postgres search
  indexes.
- `PR-Content-Ops-FAQ-Deflection-Search-Route`: expose the hosted route that
  returns `{query, results, count}` for the demo client.
- `PR-Content-Ops-FAQ-Search-Concurrency-Smoke`: seed multiple corpora and run
  concurrent filtered searches with latency and isolation assertions.
- Result highlighting/snippets and synonym expansion remain deferred until the
  core retrieval seam exists.

## Verification

- `pytest tests/test_extracted_ticket_faq_search.py -q` - 6 passed after review fixes.
- `python -m py_compile extracted_content_pipeline/ticket_faq_search.py tests/test_extracted_ticket_faq_search.py` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/local_pr_review.sh --allow-dirty` - passed after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 111 |
| Search projection module | 227 |
| Manifest entry | 3 |
| Tests | 190 |
| **Total** | **531** |

The estimate is slightly above the 400 LOC target because this is a complete
vertical proof: projection, search envelope, manifest registration, and
isolation/ranking tests. If implementation runs larger than expected, route and
database work stay deferred rather than expanding this PR further.
