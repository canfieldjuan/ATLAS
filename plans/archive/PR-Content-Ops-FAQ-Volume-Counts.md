# Content Ops FAQ Volume Counts

## Why this slice exists

FAQ Markdown now clusters support tickets by deterministic intent, and the
builder already ranks clusters by the full number of matched evidence rows.
However, `_item(...)` only receives the capped display rows, so the rendered
answer and item metadata can underreport volume. A 10-ticket intent cluster with
`max_evidence_per_item=3` can look like it only came from 3 tickets.

This weakens the customer-facing "Rank by Volume" workflow claim. This slice
keeps the source-level ranking behavior and makes the total cluster volume
visible without changing ingestion or adding LLM behavior.

## Scope (this PR)

1. Pass the full grouped evidence rows into the FAQ item builder.
2. Keep snippet/source-label rendering capped by `max_evidence_per_item`.
3. Add `ticket_count` and `displayed_evidence_count` item metadata so hosts can
   distinguish total volume from displayed evidence.
4. Render the total ticket-source count in the answer sentence.
5. Add regression coverage for a capped high-volume intent cluster.
6. Replace the stale FAQ intent-clustering in-flight row with this active slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Volume-Counts.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale FAQ intent-clustering claim with this active slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Preserve total cluster volume while capping displayed snippets. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for volume count vs displayed snippet count. |

## Mechanism

`build_ticket_faq_markdown(...)` will pass each full sorted group to `_item(...)`
alongside `max_evidence_per_item`. `_item(...)` will compute total unique source
ids from the full group, then slice only the display rows for snippets and source
labels. Existing `evidence_count` keeps its old meaning: displayed evidence row
count. New metadata carries the total volume.

## Intentional

- No change to source ingestion, date filtering, or intent clustering.
- No change to the `max_evidence_per_item` display cap; the Markdown remains
  readable even when a cluster has many tickets.
- `evidence_count` remains backward-compatible as the displayed count. New
  `ticket_count` exposes the full volume.

## Deferred

- UI ordering badges or volume chips are deferred to a frontend slice.
- Semantic clustering beyond deterministic intent rules remains deferred.
- Operator-side edit/publish workflow remains separate from Markdown generation.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_runs_faq_markdown_json - 27 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1507 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Volume-Counts.md` | +72 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +13 / -5 |
| `tests/test_extracted_ticket_faq_markdown.py` | +29 |
| Total | ~123 |
