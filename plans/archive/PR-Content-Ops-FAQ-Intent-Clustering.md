# Content Ops FAQ Intent Clustering

## Why this slice exists

The support-ticket FAQ path now supports uploaded/loaded ticket rows and a
recency window, but its clustering is still too literal: FAQ topics are mostly
pulled from `pain_points` or source titles. That means tickets with the same
user intent but different titles can render as separate FAQ entries, which
weakens the product claim that the pipeline clusters by intent and ranks by
volume.

This slice adds a small deterministic intent-clustering layer to the FAQ
Markdown builder. It stays extractive and provider-free, and it avoids mixing
in the separate storytelling/LLM generation track.

## Scope (this PR)

1. Add default FAQ intent rules for common support-ticket intents.
2. Thread optional intent rules through `TicketFAQMarkdownConfig`, service
   generation, and `build_ticket_faq_markdown`.
3. Use the intent cluster before falling back to existing pain/category/title
   grouping.
4. Add focused tests proving repeated user intent collapses into one FAQ item
   and custom intent rules can be supplied by a host.
5. Replace the stale FAQ date-window in-flight row with this active slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Intent-Clustering.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale FAQ date-window claim with this active slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Add configurable deterministic intent rules at the FAQ grouping source. |
| `tests/test_extracted_ticket_faq_markdown.py` | Builder and service regression coverage for intent clustering. |

## Mechanism

Add a module-level default intent taxonomy shaped as `(topic, keywords)` pairs.
For each evidence row, `_topic(...)` will inspect the pain fields, source title,
and evidence text, then return the first matching configured topic. If no rule
matches, it preserves the current fallback: first explicit pain point, then
source title, then `customer support issues`.

The rules are optional parameters, so hosts can override or disable the packaged
taxonomy without changing source ingestion or output rendering.

## Intentional

- This is deterministic string matching, not semantic embeddings or LLM
  clustering. That is enough to support the FAQ Markdown smoke path without
  adding provider cost or nondeterminism.
- Existing explicit pain/category values remain the fallback so current output
  shape does not regress when no intent rule matches.
- The default taxonomy is intentionally small and support-ticket focused; it is
  not a generic knowledge graph.
- The stale FAQ date-window coordination row is removed in the same file edit
  because that PR is merged and this PR claims the next FAQ slice.

## Deferred

- Semantic clustering beyond keyword rules remains deferred until a real host
  export shows the deterministic taxonomy is insufficient.
- Operator-side edit/publish workflow remains separate from Markdown
  generation.
- UI copy for the six-step FAQ workflow is not changed here.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_runs_faq_markdown_json - 26 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed
- git diff --check - passed
- bash scripts/validate_extracted_content_pipeline.sh - passed
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed
- bash scripts/check_ascii_python.sh - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1506 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Intent-Clustering.md` | +86 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +65 / -2 |
| `tests/test_extracted_ticket_faq_markdown.py` | +111 |
| Total | ~268 |
