Plan: PR-Content-Ops-FAQ-Generated-Asset-Seam

## Why this slice exists

PR-Content-Ops-Ticket-FAQ-Markdown added a grounded FAQ Markdown builder and
CLI, but the artifact is still outside the AI Content Ops control-surface
execution path. Operators can build a `.md` file manually, but `/execute` cannot
plan or run `faq_markdown` alongside the other outputs. This slice closes that
first integration gap without adding database persistence or UI review flows.

This PR is over the 400 LOC target after review-driven fixes because the seam is
only complete if Atlas's default Content Ops service bundle wires the same
deterministic FAQ service that the extracted executor advertises. Splitting that
host wiring into a separate PR would leave `/execute` reporting
`service_not_configured` for an implemented output.

## Scope (this PR)

1. Add `faq_markdown` to the control-surface output catalog as a zero-cost,
   extractive output that requires `source_material`.
2. Add a service-shaped FAQ executor around the existing deterministic builder.
3. Wire `faq_markdown` through generation planning and content-ops execution.
4. Wire Atlas's default Content Ops service bundle to expose the deterministic
   FAQ service in production routes.
5. Add focused tests for preview, plan, host bundle, and execution wiring.

### Files touched

- `docs/extraction/coordination/inflight.md`
- `atlas_brain/_content_ops_services.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `extracted_content_pipeline/control_surfaces.py`
- `extracted_content_pipeline/generation_plan.py`
- `extracted_content_pipeline/content_ops_execution.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-FAQ-Generated-Asset-Seam.md`
- `tests/test_atlas_content_ops_execution_services.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_content_control_surfaces.py`
- `tests/test_extracted_content_control_surface_api.py`
- `tests/test_extracted_content_generation_plan.py`
- `tests/test_extracted_content_ops_execution.py`

## Mechanism

`TicketFAQMarkdownService.generate(...)` accepts inline `source_material`, uses
the existing source-row adapter to normalize rows into opportunities, and then
calls `build_ticket_faq_markdown(...)`. The control surface treats the output
like `signal_extraction`: no LLM, no reasoning provider, no retry budget.

Execution dispatch registers a `faq_markdown` handler that passes
`source_material`, `target_mode`, and plan config into the service. The result
continues to use `TicketFAQMarkdownResult.as_dict()`, so execution summaries get
`markdown`, `items`, source counts, normalization warnings, and output checks
without a new result shape.

Atlas's host service factory now wires a singleton `TicketFAQMarkdownService`
alongside `SignalExtractionService`, because both are deterministic and require
no active LLM or database pool. DB-backed generated assets remain behind
`enable_db_services`.

## Intentional

- No Postgres table, generated-asset API route, or review workflow in this PR.
  The output is runnable through `/execute`, but persistence stays deferred.
- No LLM polish or reasoning pass. This keeps the first FAQ integration grounded
  and inspectable before adding summarization.
- `limit` controls max FAQ items for `/execute`; the CLI keeps its richer
  `--max-items` wording.

## Deferred

- Persist `faq_markdown` drafts as reviewable generated assets.
- Add UI selection/rendering for the FAQ output.
- Add optional public/help-center FAQ render mode after we inspect internal
  review Markdown quality.

## Verification

- Focused pytest sweep over FAQ, host bundle, control-surface, plan, execution, and API tests -> 195 passed
- Python compile sweep over edited modules/tests -> passed
- validate_extracted_content_pipeline -> passed
- forbid_atlas_reasoning_imports -> passed
- audit_extracted_standalone --fail-on-debt -> passed
- check_ascii_python -> passed
- git diff --check -> passed
- run_extracted_pipeline_checks -> 1480 passed, 1 existing torch/pynvml warning

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Service + control/execution wiring | ~205 |
| Tests | ~220 |
| Docs + coordination | ~25 |
| **Total** | **~450-525** |
