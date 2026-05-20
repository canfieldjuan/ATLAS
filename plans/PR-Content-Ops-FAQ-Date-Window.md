# PR-Content-Ops-FAQ-Date-Window

## Why this slice exists

The support-ticket FAQ pipeline can load customer CSV files and generate
grounded Markdown, but it cannot yet honestly support the "CSV - Last 90 days"
claim. Operators can pre-filter files outside Atlas, but the product seam should
also enforce a recency window when requested so stale tickets do not influence
FAQ entries.

This PR is slightly above the 400-line target because review feedback required
strict as-of-date validation at the library, plan, and CLI boundaries plus
regression tests for each path. Keeping that validation with the date-window
feature avoids shipping a reproducibility footgun.

## Scope (this PR)

1. Add optional date-window filtering to the ticket FAQ Markdown builder and
   service.
2. Thread faq_window_days and faq_as_of_date through the Content Ops generation
   plan and executor for the FAQ Markdown output.
3. Add CLI flags for the standalone FAQ Markdown script.
4. Document the last-90-days command shape in the README and host runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Date-Window.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Claim this in-flight slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Optional recency filtering and config fields. |
| `extracted_content_pipeline/generation_plan.py` | Map FAQ recency inputs into step config. |
| `extracted_content_pipeline/content_ops_execution.py` | Pass FAQ recency config into the service. |
| `scripts/build_extracted_ticket_faq_markdown.py` | Add standalone CLI flags. |
| `tests/test_extracted_ticket_faq_markdown.py` | Builder/service/CLI regression coverage. |
| `tests/test_extracted_content_generation_plan.py` | Plan config regression coverage. |
| `tests/test_extracted_content_ops_execution.py` | Executor threading regression coverage. |
| `extracted_content_pipeline/README.md` | Document the CSV recency command. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Document the host runbook recency command. |

## Mechanism

The FAQ builder will accept an optional positive window_days value and optional
as_of_date. When a window is supplied, it will keep only source opportunities or
evidence rows with a recognized date field inside the inclusive window. Rows
without a parseable date are excluded only when a date window is active.

Date lookup stays local to ticket_faq_markdown.py instead of changing the shared
source adapter. That keeps the new behavior FAQ-specific and avoids changing
campaign/source ingestion semantics.

## Intentional

- No default date filter. Existing FAQ calls remain unchanged unless the caller
  passes faq_window_days or the CLI passes --window-days.
- No semantic intent clustering in this slice. This only makes the recency part
  of the customer-facing claim real.
- No frontend control is added. The API/CLI can pass the inputs now; a later UI
  slice can expose a form field if needed.

## Deferred

- Semantic intent clustering for Step 02 remains a follow-up slice.
- Inline edit/publish for Step 06 remains a follow-up slice; review/export is
  already available.
- A polished frontend date-window selector can ride with the broader FAQ
  workflow UI.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py - 18 passed
- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py - 104 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py scripts/build_extracted_ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py - passed
- bash scripts/validate_extracted_content_pipeline.sh - passed
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline - passed
- python scripts/audit_extracted_standalone.py --fail-on-debt - passed
- bash scripts/check_ascii_python.sh - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1502 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Date-Window.md` | +95 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +140 / -1 |
| `extracted_content_pipeline/generation_plan.py` | +24 / -7 |
| `extracted_content_pipeline/content_ops_execution.py` | +2 |
| `scripts/build_extracted_ticket_faq_markdown.py` | +13 |
| `tests/test_extracted_ticket_faq_markdown.py` | +138 |
| `tests/test_extracted_content_generation_plan.py` | +31 |
| `tests/test_extracted_content_ops_execution.py` | +12 |
| `extracted_content_pipeline/README.md` | +4 / -1 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | +3 / -1 |
| Total | ~484 |
