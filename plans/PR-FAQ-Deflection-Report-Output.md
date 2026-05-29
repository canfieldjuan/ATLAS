# PR-FAQ-Deflection-Report-Output

## Why this slice exists

PR-Content-Ops-Deflection-Report-Artifact added a deterministic renderer for
the $1,500 support-ticket deflection report, but the real Content Ops execute
surface still only exposes `faq_markdown`. That leaves the deliverable usable
from the CLI but not from the normal `/content-ops/execute` product flow.

This slice wires the report artifact as a first-class deterministic Content Ops
output so a support-ticket upload can produce the customer-facing report through
the same execution path as the FAQ generator.

This exceeds the 400 LOC target because adding one real output must touch the
catalog, plan, executor, host bundle, reasoning-policy parity, and focused tests
at each integration point. Splitting those would leave a visible output without
an executable product path, or an executable path without contract coverage.

## Scope (this PR)

Ownership lane: content-ops/deflection-report

Slice phase: Vertical slice

1. Add a `faq_deflection_report` output definition and generation-plan mapping.
2. Add a service-shaped deflection report generator that reuses the existing
   `TicketFAQMarkdownService` and report renderer.
3. Wire the host execution-services bundle so the output is available without an
   LLM and persists FAQ drafts when DB services are enabled.
4. Prove the output works through the executor and the hosted execute route.

### Files touched

| File | Purpose |
|---|---|
| `extracted_content_pipeline/control_surfaces.py` | Adds the output catalog entry. |
| `extracted_content_pipeline/generation_plan.py` | Maps the output to its runner/config. |
| `extracted_content_pipeline/content_ops_execution.py` | Dispatches the output through the report service. |
| `extracted_content_pipeline/faq_deflection_report.py` | Adds the service-shaped generator. |
| `extracted_content_pipeline/reasoning_policy.py` | Marks the deterministic output as no-reasoning only. |
| `atlas_brain/_content_ops_services.py` | Wires the host service bundle. |
| `tests/test_extracted_content_ops_execution.py` | Covers executor behavior. |
| `tests/test_extracted_content_ops_live_execute_harness.py` | Covers hosted route behavior. |
| `tests/test_atlas_content_ops_execution_services.py` | Covers host bundle availability. |
| `tests/test_extracted_content_generation_plan.py` | Covers plan mapping/config. |
| `tests/test_extracted_content_reasoning_policy.py` | Covers reasoning-policy parity. |
| `plans/PR-FAQ-Deflection-Report-Output.md` | Documents the slice contract. |

## Mechanism

`FAQDeflectionReportService.generate(...)` takes the same support-ticket inputs
as `faq_markdown`, delegates row normalization and FAQ generation to
`TicketFAQMarkdownService.generate(...)`, and wraps the resulting
`TicketFAQMarkdownResult` with `build_deflection_report_artifact(...)`.

The new `faq_deflection_report` output shares the FAQ config knobs
(`faq_title`, `faq_max_evidence_per_item`, documentation terms, vocabulary-gap
rules) and adds `deflection_report_title` for the customer-facing report title.
The executor returns the artifact dictionary shape, including `markdown`,
`summary`, and the underlying `faq_result`.

## Intentional

- Keep `faq_markdown` unchanged so existing saved FAQ/search flows remain
  stable.
- Do not create a separate DB table for deflection reports in this vertical
  slice; persistence stays with the underlying saved FAQ drafts.
- Use `faq_deflection_report` instead of overloading the existing generic
  `report` output, because that output is already a different intelligence
  report generator.

## Deferred

- Future PR: persist/export named deflection report artifacts if customers need
  report lifecycle separate from saved FAQ drafts.
- Future PR: add UI controls for `faq_deflection_report` after the backend
  execute contract is stable.
- Parked hardening: none.

## Verification

- Command: python -m pytest tests/test_extracted_content_ops_execution.py tests/test_atlas_content_ops_execution_services.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_reasoning_policy.py -q -- 142 passed.
- Command: python -m py_compile extracted_content_pipeline/faq_deflection_report.py extracted_content_pipeline/control_surfaces.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py extracted_content_pipeline/reasoning_policy.py atlas_brain/_content_ops_services.py tests/test_extracted_content_ops_execution.py tests/test_atlas_content_ops_execution_services.py tests/test_extracted_content_ops_live_execute_harness.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_reasoning_policy.py -- passed.
- Command: git diff --check -- passed.
- Command: bash scripts/validate_extracted_content_pipeline.sh -- passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt -- passed.
- Command: bash scripts/check_ascii_python.sh -- passed.
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline -- passed.
- Command: bash scripts/run_extracted_pipeline_checks.sh -- 2671 passed, 9 skipped, 1 warning.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /home/juan-canfield/Desktop/atlas-pr-bodies/faq-deflection-report-output.md -- pending.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Service, catalog, plan, executor, and host wiring | 148 |
| Integration tests | 249 |
| Plan doc | 100 |
| **Total** | **497** |
