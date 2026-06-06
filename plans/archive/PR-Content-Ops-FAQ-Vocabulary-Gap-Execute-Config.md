# PR-Content-Ops-FAQ-Vocabulary-Gap-Execute-Config

## Why this slice exists

Vocabulary-gap detection works in the FAQ generator and CLI, and reviewers can
now see generated mappings in the asset review drawer. The hosted Content Ops
execution path still cannot pass documentation terms or custom vocabulary-gap
rules from request inputs into `TicketFAQMarkdownService.generate`, so hosted
runs cannot intentionally exercise that feature.

This slice threads the existing generator configuration through the real
`/content-ops/execute` planning and dispatch path.

## Scope (this PR)

Ownership lane: content-ops/faq-generator-execute-config

1. Accept `faq_documentation_terms` from request inputs and pass them into FAQ
   Markdown generation config.
2. Accept `faq_vocabulary_gap_rules` as an array of term arrays and pass them
   into FAQ Markdown generation config.
3. Include those values in the FAQ plan step config so execution dispatch can
   use the same planned values.
4. Add focused execution coverage proving hosted FAQ generation emits a
   vocabulary-gap mapping.
5. Reuse the FAQ library vocabulary-rule normalizer and pin invalid hosted API
   input to a clean `400` response.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Vocabulary-Gap-Execute-Config.md` | Plan doc for this hosted execution config slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Exposes the shared vocabulary-gap rule normalizer for plan/dispatch reuse. |
| `extracted_content_pipeline/generation_plan.py` | Reads FAQ documentation terms and vocabulary rules from request inputs and includes them in step config. |
| `extracted_content_pipeline/content_ops_execution.py` | Dispatches FAQ documentation terms and vocabulary rules to the service. |
| `tests/test_extracted_content_control_surface_api.py` | Proves invalid hosted FAQ vocabulary rules return a clean 400 response. |
| `tests/test_extracted_content_ops_execution.py` | Proves hosted execution produces term mappings from request-supplied config. |

## Mechanism

`TicketFAQMarkdownConfig` already has `documentation_terms` and
`vocabulary_gap_rules`, and `TicketFAQMarkdownService.generate(...)` already
accepts both as per-call overrides. This slice fills the missing middle:

```python
request.inputs -> GenerationPlanStep.config -> _dispatch_faq_markdown(...)
```

`faq_documentation_terms` uses the existing string-sequence input helper.
`faq_vocabulary_gap_rules` accepts a JSON-like sequence of string sequences,
matching the service/library shape. Both the generation plan and dispatcher use
the FAQ library normalizer so the array-of-arrays contract does not drift.

## Intentional

- Backend execution threading only. No new generator scoring, Markdown
  rendering, persistence, CLI, or hosted UI controls.
- Custom vocabulary rules must be structured arrays in this slice. Textarea or
  CSV parsing for hosted UI input remains separate.
- Empty or missing optional inputs preserve existing behavior.

## Deferred

- Hosted UI controls for entering documentation terms and vocabulary rules
  remain a separate product slice.
- Per-file documentation term upload in hosted runs remains separate.
- Parked hardening considered: current `HARDENING.md` entries are landing-page
  repair items and do not touch this FAQ lane.

## Verification

- `python -m pytest tests/test_extracted_content_ops_execution.py::test_execute_runs_faq_markdown_service_from_source_material` - passed, 1 test.
- `python -m pytest tests/test_extracted_content_control_surface_api.py::test_execute_generation_route_rejects_invalid_faq_vocabulary_rules_as_400 tests/test_extracted_content_ops_execution.py::test_execute_runs_faq_markdown_service_from_source_material tests/test_extracted_ticket_faq_markdown.py::test_build_ticket_faq_markdown_rejects_invalid_custom_vocabulary_gap_rules` - passed, 5 tests.
- `python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py extracted_content_pipeline/generation_plan.py extracted_content_pipeline/content_ops_execution.py tests/test_extracted_content_control_surface_api.py tests/test_extracted_content_ops_execution.py` - passed.
- `git diff --check` - passed.
- `python -m pytest tests/test_extracted_content_ops_execution.py::test_execute_runs_faq_markdown_service_from_source_material tests/test_extracted_content_ops_execution.py tests/test_extracted_content_ops_execution_smoke.py` - passed, 67 tests.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed, 0 Atlas runtime import findings.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - passed, 1759 tests and 1 existing `torch`/`pynvml` warning.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~90 |
| Shared vocabulary-rule normalizer | ~30 |
| Generation plan threading | ~35 |
| Execution dispatch threading | ~20 |
| Tests | ~65 |
| **Total** | ~240 |
