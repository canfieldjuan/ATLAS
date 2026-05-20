# PR-Content-Ops-FAQ-Execution-Smoke

## Why this slice exists

`faq_markdown` is now a runnable Content Ops output and has persisted
generated-asset review/export support, but the offline execution smoke still
cannot prove that output through the same `POST /content-ops/execute` seam. The
smoke service bundle does not include `TicketFAQMarkdownService`, and the smoke
payload validator treats every non-signal output as requiring `saved_ids`.

This slice makes the host smoke prove FAQ Markdown execution without requiring
Postgres, providers, or Atlas runtime imports.

## Scope (this PR)

1. Wire `TicketFAQMarkdownService` into the offline execution smoke services.
2. Add a configurable source type to the smoke source row so FAQ generation can
   identify support-ticket evidence.
3. Teach the smoke validator that `faq_markdown` is a deterministic Markdown
   output validated by Markdown, FAQ items, and output checks rather than
   `saved_ids`.
4. Add focused subprocess coverage for `--outputs faq_markdown`.
5. Update docs/status examples for the new smoke command.
6. Replace the stale in-flight row with this slice's coordination claim.

### Files touched

- `plans/PR-Content-Ops-FAQ-Execution-Smoke.md`
- `docs/extraction/coordination/inflight.md`
- `scripts/smoke_extracted_content_ops_execution.py`
- `tests/test_extracted_content_ops_execution_smoke.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/control_surface_preview_api.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`

## Mechanism

The smoke script will import `TicketFAQMarkdownService` and provide it through
`ContentOpsExecutionServices(faq_markdown=...)`. The smoke payload will include
`source_type` from a new `--source-type` option, defaulting to `support_ticket`,
so the FAQ builder receives support-ticket-shaped source material.

`_step_has_output_payload(...)` will keep the existing `saved_ids` policy for
provider-backed generated assets, keep the `opportunities` policy for
`signal_extraction`, and add an explicit FAQ policy: Markdown must be non-empty,
items must be present, and every `output_checks` value must be exactly `True`.

## Intentional

- This is still an offline smoke. It does not persist FAQ drafts or exercise the
  generated-asset export/review switchboards from the previous PR.
- `faq_markdown` remains excluded from reasoning usage validation because the
  FAQ builder is deterministic and does not consume reasoning contexts.
- The smoke validates the three FAQ output checks indirectly through
  `output_checks`; detailed FAQ rendering tests stay in
  `tests/test_extracted_ticket_faq_markdown.py`.

## Deferred

- A real Postgres-backed FAQ persistence smoke can follow if host operators need
  one. The current goal is execution seam coverage without database setup.
- Dashboard rendering for persisted FAQ drafts remains a later UI slice.

## Verification

Local checks:

- pytest tests/test_extracted_content_ops_execution_smoke.py tests/test_extracted_ticket_faq_markdown.py -> 30 passed
- python -m py_compile scripts/smoke_extracted_content_ops_execution.py tests/test_extracted_content_ops_execution_smoke.py -> passed
- bash scripts/validate_extracted_content_pipeline.sh -> passed
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -> passed
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed
- bash scripts/check_ascii_python.sh -> passed
- bash scripts/run_extracted_pipeline_checks.sh -> 1493 passed, 1 existing torch/pynvml warning
- bash scripts/local_pr_review.sh -> passed after commit

## Estimated diff size

| Area | Estimate |
|---|---:|
| Smoke script | 35 |
| Tests | 45 |
| Docs and plan | 115 |
| **Total** | **195** |
