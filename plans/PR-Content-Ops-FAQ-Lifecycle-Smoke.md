# Content Ops FAQ Lifecycle Smoke

## Why this slice exists

The support-ticket FAQ path now covers source-row ingestion, deterministic FAQ
Markdown generation, Postgres persistence, export, and review/status updates.
Those seams are individually tested, but there is no single host-facing command
that proves the customer workflow works end to end:

1. source rows in,
2. FAQ Markdown draft saved,
3. draft exported for review,
4. draft moved to a host-defined publish/review status,
5. reviewed draft exported by status.

This slice closes that proof gap without adding another generation path or
changing the FAQ artifact shape.

This PR intentionally exceeds the 400 LOC soft cap because the operator smoke
and its fake-pool lifecycle tests need to ship together; splitting them would
leave either an untested host command or tests for a command that does not exist.

## Scope (this PR)

1. Add a focused FAQ lifecycle smoke command for host installs.
2. Reuse the real FAQ service, Postgres repository, export helper, and review
   status update seam.
3. Add fake-pool tests for the success path, schema fail-closed behavior, and
   status/export failure behavior.
4. Update Content Ops docs/status to mention the one-command lifecycle proof.
5. Replace the stale merged FAQ output-check in-flight row with this active
   slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Lifecycle-Smoke.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale merged FAQ row with this active slice. |
| `scripts/smoke_content_ops_faq_lifecycle.py` | Host-facing source rows -> persisted FAQ -> export -> review/status smoke. |
| `tests/test_smoke_content_ops_faq_lifecycle.py` | Focused CLI/run-function coverage with an asyncpg-shaped fake pool. |
| `scripts/run_extracted_pipeline_checks.sh` | Include the new lifecycle smoke test in the extracted gauntlet. |
| `extracted_content_pipeline/README.md` | Add the lifecycle smoke command to the FAQ docs. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Add the host install runbook command. |
| `extracted_content_pipeline/STATUS.md` | Record the lifecycle smoke capability. |

## Mechanism

`scripts/smoke_content_ops_faq_lifecycle.py` loads a JSON/JSONL/CSV source-row
file using the existing source adapter path, injects `PostgresTicketFAQRepository`
into `TicketFAQMarkdownService`, and calls `generate(...)` with the provided
`TenantScope`. When the service returns a saved FAQ id, the smoke exports the
new draft through `export_ticket_faq_drafts`, updates the same id through
`PostgresTicketFAQRepository.update_status(...)`, and exports the reviewed status
again.

The smoke validates each integration point instead of only checking that calls
return:

- `ticket_faq_markdown` table exists before generation starts.
- generation returns at least the requested number of saved ids.
- draft export returns the saved id and Markdown.
- status update returns a hit.
- reviewed-status export returns the same id with the requested status.

## Intentional

- No new storage table or API route. This proves the existing product seams.
- No LLM/provider dependency. FAQ Markdown remains deterministic and extractive.
- No source import into `campaign_opportunities`; FAQ generation consumes source
  rows directly and persists to `ticket_faq_markdown`.
- Review status remains host-defined. The smoke defaults to `published` because
  the goal is a publish-ready lifecycle, but hosts can pass another status.

## Deferred

- A real hosted UI button for this lifecycle remains a frontend/API follow-up.
- Semantic clustering remains separate from this lifecycle proof.
- Live database execution is host/operator responsibility; this PR tests the
  DB contract with an asyncpg-shaped fake pool and keeps the CLI ready for a
  real DSN.

## Verification

- pytest tests/test_smoke_content_ops_faq_lifecycle.py - 4 passed
- pytest tests/test_smoke_content_ops_faq_lifecycle.py tests/test_extracted_ticket_faq_postgres.py tests/test_extracted_ticket_faq_export.py tests/test_extracted_content_ops_execution.py::test_execute_runs_faq_markdown_service_from_source_material - 12 passed
- python -m py_compile scripts/smoke_content_ops_faq_lifecycle.py tests/test_smoke_content_ops_faq_lifecycle.py - passed
- python scripts/smoke_content_ops_faq_lifecycle.py --help - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1524 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Lifecycle-Smoke.md` | +105 |
| `docs/extraction/coordination/inflight.md` | +1 / -1 |
| `scripts/smoke_content_ops_faq_lifecycle.py` | +296 |
| `tests/test_smoke_content_ops_faq_lifecycle.py` | +175 |
| `scripts/run_extracted_pipeline_checks.sh` | +1 |
| `extracted_content_pipeline/README.md` | +14 |
| `extracted_content_pipeline/docs/host_install_runbook.md` | +17 |
| `extracted_content_pipeline/STATUS.md` | +3 |
| Total | ~612 |

The overage is justified in **Why this slice exists**.
