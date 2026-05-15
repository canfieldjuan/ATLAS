# Content Ops Source Ticket Adapter

## Why This Slice Exists

The active AI Content Ops backlog is source breadth. Hosts can already convert
reviews, transcripts, complaints, and generic documents into campaign
opportunities. Support tickets, cases, and customer conversations are another
common customer-specific source bundle, but their field names do not currently
normalize cleanly.

## Scope

- Extend `extracted_content_pipeline/campaign_source_adapters.py` to recognize
  support-ticket and conversation row shapes.
- Update the packaged source-row example.
- Add focused source-adapter tests.
- Refresh host-facing docs and coordination state.

## Mechanism

- Add common source collection keys such as `support_tickets`, `tickets`,
  `cases`, and `conversations`.
- Add source id keys such as `ticket_id`, `case_id`, and `conversation_id`.
- Add source text keys such as `message`, `description`, `summary`, and
  `notes`.
- Treat ticket `subject` as source-title metadata so it does not become an
  email subject, contact title, company name, or contact name.
- Copy source ids into the normalized opportunity `id` when the source row does
  not already have an `id`, giving reviews, transcripts, documents, and tickets
  the same stable target-id behavior.
- Infer `source_type` as `support_ticket`, `case`, or `conversation` when
  source rows use those field families.

## Intentional

- No new CLI flags.
- No generated-asset changes.
- No database or migration changes.
- No new source format; JSON, JSONL, and CSV continue to use the existing
  loader path.

## Deferred

- Richer attachment/thread-message modeling.
- A dedicated source-quality score for support-ticket bundles.
- LLM reasoning over multi-message ticket threads.

## Verification

- Focused source-adapter tests.
- Compile check for touched Python files.
- Local PR review gate.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/examples/campaign_source_rows.jsonl`
- `plans/PR-Content-Ops-Source-Ticket-Adapter.md`
- `tests/test_extracted_campaign_source_adapters.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Source adapter | ~20 |
| Tests and example | ~80 |
| Docs and coordination | ~40 |
| Plan doc | ~45 |
| **Total** | ~185 |
