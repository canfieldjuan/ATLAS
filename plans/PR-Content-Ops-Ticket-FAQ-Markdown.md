# Content Ops Ticket FAQ Markdown

## Why this slice exists

The product can ingest support-ticket-like source rows and generate campaign
drafts from them, but operators do not yet have a fast, inspectable output that
turns those tickets into a grounded artifact. The requested first artifact is a
lightweight `.md` FAQ file based on ingested customer tickets so we can inspect
the evidence, verify grounding, and iterate quickly before adding storage or UI.

## Scope (this PR)

1. Add a deterministic ticket FAQ Markdown builder over normalized source-row
   opportunities.
2. Add a host-facing CLI that loads existing source-row JSON/JSONL/CSV files
   and writes or prints a Markdown FAQ.
3. Register the module in the extracted content manifest and CI runner.
4. Document the command in README/runbook/status.
5. Replace the stale merged #650 coordination row with this in-flight claim.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `scripts/build_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Ticket-FAQ-Markdown.md`

## Mechanism

The builder consumes the existing normalized opportunity shape produced by
`load_source_campaign_opportunities_from_file`. It keeps only ticket-like source
types (`support_ticket`, `case`, `conversation`, `complaint`), groups evidence
by the first pain point or source title, and renders one Markdown FAQ per group.
Answers are extractive: they quote compact snippets from the source evidence,
include a short "What to do next" action list, and list source ids/titles below
each answer. The result also exposes a small output-check map for the three
operator questions: user vocabulary, condensation, and action-item presence.

The CLI is intentionally offline and deterministic:

```bash
python scripts/build_extracted_ticket_faq_markdown.py \
  extracted_content_pipeline/examples/support_ticket_sources.csv \
  --source-format csv \
  --output support_ticket_faq.md
```

## Intentional

- No LLM in this first slice. The first FAQ output should be auditable and
  grounded before we add generative answer polish.
- No database table, control-surface output, or review workflow yet. This is a
  file artifact for fast iteration.
- No changes to source-row normalization. The builder uses the existing
  ingestion contract instead of creating a parallel ticket parser.

## Deferred

- Persisted `faq_markdown` generated-asset output and UI wiring.
- Optional LLM polish pass that must preserve citations and source snippets.
- Postgres-backed FAQ generation from imported opportunities.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py` -> 8 passed
- `python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py scripts/build_extracted_ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py` -> passed
- `python scripts/build_extracted_ticket_faq_markdown.py extracted_content_pipeline/examples/support_ticket_sources.csv --source-format csv --output /tmp/support_ticket_faq.md` -> passed
- `bash scripts/validate_extracted_content_pipeline.sh` -> passed
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -> passed
- `python scripts/audit_extracted_standalone.py --fail-on-debt` -> passed
- `bash scripts/check_ascii_python.sh` -> passed
- `bash scripts/local_pr_review.sh` -> passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Production + CLI | ~250 |
| Tests | ~175 |
| Docs + manifest + coordination | ~155 |
| **Total** | **~580** |

This is slightly over the 400 LOC target because this is the first instance of
the new FAQ artifact type and includes production code, CLI, tests, docs, and
manifest wiring. The implementation itself stays small and deterministic.
