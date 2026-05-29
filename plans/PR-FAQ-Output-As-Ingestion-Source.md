# FAQ Output As Ingestion Source

## Why this slice exists

The FAQ report lane can already turn support tickets into ranked FAQ items, and
the blog/landing lanes can already ingest generic source material. The missing
contract is the bridge between those two systems: a generated FAQ report should
be reusable as source material without creating a one-off blog or landing-page
patch.

This slice builds that bridge at the source-adapter boundary so future FAQ
output shape changes can be absorbed in one adapter instead of scattered across
content generators.

This is slightly over the 400-LOC soft cap because the adapter, source-material
wiring, runner enrollment, and biting tests need to land together for the new
contract to be real.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Vertical slice

1. Add a small FAQ-output ingestion adapter that converts FAQ result dictionaries
   into canonical source rows.
2. Wire `source_material_to_source_rows(...)` to recognize FAQ result bundles and
   route them through the adapter before generic bundle expansion.
3. Add focused tests proving FAQ output rows normalize into campaign
   opportunities while preserving source type, customer wording, FAQ metadata,
   and no-items behavior.

### Files touched

- `extracted_content_pipeline/faq_output_ingestion.py`
- `extracted_content_pipeline/campaign_source_adapters.py`
- `tests/test_extracted_ticket_faq_output_ingestion.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-FAQ-Output-As-Ingestion-Source.md`

## Mechanism

`faq_output_to_source_rows(...)` accepts the dictionary shape returned by
`TicketFAQMarkdownResult.as_dict()` and produces rows with:

- `source_type="faq_output"` so downstream consumers know the evidence is a
  derived FAQ artifact, not raw support tickets.
- stable source IDs from saved FAQ draft IDs when present, otherwise ranked FAQ
  item IDs.
- composed text containing the FAQ question, summary, steps, answer evidence
  status, and customer-language evidence quotes.
- metadata such as topic, question source, source ticket IDs, answer evidence
  status, and opportunity score.

`source_material_to_source_rows(...)` detects FAQ result bundles before generic
bundle expansion. That keeps the integration point central: callers that already
pass `source_material` to the campaign source adapter can pass FAQ output
directly without knowing about FAQ internals.

## Intentional

- This does not change FAQ generation, saved FAQ draft shape, blog prompts, or
  landing-page prompts. It only makes FAQ output consumable as source material.
- The adapter treats FAQ output as a derived source type (`faq_output`) instead
  of pretending the original support tickets are being ingested again.
- The adapter is forgiving about item fields because the FAQ output shape may
  evolve. Missing optional fields are skipped; items without usable question,
  summary, steps, answer, or evidence text are ignored.

## Deferred

- Future PR: add UI/API selection so a user can intentionally feed a saved FAQ
  report into blog or landing-page generation.
- Future PR: adapt this adapter if the FAQ lane introduces standalone FAQ article
  records with their own canonical IDs and URLs.
- Parked hardening: none.

## Verification

Ran locally:

- Command: python -m pytest tests/test_extracted_ticket_faq_output_ingestion.py tests/test_extracted_campaign_source_adapters.py -q
  - 72 passed; rerun after addressing the saved FAQ draft traceability review
    comment
- Command: python -m py_compile extracted_content_pipeline/faq_output_ingestion.py extracted_content_pipeline/campaign_source_adapters.py tests/test_extracted_ticket_faq_output_ingestion.py
  - passed
- Command: python scripts/audit_extracted_pipeline_ci_enrollment.py
  - passed; 125 matching tests are enrolled
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - passed
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - passed
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - passed
- Command: bash scripts/check_ascii_python.sh
  - passed
- Command: bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline
  - passed; no additional file changes
- Command: bash scripts/run_extracted_pipeline_checks.sh
  - passed; 2648 passed, 9 skipped
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-output-as-ingestion-source.md
  - passed
- Review update: addressed the bot P2 by preserving a report-level
  `saved_ids[0]` value as `faq_draft_id` on every generated item row and using
  an item suffix for unique `source_id` values in multi-item reports.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| FAQ output adapter | ~230 |
| Source adapter wiring | ~10 |
| Focused tests + runner enrollment | ~135 |
| Plan doc | ~110 |
| **Total** | **~485** |
