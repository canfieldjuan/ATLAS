# FAQ Output Selection Wiring

## Why this slice exists

#1109 made FAQ output consumable as source rows and #1113 bridged grounded FAQ
answers into the existing resolution-evidence contract. The next missing
product seam is selection: when the UI/API already has a saved FAQ draft payload,
the Atlas input provider should recognize that payload as valid Content Ops
source material and route it through the same support-ticket package path.

This keeps saved FAQ reuse at the ingestion boundary. The database fetch/UI
picker can come later; the request contract can already carry the selected FAQ
draft as `inputs.source_material`.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Vertical slice

1. Preserve a top-level saved FAQ draft ID when the input is a
   `TicketFAQDraft.as_dict()`-shaped bundle without `saved_ids`.
2. Teach the Atlas Content Ops input provider to accept FAQ output bundles and
   `faq_output` rows as support-ticket-derived source material.
3. Expand selected FAQ output bundles inside list-shaped `source_material` so
   mixed selections such as `[ticket_row, saved_faq_output]` keep the selected
   FAQ report.
4. Preserve FAQ output provenance fields when the support-ticket input package
   normalizes selected FAQ source rows.
5. Add focused tests proving a selected FAQ draft payload flows through the
   provider with `faq_output` provenance, draft traceability, and resolution
   evidence intact.

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/faq_output_ingestion.py`
- `extracted_content_pipeline/support_ticket_input_package.py`
- `atlas_brain/_content_ops_input_provider.py`
- `tests/test_extracted_ticket_faq_output_ingestion.py`
- `tests/test_atlas_content_ops_input_provider.py`
- `plans/PR-FAQ-Output-Selection-Wiring.md`

## Mechanism

The FAQ adapter now treats a top-level `id`, `faq_id`, `faq_draft_id`, or
`draft_id` as the report-level draft ID when `saved_ids` is absent. Multi-item
reports still get unique source IDs via the existing `:item-{rank}` suffix while
each row keeps `faq_draft_id`.

The Atlas provider imports the FAQ-output detector and source type constant:

```python
if is_faq_output_bundle(source_material):
    return SupportTicketInputProvider(...).build_content_ops_input_package(...)
```

It also accepts already-expanded rows where `source_type == "faq_output"`. From
there the existing `SupportTicketInputProvider` and
`build_support_ticket_input_package(...)` own normalization, output defaults,
resolution evidence, and request merge behavior.

The shared source-material adapter expands FAQ output bundles even when they
appear inside list-shaped `source_material`, which keeps mixed selections from
silently dropping selected FAQ reports.

The support-ticket input package keeps FAQ provenance fields such as
`faq_draft_id`, `faq_rank`, `faq_question`, and `faq_answer_evidence_status`
while normalizing selected source rows so generation and later diagnostics can
trace the selected report.

## Intentional

- This does not fetch FAQ drafts from Postgres. The route still receives
  selected source material from the caller; tenant-scoped fetching belongs in a
  later API/UI slice.
- This does not change generated content prompts or generation services.
- `faq_output` remains a derived source type, not raw support-ticket evidence.

## Deferred

- Future PR: add tenant-scoped API/UI selection by saved FAQ draft ID.
- Future PR: add an execute/preview smoke that selects a persisted FAQ draft
  through the real API once the fetch layer exists.
- Parked hardening: none.

## Verification

Ran locally:

- Command: python -m pytest tests/test_extracted_ticket_faq_output_ingestion.py tests/test_atlas_content_ops_input_provider.py tests/test_extracted_campaign_source_adapters.py tests/test_extracted_support_ticket_input_package.py -q
  - 114 passed, 1 warning
- Command: python -m py_compile extracted_content_pipeline/campaign_source_adapters.py extracted_content_pipeline/faq_output_ingestion.py extracted_content_pipeline/support_ticket_input_package.py atlas_brain/_content_ops_input_provider.py tests/test_extracted_ticket_faq_output_ingestion.py tests/test_atlas_content_ops_input_provider.py
  - passed
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
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-output-selection-wiring.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| FAQ adapter draft-ID fallback | ~15 |
| Source-material list expansion | ~15 |
| Atlas provider detection | ~20 |
| Support-ticket provenance passthrough | ~20 |
| Focused tests | ~125 |
| Plan doc | ~95 |
| **Total** | **~290** |
