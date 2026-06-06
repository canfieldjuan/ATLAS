# FAQ Output Resolution Evidence Contract

## Why this slice exists

#1109 made generated FAQ output reusable as source material, but the reviewer
flagged the next real integration gap: the FAQ adapter preserves
`faq_answer_evidence_status`, yet the support-ticket truthfulness gate reads the
existing `resolution_text` contract. Without that bridge, grounded FAQ answers
can be collected but not counted by the same gate that protects blog and landing
generation from invented procedural steps.

This slice keeps the fix at the ingestion boundary so blog, landing, and support
ticket packaging all consume the same evidence shape.

## Scope (this PR)

Ownership lane: content-ops/faq-output-ingestion
Slice phase: Vertical slice

1. Map FAQ items with `answer_evidence_status="resolution_evidence"` to the
   existing `resolution_text` source-row field using their generated,
   resolution-backed steps.
2. Keep draft FAQ items (`draft_needs_review`) out of the resolution-evidence
   contract even when they contain placeholder review steps.
3. Add focused tests proving the adapter exposes resolution evidence and that
   the support-ticket input package counts it through the existing
   `support_ticket_resolution_evidence_*` fields.

### Files touched

- `extracted_content_pipeline/faq_output_ingestion.py`
- `tests/test_extracted_ticket_faq_output_ingestion.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `plans/PR-FAQ-Output-Resolution-Evidence-Contract.md`

## Mechanism

The FAQ output adapter already emits `faq_answer_evidence_status`. This slice
adds a narrow mapping:

```python
if answer_evidence_status == "resolution_evidence" and steps:
    row["resolution_text"] = " ".join(steps)
```

That uses the existing support-ticket package path instead of adding a parallel
FAQ-only gate. `build_support_ticket_input_package(...)` already treats
`resolution_text` as the canonical signal for:

- `support_ticket_resolution_evidence_present`
- `support_ticket_resolution_evidence_count`
- `support_ticket_resolution_examples`

So FAQ output can unlock grounded procedural answers only when the FAQ lane has
already marked the item as resolution-backed.

## Intentional

- This does not teach the generator a new FAQ-specific truthfulness concept. It
  bridges FAQ output into the existing `resolution_text` evidence contract.
- Draft/review-needed FAQ items keep their customer wording, topic, and
  placeholder steps, but they do not set `resolution_text`.
- This remains source-material plumbing. It does not fetch saved FAQ drafts from
  a tenant store or add UI/API controls for selecting a FAQ report.

## Deferred

- Future PR: add UI/API selection so a user can intentionally feed a saved FAQ
  report into blog or landing-page generation.
- Parked hardening: none.

## Verification

Ran locally:

- Command: python -m pytest tests/test_extracted_ticket_faq_output_ingestion.py tests/test_extracted_support_ticket_input_package.py -q
  - 28 passed
- Command: python -m py_compile extracted_content_pipeline/faq_output_ingestion.py tests/test_extracted_ticket_faq_output_ingestion.py tests/test_extracted_support_ticket_input_package.py
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
- Command: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr-faq-output-resolution-evidence-contract.md
  - passed

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Adapter mapping | ~20 |
| Focused tests | ~65 |
| Plan doc | ~85 |
| **Total** | **~170** |
