# PR-Deflection-Action-Section-Contract

## Why this slice exists

Issue #1612's report-shape plan is now locked around "work queue, not ticket
archive." #1743 supplied the Signal Spike and confirmed the S1 assumptions:
resolution/status signals are usable, CSAT and cost need honest degradation,
owner-lane/fix-type must allow `Unknown`, and snippet/phrasing fields remain
unsafe unless the free Snapshot projection is fail-closed.

Root cause: the free Snapshot/teaser boundary is still mostly protected by
handwritten denylist/scrub checks after report data is assembled. That is a
fail-open boundary: every new paid report field can accidentally become a new
free projection path unless the builder remembers to hide it.

This PR fixes the root for the S1 contract by making Snapshot projection an
allowlist construction tied to the paid `deflection.v1` section registry. The
paid model can add action-oriented sections, but those fields are absent from
Snapshot until explicitly marked snapshot-safe and projected through the shared
builder.

This slice exceeds the 400 LOC soft budget because the model contract is
intentionally golden-tested and documented: adding five section ids plus
`snapshot_safe_fields` updates the producer example JSON and explicit contract
expectations in the same PR. Splitting those tests/docs from the contract would
make the privacy boundary look smaller while leaving it less reviewable.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-actionability
Slice phase: Vertical slice

1. Extend the `deflection.v1` section contract with action-oriented paid
   sections: priority fix queue, top unresolved repeats, drafted resolutions,
   already-covered-still-recurring, and backlog table.
2. Add snapshot-safe field metadata to the section registry and make the free
   Snapshot derive from an allowlist projection of the report model instead of
   broad payload subtraction.
3. Keep the visible Markdown/email/PDF rendering unchanged in this slice; this
   is the model/projection contract that later renderers consume.
4. Archive the just-merged #1743 plan in the same branch per AGENTS teardown.

### Review Contract

- Acceptance criteria:
  - The paid report model contains the new action-section ids with deterministic
    data for status, owner lane, fix type, CSAT signal, confidence, cost basis,
    recommended action/title, representative phrasing, and capped evidence
    examples where available.
  - Snapshot-safe fields are explicit registry metadata, not implicit string
    filtering after the fact.
  - Snapshot-safe row fields are safe for every projected row; paid answer
    bodies and steps are not marked snapshot-safe and only flow through the
    separately gated teaser full-answer path.
  - A paid model field that is not snapshot-allowlisted is absent from the
    Snapshot projection, with a negative test that injects an unallowlisted
    paid field.
  - New action sections are absent from Snapshot by default, proving S3/S4 can
    add customer-facing snippets later without automatically widening the free
    teaser.
  - Legacy `deflection.v1` report models that lack the row fields Snapshot
    now requires fall back to the existing FAQ payload instead of emitting
    zeros/blanks.
  - Action-section evidence examples match quotes to their source IDs; sparse
    or out-of-order quotes do not attach to the wrong ticket.
  - `priority_fix_queue` carries enough rows for the largest advertised
    bounded surface (`pdf_limit`) while result pages still render their top-3
    slice, and `top_unresolved_repeats` excludes single-ticket items.
  - Cost/CSAT fields degrade honestly: benchmark-only cost is labeled as such,
    and missing CSAT reports `insufficient_data` rather than fabricated values.
- Affected surfaces: `extracted_content_pipeline.faq_deflection_report`,
  frontend report contract docs/examples, Snapshot projection tests, report
  model contract tests, plan/archive index.
- Risk areas: privacy/paywall boundary, model contract version drift, renderer
  assumptions about section shape, and report-size/actionability semantics.
- Reviewer rules triggered: R1, R2, R3, R8, R9, R10, R13, R14.
- boundary-probe: Required. This PR changes a privacy/paywall projection guard;
  review must inspect the injected-field and action-section absence tests.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/deflection_report_access.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Action-Section-Contract.md`
- `plans/archive/PR-Deflection-Signal-Spike.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The section registry gains `snapshot_safe_fields`, and the report-model contract
shape exposes those fields so future section changes require an intentional
version-contract update.

The paid model builder derives the five action sections from the same FAQ items
that already feed ranked questions and question details. The derivation is
deterministic: it ranks by ticket count/cost/opportunity, classifies each item
as draft-ready, needs-answer, needs-review, low-confidence, or
already-covered-still-recurring, and carries explicit cost/CSAT basis fields
instead of pretending sparse data is complete.

The Snapshot builder then reads a report-model projection helper. The helper
copies only fields marked snapshot-safe for `support_tax`, `ranked_questions`,
and row metadata in `question_details`. Paid answer bodies and steps are not
snapshot-safe row fields; the teaser reads raw detail rows only through the
existing scoped `resolution_evidence` eligibility gate. The projection refuses
legacy models that lack required row fields and falls back to the pre-existing
FAQ payload path instead of manufacturing blank/zero rows.

The action-section derivation keeps surface caps honest: the priority queue
stores enough items for the largest bounded consumer while exposing
`result_page_limit` and `pdf_limit`, unresolved repeats require at least two
tickets, and top evidence examples reuse the source-ID quote matcher from the
complete evidence export.

## Intentional

- No renderer refresh in this slice. Result page, email, and PDF action-layout
  changes are S3/S4 after the contract exists.
- No LLM classification. Owner lane and fix type are deterministic-only with
  explicit `Unknown` fallbacks.
- No complete evidence dump in bounded sections. Paid action sections carry
  top evidence examples only; the uncapped trail remains the evidence export.
- No Snapshot snippet/phrasing expansion yet. The allowlist boundary must land
  before S3 starts surfacing more customer-worded content.

## Deferred

- S2 priority scoring/status tuning beyond the deterministic baseline.
- S3 result-page actionable dashboard using the new section contract.
- S4 curated email/PDF refresh using the same model sections.
- S5 cross-surface QA checks for the new action sections.
- S6 monthly delta and macro/writeback upsell fields; deliberately kept out of
  this required v1 contract.

Parked hardening: none.

## Verification

- Python compile check for the deflection report producer, focused report tests,
  and frontend contract-doc tests -- passed.
- `python -m pytest tests/test_content_ops_deflection_report.py tests/test_deflection_snapshot_report_drift.py -q` -- 99 passed.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_snapshot_example_matches_producer_shape tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii tests/test_content_ops_deflection_report.py tests/test_deflection_snapshot_report_drift.py -q` -- 103 passed.
- Extracted pipeline CI enrollment audit -- passed; 187 matching tests enrolled.
- Full extracted pipeline check bundle -- passed; reasoning core 295 passed; extracted content 4733 passed / 15 skipped.
- `python scripts/sync_pr_plan.py plans/PR-Deflection-Action-Section-Contract.md --check` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 379 |
| `docs/frontend/content_ops_faq_report_contract.md` | 23 |
| `extracted_content_pipeline/deflection_report_access.py` | 1 |
| `extracted_content_pipeline/faq_deflection_report.py` | 603 |
| `plans/INDEX.md` | 1 |
| `plans/PR-Deflection-Action-Section-Contract.md` | 164 |
| `plans/archive/PR-Deflection-Signal-Spike.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 576 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 8 |
| **Total** | **1755** |
