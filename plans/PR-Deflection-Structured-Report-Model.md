# PR-Deflection-Structured-Report-Model

## Why this slice exists

Epic #1588's next step is to stop treating the paid deflection report as one
Markdown blob and start moving toward a structured `deflection.v1` report model
that web, PDF, email, Markdown, and export surfaces can render independently.

The root cause is architectural: the current paid report computes report facts
and renders Markdown in the same helper chain, so every downstream surface either
consumes the full Markdown blob or parses it back into shape-specific pieces.
#1594 intentionally shipped a Markdown curator only as an interim bridge; this
slice fixes the root at the first safe upstream point by introducing structured
section records behind the existing Markdown output. It does not yet redesign
the buyer-facing report shape.

Per the #1588 implementation note, this slice first pins the current paid-report
Markdown with a golden snapshot so the strangler refactor has a behavior-stable
guard before any section-model migration proceeds.

This PR is over the 400 LOC soft cap because the behavior-preserving proof is a
literal golden Markdown snapshot, the documented deflection artifact example now
has to carry the additive `report_model` contract, and the slice also folds in
required #1594 plan-archive housekeeping. Splitting the snapshot from the
refactor would remove the guard the epic explicitly asked to add first.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add `deflection.v1` report-section records in
   `extracted_content_pipeline.faq_deflection_report` and render current
   Markdown from those records.
2. Expose the structured model through `DeflectionReportArtifact.as_dict()` so
   host/portfolio surfaces can consume the model in later slices without
   parsing Markdown.
3. Keep current paid-report Markdown byte-stable.
4. Document the additive `report_model` artifact contract and keep the example
   JSON producer-bound.
5. Archive the now-merged #1594 plan doc and refresh `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] The representative paid report Markdown matches the committed golden
        snapshot byte-for-byte after the section-model refactor.
  - [ ] The structured model reports schema version `deflection.v1`, ordered
        section IDs/priorities, surfaces, default limits, and meaningful data
        for support-tax, SEO targets, ranked questions, diagnostics, question
        details, and export-only complete evidence.
  - [ ] Markdown rendering is driven from the structured sections, not a
        separate imperative list that can drift from the model.
  - [ ] The documented deflection example compares the full nested
        `report_model` to producer output, not only the top-level key set.
  - [ ] Support-tax structured `data` is pinned against the rendered section
        Markdown for both complete-window and unknown-window cost branches.
  - [ ] Unknown future sections are skippable by consumers via normal section
        metadata; this slice does not require any web/PDF consumer change.
  - [ ] Existing snapshot and evidence-export behavior remains unchanged.
- Affected surfaces: extracted package report artifact contract, frontend
  contract docs, paid report Markdown, tests, plan archive/index.
- Risk areas: backward compatibility, producer/consumer drift, report
  readability regressions, extracted package standalone discipline.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Structured-Report-Model.md`
- `plans/archive/PR-Deflection-Curated-PDF-TOC.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The extracted report module gains lightweight frozen dataclasses:

- `DeflectionReportSection` -- `id`, `title`, `priority`, `surfaces`,
  `default_limit`, `data`, and the current Markdown lines for the interim
  Markdown renderer.
- `DeflectionStructuredReport` -- schema version, title, summary, and ordered
  sections.

`build_deflection_report_artifact(...)` builds the model once, renders Markdown
from `model.sections`, and stores both on the artifact. The existing section
helpers still generate the current Markdown lines, but the call site now wraps
each helper's output with stable section metadata and structured data. This is
the strangler step: current Markdown stays stable while future web/PDF/export
renderers can move onto section data one surface at a time.

The frontend contract doc now names `report_model` and the `deflection.v1`
section schema. The contract fixture test compares the full nested
`report_model` to fresh producer output so docs cannot drift from producer
section IDs, priorities, surfaces, limits, or data.

## Intentional

- The model initially carries `markdown_lines` as an interim rendering bridge.
  That is deliberate: the root change is to centralize section identity/order
  and data now, without rewriting every surface in the same PR.
- This slice does not redesign the PDF or hosted result page. #1590-#1594
  already made those surfaces usable; this PR creates the model they can later
  consume directly.
- The complete evidence export remains the uncapped completeness surface.
  Structured report sections may reference export data, but this PR does not
  duplicate the full export rows into every rendered surface.

## Deferred

- Later #1588 slice: persist the structured report JSON in the paid-report store
  so old reports can be re-rendered without rerunning the pipeline.
- Later #1588 slice: move hosted web and PDF renderers to consume
  `DeflectionStructuredReport` directly, then remove the Markdown parsing/line
  bridge from the PDF curator.
- Later #1588 slice: make the Markdown renderer read directly from section
  `data` for support-tax/ranked/SEO sections so `data` and Markdown stop being
  dual computations. This PR adds guard assertions while the interim bridge is
  still in place.
- Later #1588 slice: introduce a section registry/config layer for hiding,
  reordering, or per-surface section selection without editing the renderer.

Parked hardening: none.

## Verification

- Pytest `tests/test_content_ops_deflection_report.py` and
  `tests/test_content_ops_faq_report_contract_docs.py` -- 54 passed.
- Focused review-fix pytest for the model/contract guards -- 6 passed.
- Compile check for `extracted_content_pipeline/faq_deflection_report.py`,
  `tests/test_content_ops_deflection_report.py`, and
  `tests/test_content_ops_faq_report_contract_docs.py` -- passed.
- Extracted validation via `scripts/validate_extracted_content_pipeline.sh` --
  passed.
- Reasoning import guard via
  `extracted/_shared/scripts/forbid_atlas_reasoning_imports.py` -- passed.
- Standalone audit via `scripts/audit_extracted_standalone.py` -- passed.
- ASCII Python audit via `scripts/check_ascii_python.sh` -- passed.
- Contract-doc focused pytest
  `tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape`
  -- 1 passed.
- Full extracted CI check via `scripts/run_extracted_pipeline_checks.sh` --
  4309 passed, 10 skipped.
- Pending before push:
  - `python scripts/sync_pr_plan.py plans/PR-Deflection-Structured-Report-Model.md --check`
  - push-wrapper local review

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 233 |
| `docs/frontend/content_ops_faq_report_contract.md` | 45 |
| `extracted_content_pipeline/faq_deflection_report.py` | 343 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Structured-Report-Model.md` | 163 |
| `plans/archive/PR-Deflection-Curated-PDF-TOC.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 325 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 6 |
| **Total** | **1118** |
