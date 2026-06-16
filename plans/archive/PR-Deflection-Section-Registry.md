# PR-Deflection-Section-Registry

## Why this slice exists

Epic #1588 slice 5 calls for a report-section registry so the paid deflection
report can add, remove, hide, or reorder sections as real customers teach us
what they need. #1596 introduced `deflection.v1` section records, but the
section metadata still lives inline at each construction call: section IDs,
titles, priorities, surfaces, and default limits are repeated in the builder
rather than owned by one registry.

The root cause is still upstream of the renderer: adding a section requires
editing imperative construction metadata instead of registering the section's
contract once. This slice moves the metadata to a single extracted-package
registry and makes section builders consume it. That is the smallest safe
upstream fix: current Markdown remains byte-stable, while future web/PDF/export
renderers can trust one registry for ordering, visibility, and required data.

This PR is slightly over the 400 LOC soft cap because the additive public
contract requires regenerating the producer-bound example JSON, documenting the
new section field, and archiving the merged #1596 plan in the same slice. The
code/test change itself stays narrow.

## Scope (this PR)

Ownership lane: content-ops/deflection-launch-readiness
Slice phase: Vertical slice

1. Add a `DeflectionReportSectionDefinition` registry in
   `extracted_content_pipeline.faq_deflection_report` covering each current
   paid report section.
2. Make `_report_section(...)` look up section title, priority, surfaces,
   default limit, and required data from that registry instead of accepting
   scattered metadata at call sites.
3. Add `required_data` to each emitted `DeflectionReportSection` so consumers
   can tell which top-level data keys a section promises.
4. Fail closed if a section builder omits a registry-required data key.
5. Keep Markdown byte-stable and update the frontend contract/example to
   document the additive `required_data` field.
6. Archive the now-merged #1596 plan doc and refresh `plans/INDEX.md`.

### Review Contract

- Acceptance criteria:
  - [ ] Registry entries define `id`, `title`, `priority`, `surfaces`,
        `default_limit`, and `required_data` for every current report section.
  - [ ] Built report sections take their metadata from the registry; call sites
        no longer pass duplicated title/priority/surface/default-limit values.
  - [ ] Section priority and ID uniqueness are pinned by tests.
  - [ ] Missing required data fails closed during section construction, with a
        focused negative test.
  - [ ] The representative paid report Markdown remains byte-for-byte identical
        to the #1596 golden snapshot.
  - [ ] The documented artifact example remains producer-bound and includes
        `required_data` for each section.
- Affected surfaces: extracted package report model, frontend contract docs,
  paid report example JSON, deflection report tests, plan archive/index.
- Risk areas: report contract drift, byte-stability regressions, fail-open
  missing data, extracted package standalone discipline.
- Reviewer rules triggered: R1, R2, R5, R10, R12, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_report_contract.md`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/INDEX.md`
- `plans/PR-Deflection-Section-Registry.md`
- `plans/archive/PR-Deflection-Structured-Report-Model.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The module gains a frozen `DeflectionReportSectionDefinition` dataclass and a
single `DEFLECTION_REPORT_SECTION_REGISTRY` keyed by section ID. Each definition
contains the metadata that #1588 named as the registry contract:

- `id`
- `title`
- `priority`
- `surfaces`
- `default_limit`
- `required_data`

`_report_section(...)` now receives only `section_id`, `data`, and
`markdown_lines`. It looks up the definition, validates that every required
top-level `data` key is present, and emits a `DeflectionReportSection` carrying
the registry metadata plus the interim Markdown bridge.

The renderer still sorts by section priority and skips sections without the
`markdown` surface, so current Markdown output should remain identical. The new
registry is additive contract shape for future renderers; this PR does not move
web/PDF renderers onto new section-specific code.

## Intentional

- `required_data` validates top-level section data keys only. Nested schemas
  stay owned by the existing section-specific tests; this keeps the registry
  small enough to be useful for renderer selection without becoming a full JSON
  schema system.
- The registry lives in `extracted_content_pipeline`, not the portfolio or PDF
  host, because the extracted package is the producer of the `deflection.v1`
  contract and must stay deterministic/local.
- This slice keeps `markdown_lines` as the current strangler bridge. Removing
  that bridge waits until each consumer renders directly from `data`.

## Deferred

- Later #1588 slice: persist structured report JSON and render old reports from
  stored model data without rerunning the pipeline.
- Later #1588 slice: move hosted web and PDF surfaces to consume registry-backed
  section `data` directly.
- Later #1588 slice: renderer-specific section visibility policy, such as
  tenant/plan-level hiding, can layer on top of the registry once customers
  teach us which sections matter most.

Parked hardening: none.

## Verification

- Focused registry/contract pytest:
  python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_model_keeps_current_markdown_golden_snapshot tests/test_content_ops_deflection_report.py::test_deflection_report_artifact_exposes_structured_model_sections tests/test_content_ops_deflection_report.py::test_deflection_report_section_registry_drives_section_metadata tests/test_content_ops_deflection_report.py::test_deflection_report_section_registry_fails_closed_on_missing_data tests/test_content_ops_deflection_report.py::test_deflection_report_section_registry_rejects_unknown_section_id tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_report_contract_links_example tests/test_content_ops_faq_report_contract_docs.py::test_content_ops_faq_deflection_example_matches_producer_shape -q
  -- 7 passed.
- Touched test files:
  python -m pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py -q
  -- 57 passed.
- Compile check:
  python -m compileall extracted_content_pipeline/faq_deflection_report.py tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_report_contract_docs.py
  -- passed.
- Extracted package gauntlet:
  bash scripts/validate_extracted_content_pipeline.sh -- passed;
  python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  -- passed; python scripts/audit_extracted_standalone.py --fail-on-debt --
  passed; bash scripts/check_ascii_python.sh -- passed.
- Full extracted CI check: bash scripts/run_extracted_pipeline_checks.sh --
  4312 passed, 10 skipped.
- Pending before push:
  - python scripts/sync_pr_plan.py plans/PR-Deflection-Section-Registry.md --check
  - push-wrapper local review

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 30 |
| `docs/frontend/content_ops_faq_report_contract.md` | 3 |
| `extracted_content_pipeline/faq_deflection_report.py` | 170 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Deflection-Section-Registry.md` | 154 |
| `plans/archive/PR-Deflection-Structured-Report-Model.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 79 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 1 |
| **Total** | **440** |
