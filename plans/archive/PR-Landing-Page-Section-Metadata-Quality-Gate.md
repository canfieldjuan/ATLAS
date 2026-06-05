# PR-Landing-Page-Section-Metadata-Quality-Gate

## Why this slice exists

PR #747 centralized the landing-page section-kind contract for review/export
readiness. The landing-page quality pack still ignores that metadata entirely,
so invalid section kinds and hidden answer summaries only show up later in
review readiness.

This slice moves the softest checks into the deterministic quality pack as
warnings. The goal is to guide generation and repair without blocking older
drafts or turning AEO/GEO metadata into a hard persistence requirement.

## Scope (this PR)

Ownership lane: content-ops/landing-page-section-metadata-quality-gate

1. Move the section-kind contract to `extracted_quality_gate`, the lower-level
   package that owns landing-page quality validation.
2. Update content-pipeline readiness scoring and prompt-regression tests to
   import the contract from `extracted_quality_gate`.
3. Add warning-only quality-pack checks for missing/invalid section kind
   metadata.
4. Add warning-only checks that question-shaped sections keep
   `answer_summary` visible at the start of `body_markdown`.
5. Cover the quality-pack behavior with focused tests.

### Files touched

- `plans/PR-Landing-Page-Section-Metadata-Quality-Gate.md`
- `extracted_quality_gate/landing_page_section_contract.py`
- `extracted_quality_gate/landing_page_pack.py`
- `extracted_content_pipeline/landing_page_export.py`
- `tests/test_landing_page_section_contract.py`
- `tests/test_extracted_campaign_skill_registry.py`
- `tests/test_extracted_landing_page_generation.py`
- `tests/test_extracted_quality_gate_landing_page_pack.py`

## Mechanism

`extracted_quality_gate.landing_page_section_contract` owns the canonical
section-kind sets and normalizer. The content-pipeline readiness helper imports
that lower-level contract, preserving the dependency direction that already
exists between Content Ops and quality-gate packages.

The quality pack adds warnings only:

- `section_missing_kind`
- `section_invalid_kind`
- `section_missing_answer_summary`
- `section_answer_summary_not_visible`

Warnings still reduce score, but they do not create blockers. With default
thresholds, one or two metadata warnings still let the draft persist while
making the issue visible to the repair loop and review output.

## Intentional

- No blockers for section metadata. This protects older drafts and avoids
  making prompt-style metadata a hard save requirement.
- No prompt copy change.
- No public API shape change.
- No import from `extracted_quality_gate` back into `extracted_content_pipeline`.

## Deferred

- `PR-Landing-Page-Publish-Structured-Data` can map FAQ/objection sections into
  public JSON-LD once generated landing pages have a renderer.

## Verification

- `pytest tests/test_extracted_landing_page_generation.py tests/test_extracted_quality_gate_landing_page_pack.py tests/test_landing_page_section_contract.py tests/test_extracted_landing_page_export.py tests/test_extracted_campaign_skill_registry.py -q` - 90 passed.
- `scripts/run_extracted_pipeline_checks.sh` - 1623 passed.
- Python compile command over the touched quality-gate, content-pipeline, and
  test Python files - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~85 |
| Contract move | ~0 |
| Quality-pack warnings | ~65 |
| Import updates | ~15 |
| Tests | ~80 |
| **Total** | **~230** |
