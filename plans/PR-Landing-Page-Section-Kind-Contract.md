# PR-Landing-Page-Section-Kind-Contract

## Why this slice exists

PR #746 made landing-page readiness score `sections.metadata.kind`, but the
valid section-kind enum now lives as a manual mirror in the readiness helper and
the bundled prompt. That is the same drift pattern we already corrected for
landing-page repair attempts.

This slice creates one small source of truth for landing-page section kind
values and makes the scorer and prompt regression tests use it.

## Scope (this PR)

Ownership lane: content-ops/landing-page-section-kind-contract

1. Add a canonical landing-page section-kind contract module.
2. Replace readiness-helper local section-kind constants with the shared
   contract.
3. Add tests that validate kind normalization and prompt enum coverage.

### Files touched

- `plans/PR-Landing-Page-Section-Kind-Contract.md`
- `extracted_content_pipeline/landing_page_section_contract.py`
- `extracted_content_pipeline/landing_page_export.py`
- `tests/test_landing_page_section_contract.py`
- `tests/test_extracted_campaign_skill_registry.py`

## Mechanism

`landing_page_section_contract.py` owns:

- `LANDING_PAGE_SECTION_KINDS`
- `LANDING_PAGE_QUESTION_SECTION_KINDS`
- `LANDING_PAGE_PROBLEM_SECTION_KINDS`
- `LANDING_PAGE_SOLUTION_SECTION_KINDS`
- `LANDING_PAGE_OBJECTION_SECTION_KINDS`
- `normalize_landing_page_section_kind(...)`

The export readiness helper imports those constants instead of carrying a local
copy. The prompt remains static markdown, but the packaged-prompt regression
test now loops over the canonical kind list and fails if any canonical value is
missing from the prompt's section-kind instruction.

## Intentional

- No prompt copy change. This slice verifies the existing prompt against the
  backend contract.
- No API output change.
- No quality-gate blocker. This only removes a manual mirror.

## Deferred

- `PR-Landing-Page-Section-Metadata-Quality-Gate` can decide whether invalid
  section kinds should become quality-pack warnings.

## Verification

- `pytest tests/test_landing_page_section_contract.py tests/test_extracted_campaign_skill_registry.py tests/test_extracted_landing_page_export.py -q` - 30 passed.
- Python compile command over `extracted_content_pipeline/landing_page_section_contract.py`,
  `extracted_content_pipeline/landing_page_export.py`,
  `tests/test_landing_page_section_contract.py`, and
  `tests/test_extracted_campaign_skill_registry.py` - passed.
- `git diff --check` - passed.
- `bash scripts/local_pr_review.sh origin/main` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~75 |
| Contract module | ~52 |
| Export import swap | ~55 |
| Tests | ~40 |
| **Total** | **~220** |
