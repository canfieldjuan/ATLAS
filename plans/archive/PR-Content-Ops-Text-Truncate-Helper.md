# PR-Content-Ops-Text-Truncate-Helper

## Why this slice exists

Issue #1290 tracks repeated ellipsis truncation logic in the deterministic card
generators. This slice centralizes the four private copies and adds CI-enrolled
tests for bounded output, suffix reservation, and current card behavior.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Production hardening

1. Add a package-owned shared ellipsis truncation helper.
2. Migrate social posts, ad copy, quote cards, and stat cards to it.
3. Add CI-enrolled helper/generator tests; leave non-card contracts untouched.

### Review Contract

- Acceptance criteria:
  - [ ] The helper reserves suffix length, including custom suffixes and tiny
        limits.
  - [ ] The four card generators no longer carry private `_truncate` copies.
  - [ ] Card-generator truncation is preserved for all reachable configs: the
        first three generators still compact whitespace, and stat-card still
        preserves spacing plus the small-limit `text[:limit]` contract.
  - [ ] Intentional change: social/ad/quote at `max_chars <= 2` now return a
        bounded `"." * limit` / `""` instead of the old `"..."` overshoot.
  - [ ] New tests are enrolled in `scripts/run_extracted_pipeline_checks.sh`.
- Affected surfaces: deterministic card generators and package tests.
- Risk areas: generated copy drift, CI enrollment, manifest ownership.
- Reviewer rules triggered: R1, R2, R10, R12, R14.

### Files touched

- `extracted_content_pipeline/ad_copy_generation.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/quote_card_generation.py`
- `extracted_content_pipeline/social_post_generation.py`
- `extracted_content_pipeline/stat_card_generation.py`
- `extracted_content_pipeline/text_truncate.py`
- `plans/PR-Content-Ops-Text-Truncate-Helper.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_content_text_truncate.py`

## Mechanism

The helper normalizes decoded text, reserves `len(suffix)`, then appends the
suffix. The default compacts whitespace for social/ad/quote; keyword options
preserve stat-card spacing and small-limit behavior. No service API, config
field, or generated result shape changes.

## Intentional

- The helper keeps a whitespace-compaction option because stat cards differ.
- Word-boundary, trim-only, report-builder, and campaign-sequence truncation
  paths stay out of this slice.
- Generated field limits and service configuration defaults do not change.

## Deferred

- `ticket_faq_markdown.py` overshoots stay deferred to the deflection lane.
- Campaign sequence `-0` sites need call-site bound assessment first.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_content_text_truncate.py tests/test_extracted_quote_card_generation.py tests/test_extracted_stat_card_generation.py -q` - 31 passed.
- `scripts/validate_extracted_content_pipeline.sh` via bash - Passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - Passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - Passed.
- `scripts/check_ascii_python.sh` via bash - Passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ad_copy_generation.py` | 15 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/quote_card_generation.py` | 14 |
| `extracted_content_pipeline/social_post_generation.py` | 15 |
| `extracted_content_pipeline/stat_card_generation.py` | 42 |
| `extracted_content_pipeline/text_truncate.py` | 50 |
| `plans/PR-Content-Ops-Text-Truncate-Helper.md` | 88 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_extracted_content_text_truncate.py` | 163 |
| **Total** | **391** |
