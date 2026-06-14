# PR-Deflection-Exact-Repeat-Join

## Why this slice exists

Issue #1504 shows the repeat-question subclustering path says it groups
questions at exact Jaccard >= 1/3, but the shipped MinHash banding only
nominates a small fraction of pairs at that bar and is sensitive to row order.
Issue #1481 is the buyer-visible symptom: the deflection snapshot can present
category membership as repeat-question volume when the generator does not
measure question-level repetition tightly enough.

This vertical slice closes the lexical half of that gap. It makes the existing
documented lexical rule literal inside the real FAQ generation path by replacing
the approximate LSH candidate nomination with a deterministic prefix-filtered
exact join. The semantic embedding booster discussed in #1504 remains separate
because it needs a host-injected model port, calibration, and auditability
decisions.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Replace probabilistic MinHash-LSH nomination inside degraded topic buckets
   with an exact lexical candidate join that verifies the same Jaccard >= 1/3
   repeat-question rule already documented by the report methodology.
2. Preserve existing exact-duplicate behavior, empty-gist singleton behavior,
   and deterministic ordering for rendered FAQ items.
3. Add focused regression coverage for order-insensitive clustering and
   borderline lexical-overlap pairs that the old LSH shape could miss.
4. Keep embeddings, model dependencies, and live CFPB re-baselining out of this
   PR.

### Review Contract

- Acceptance criteria:
  - [ ] Reworded question gists whose exact token-set Jaccard is at least 1/3
        merge regardless of whether they share a MinHash band.
  - [ ] The same rows in a different input order produce the same repeat
        clusters and ticket counts.
  - [ ] Rows below the Jaccard threshold stay separate and continue to be
        excluded as non-repeat tickets when singleton-only.
  - [ ] Empty-gist rows never merge.
  - [ ] The implementation remains deterministic and bounded by prefix-filtered
        candidate generation rather than a naive all-pairs scan.
- Affected surfaces: extracted FAQ Markdown generation and its tests.
- Risk areas: repeat-count movement, performance on large degraded buckets,
  deterministic ordering, and over-merging unrelated tickets.
- Reviewer rules triggered: R1, R2, R7, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Exact-Repeat-Join.md`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

For each degraded topic bucket, derive the same question gist token sets used
today. Exact duplicate token sets still union immediately. For non-identical
gists, index each row by a rare-token-first prefix whose length follows the
standard prefix-filter shape for thresholded Jaccard matching. A row only
compares against earlier rows that share at least one indexed prefix token and
pass the threshold length-ratio filter; every candidate edge is still verified
with exact Jaccard before union.

The rendered cluster ordering stays root-index based, so the first occurrence
continues to anchor deterministic output. This removes the LSH recall gap
without adding embeddings, external services, or nondeterministic model calls.

## Intentional

- This PR keeps the lexical Jaccard rule. It does not try to catch same-meaning
  questions with near-zero token overlap; that is the embedding booster
  discussion in #1504 and needs a separate model-port slice.
- This PR accepts that published repeat counts can move. That movement is the
  point: the delivered rule should match the documented lexical rule.
- This PR avoids a naive all-pairs scan. The exact join may do more comparisons
  than LSH, but only after prefix and length-ratio filtering.

## Deferred

- Embedding booster with mutual-nearest-neighbor plus margin/floor calibration
  from #1504.
- Live CFPB artifact re-baseline after the lexical join lands.
- Issue #1481 copy/metric alignment for Support Tax wording once the
  question-level counts are on the corrected lexical foundation.
- Issue #1518 status vocabulary bug; useful but not part of repeat-question
  clustering.

Parked hardening: `Customer-wording FAQ headings can publish raw PII` and
`Safe-vocabulary representative label collisions render duplicate FAQ headings`
remain parked because they touch the same renderer but do not block this exact
repeat-join slice.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py -q - 241 passed.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh - 4053 passed, 10 skipped, 1 warning.
- Pending before push: local PR review wrapper.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 105 |
| `plans/PR-Deflection-Exact-Repeat-Join.md` | 114 |
| `tests/test_extracted_ticket_faq_markdown.py` | 99 |
| **Total** | **318** |
