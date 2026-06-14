# PR-Deflection-Embedding-Booster-Core

## Why this slice exists

#1504 closed the lexical mismatch with #1531: the repeat-question rule now
matches the documented exact Jaccard floor. The remaining gap is semantic
recall: two tickets can ask the same thing with near-zero shared tokens, so the
exact lexical floor still fragments the reworded long tail.

#1515 approved the hybrid direction: keep the exact lexical floor, then add an
embedding recall booster through a host-injected port so the extracted package
stays model-free. The later #1504 calibration note replaced a flat cosine
threshold with mutual nearest neighbor, a margin, and a loose cosine floor.
This slice lands that deterministic extracted core only.

The first review found three safety holes in that core: non-finite vectors
could return `nan` and bypass both gates, margin was enforced from only one side
of a mutual-nearest pair, and a non-singleton lexical component could amplify
embedding-only unions into a hub-and-spoke collapse. Those are core acceptance
criteria for this slice, so the fixes and regressions stay here even though the
review update pushes the PR above the soft 400 LOC target.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Add a model-free embedding port contract and cosine helper owned by
   `extracted_content_pipeline`.
2. Thread an optional embedding port through the FAQ Markdown config, service,
   builder, and sub-clustering path.
3. Add a deterministic embedding booster that only considers pairs left
   unmerged by the lexical floor, limits semantic edges to singleton lexical
   components, and merges mutual-nearest-neighbor pairs only when both rows
   clear the margin plus loose-floor thresholds.
4. Prove default behavior is unchanged when no port is injected.
5. Prove a deterministic stub port merges a same-meaning no-token-overlap pair
   through the full `build_ticket_faq_markdown` path.
6. Prove non-finite vectors, one-sided-margin ambiguity, and lexical-hub
   amplification fail closed.

### Review Contract

- Acceptance criteria:
  - [ ] No embedding port keeps the existing lexical floor behavior and output
        shape unchanged.
  - [ ] A deterministic stub embedding port can merge two no-token-overlap
        rewordings in the full FAQ Markdown builder.
  - [ ] The booster only runs after lexical clustering and cannot merge pairs
        unless they are singleton lexical components, mutual nearest
        neighbors, both meet the configured margin, and meet the loose cosine
        floor.
  - [ ] Empty, malformed, or non-finite embedding vectors fail closed by
        skipping the semantic merge instead of raising or merging.
  - [ ] Non-singleton lexical components cannot create hub-and-spoke semantic
        fan-out through the embedding booster.
  - [ ] The extracted package remains model-free: no `atlas_brain`,
        `sentence_transformers`, network, DB, or host imports.
- Affected surfaces: FAQ clustering internals, embedding port contract,
  focused tests, and package manifest.
- Risk areas: accidental default output drift, opaque semantic over-merging,
  malformed injected vectors, and extracted package boundary violations.
- Reviewer rules triggered: R1, R2, R9, R10, R12, R14.

### Files touched

- `extracted_content_pipeline/embedding_port.py`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Embedding-Booster-Core.md`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

The lexical floor remains the first pass and keeps the published floor
guarantee. The optional port receives one text per singleton lexical component,
returns vectors, and the core computes exact finite cosine values in process.
Each row chooses its best not-yet-merged singleton neighbor; only reciprocal best
matches above the loose floor and both rows' margins merge. Iteration order and
tie-breaking are stable, non-finite vectors return no score, and no host model
is loaded by default.

## Intentional

- This PR does not import or configure the live mxbai model. The extracted core
  must stay model-free; host adapter wiring belongs in the next slice.
- This PR uses mutual-nearest-neighbor plus margin and loose floor, not a flat
  cosine threshold, because #1504 measured that flat thresholds do not transfer
  across real corpora.
- This PR keeps the existing lexical floor as the default and first pass. The
  booster can only add recall when a caller explicitly injects a port.
- This PR limits embedding edges to singleton lexical components. That is a
  conservative over-merge guard: it preserves the no-token-overlap recall path
  while deferring multi-row lexical component expansion until a later slice can
  prove a safer cluster-level criterion.

## Deferred

- Host adapter and service wiring for the pinned/offline mxbai model.
- Live CFPB re-baseline after host wiring is available and operator-approved.
- #1518 status vocabulary fix remains in the deflection backlog, but it is a
  status-normalization lane item rather than this clustering slice.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile extracted_content_pipeline/embedding_port.py extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py::test_embedding_booster_merges_no_overlap_reworded_repeat_in_builder tests/test_extracted_ticket_faq_markdown.py::test_embedding_booster_skips_malformed_vectors_without_merging tests/test_extracted_ticket_faq_markdown.py::test_embedding_booster_skips_non_finite_vectors_without_merging tests/test_extracted_ticket_faq_markdown.py::test_embedding_booster_requires_both_mutual_neighbor_margins tests/test_extracted_ticket_faq_markdown.py::test_embedding_booster_ignores_non_singleton_lexical_components tests/test_extracted_ticket_faq_markdown.py::test_cosine_similarity_rejects_malformed_vectors tests/test_extracted_ticket_faq_markdown.py::test_ticket_faq_service_config_threads_embedding_port -q - 9 passed.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py -q - 250 passed.
- Command passed: bash scripts/validate_extracted_content_pipeline.sh.
- Command passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Command passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh - 4089 passed, 10 skipped, 1 warning.
- Pending before push: plan sync and local PR review.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/embedding_port.py` | 45 |
| `extracted_content_pipeline/manifest.json` | 3 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 106 |
| `plans/PR-Deflection-Embedding-Booster-Core.md` | 127 |
| `tests/test_extracted_ticket_faq_markdown.py` | 224 |
| **Total** | **505** |
