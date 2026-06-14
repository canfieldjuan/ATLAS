# PR-Deflection-Mxbai-Merge-Spotcheck

## Why this slice exists

#1544 proved the pinned local mxbai path runs on live CFPB data and has a
bounded effect, but the review left one launch-readiness question open before
any route flag flip: the 18 semantic merges need a spot-check. The committed
#1544 artifact records counts and probe batches, not the exact accepted semantic
edges, so an operator cannot inspect whether those merges are semantically
correct without rerunning local raw output.

This slice adds reviewable semantic-edge telemetry to the existing CFPB compare
harness and commits a sanitized live spot-check artifact. It does not enable the
production flag, change thresholds, or change FAQ output.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Functional validation

1. Add an optional recorder callback to the existing embedding-booster pair
   selection so accepted mutual-nearest-neighbor edges can be inspected.
2. Surface recorded edges in the CFPB compare smoke payload under the embedding
   comparison, with source ids, question gist text, cosine score, runner-up
   margins, and token Jaccard.
3. Add focused deterministic tests proving the recorder captures accepted edges,
   omits failed embedding paths, and does not change FAQ output.
4. Run the live CFPB fees sample again and commit a sanitized spot-check summary
   for the accepted semantic merge edges.
5. Leave raw CFPB JSONL rows and full generated Markdown under `tmp/` only.

### Files touched

- `docs/extraction/validation/deflection_mxbai_merge_spotcheck_2026-06-14.md`
- `docs/extraction/validation/fixtures/deflection_mxbai_merge_spotcheck_20260614/summary.json`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Mxbai-Merge-Spotcheck.md`
- `scripts/smoke_content_ops_cfpb_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] Semantic merge recorder data comes from the same accepted MNN pairs that
        are unioned by the booster; no separate clustering heuristic.
  - [ ] The CFPB compare payload exposes enough edge data to spot-check merge
        quality without committing raw complaint narratives.
  - [ ] Failure/invalid embedding paths do not report accepted semantic edges.
  - [ ] FAQ Markdown output, thresholds, route flags, and production defaults
        remain unchanged.
- Affected surfaces: extracted FAQ private clustering helper, CFPB compare
  smoke harness, focused tests, validation docs/artifacts.
- Risk areas: misleading provenance, accidental raw-data leak, API drift from
  optional recorder plumbing.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

## Mechanism

Thread an optional `embedding_merge_recorder` callback through
`build_ticket_faq_markdown` into the private question sub-clustering path. The
recorder is invoked only after the existing MNN floor and two-sided margin gates
accept a pair, immediately before the existing union operation. The callback
receives row indexes, source ids, question gist text, cosine score, each side's
runner-up score, each side's margin, and token Jaccard.

The CFPB compare smoke passes a local recorder only for the boosted run and adds
the recorded edges to `embedding_comparison`. The committed validation artifact
summarizes those edges; raw source rows and full Markdown stay outside git.

## Intentional

- No route flag flip. This slice creates the spot-check evidence needed before
  an operator decides whether the 1.8% CFPB lift is worth enabling.
- No threshold or margin tuning. If the spot-check exposes bad merges, that
  becomes a separate scoped follow-up.
- The recorder is optional and inert by default, so existing callers keep the
  same output shape unless they explicitly request edge telemetry.
- The committed artifact may include short question gists and source ids because
  those are the inspection surface; full complaint narratives and raw JSONL rows
  are not committed.
- The recorder emits customer-text snippets to its explicit caller. It is for
  validation/inspection surfaces only and must not be wired into private-ticket
  logs or committed artifacts without redaction.

## Deferred

- Production enablement decision for `ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED`.
- Threshold/margin changes if the spot-check finds incorrect semantic merges.
- Buyer-facing semantic merge provenance in the final report UI.

Parked hardening: none.

## Verification

- Focused FAQ Markdown and CFPB smoke test files -- 260 passed.
- Live CFPB compare command with semantic edge telemetry -- passed, exit 0;
  recorded 9 accepted semantic merge edges matching the -18 non-repeat ticket
  delta.
- Extracted pipeline checks -- 4118 passed, 10 skipped, 1 warning.
- Local PR review bundle with planned PR body -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/extraction/validation/deflection_mxbai_merge_spotcheck_2026-06-14.md` | 60 |
| `docs/extraction/validation/fixtures/deflection_mxbai_merge_spotcheck_20260614/summary.json` | 37 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 35 |
| `plans/PR-Deflection-Mxbai-Merge-Spotcheck.md` | 114 |
| `scripts/smoke_content_ops_cfpb_faq_markdown.py` | 35 |
| `tests/test_extracted_ticket_faq_markdown.py` | 79 |
| `tests/test_smoke_content_ops_cfpb_faq_markdown.py` | 14 |
| **Total** | **374** |
