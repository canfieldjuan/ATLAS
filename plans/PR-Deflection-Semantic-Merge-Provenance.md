# PR-Deflection-Semantic-Merge-Provenance

## Why this slice exists

#1547 completed the mxbai validation arc by proving the accepted semantic merge
edges are coherent on the live CFPB sample. The remaining #1504 follow-up is
buyer-facing semantic merge provenance: if the paid report uses an embedding
edge to count two low-lexical-overlap tickets as one repeated question, the
unlocked report should say that the grouping was supported by semantic matching
and show inspectable, bounded evidence.

This is the next vertical slice before any production flag decision because it
keeps the booster auditable when enabled. It does not flip
`ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED`, tune thresholds, or add a new
model path.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Vertical slice

1. Attach sanitized semantic-merge provenance from accepted embedding booster
   edges to the FAQ item that contains both originating row ids.
2. Add compact paid-report Markdown that summarizes semantic edges by question
   with source ids, cosine, margins, and token-overlap context.
3. Add summary counts so CLI/result consumers can tell whether semantic matching
   contributed without parsing Markdown.
4. Keep free snapshot projection locked: no semantic merge source ids, snippets,
   or paid provenance leak into the pre-payment snapshot.
5. Leave the route flag default, thresholds, and host model wiring unchanged.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Semantic-Merge-Provenance.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] Only edges accepted by the existing embedding booster are recorded; no
        new clustering heuristic is introduced.
  - [ ] Provenance is attached only to rendered FAQ items whose originating row
        ids contain both sides of the accepted edge, so duplicate source-id
        pairs across multiple questions cannot misattribute provenance.
  - [ ] Paid report output exposes compact semantic matching evidence without
        requiring raw JSON artifacts.
  - [ ] Free snapshot output continues to omit source ids, snippets, and
        semantic merge details.
  - [ ] Production defaults, thresholds, route flags, and host mxbai wiring stay
        unchanged.
- Affected surfaces: extracted FAQ item metadata, deflection paid-report summary
  and Markdown, free snapshot projection tests.
- Risk areas: misleading provenance, private text leakage, accidental behavior
  change in no-embedding callers, schema drift for decoded input.
- Reviewer rules triggered: R1, R2, R5, R10, R11, R12, R14.

## Mechanism

Keep the existing recorder callback as the source of truth. During FAQ build,
assign each normalized evidence row an internal semantic row id, collect accepted
semantic edges locally, and, after selected groups become FAQ items, attach each
edge to the item whose selected rows contain both edge row ids. The attached
item payload is sanitized to source ids plus scores/margins and token Jaccard;
it must not carry the raw left/right text snippets or internal row ids from the
recorder.

The report summary derives aggregate semantic-match counts from item metadata.
The paid Markdown gets a small provenance section after ranked opportunities.
The free snapshot projection remains unchanged except for a regression test
proving the semantic metadata is absent from its encoded payload.

## Intentional

- This PR adds provenance only for semantic edges that survive into rendered FAQ
  items. Edges accepted inside an overflow group are still item-level provenance
  for that overflow item; edges for excluded singletons do not render because
  they did not contribute to a paid repeated-question item.
- Raw `left_text` and `right_text` stay internal to validation callers from
  #1547. The buyer-facing report uses source ids and numeric diagnostics so the
  same path is safer for private support-ticket uploads.
- No production flag flip. The report surface can describe semantic matching
  when a caller enables the existing flag, but this slice does not change when
  the booster runs.

## Deferred

- Operator decision for enabling
  `ATLAS_CONTENT_OPS_FAQ_EMBEDDING_BOOSTER_ENABLED`.
- Any threshold/margin tuning if future private-ticket validation finds bad
  semantic matches.
- UI-specific styling for semantic provenance in atlas-portfolio or
  atlas-intel-ui; this PR lands the extracted report contract first.

Parked hardening: none.

## Verification

- Review-fix regression and semantic provenance spot checks -- 3 passed.
- Focused FAQ Markdown/report test files -- 295 passed.
- Extracted pipeline checks -- 4136 passed, 10 skipped, 1 warning.
- Local PR review bundle with planned PR body -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 113 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 75 |
| `plans/PR-Deflection-Semantic-Merge-Provenance.md` | 115 |
| `tests/test_content_ops_deflection_report.py` | 45 |
| `tests/test_extracted_ticket_faq_markdown.py` | 47 |
| **Total** | **395** |
