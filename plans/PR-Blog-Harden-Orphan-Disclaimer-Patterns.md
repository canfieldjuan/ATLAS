# PR-Blog-Harden-Orphan-Disclaimer-Patterns

## Why this slice exists

The blog generator strips a disclaimer paragraph that sits next to an
ungrounded/stripped blockquote (`_remove_unmatched_quote_lines` ->
`_looks_like_orphan_disclaimer` -> `_ORPHAN_DISCLAIMER_RES`). The same four
patterns are mirrored by the `seo-geo-aeo-blog-post` skill's audit so the
two stay in sync.

A corpus audit found those four patterns were too narrow. Posts shipped with
disclaimers admitting an off-topic keyword-match quote -- "copper" matched
copper-the-metal, audio cables, an ISP complaint, a financial-planning note --
in wording the patterns missed:

- *"While the quote references internet service providers rather than CRM
  software ..."*
- *"This quote does not directly reference Copper CRM ..."*
- *"... appears to reference audio equipment rather than the software product,
  indicating data noise in the corpus."*
- *"Two complaints ... lacked CRM-specific context."*
- *"Not every mention of 'Copper' and 'pricing' refers to the software
  product."*

Because the generator stripper and the audit share the pattern set, these
escaped **both** upstream stripping (so they shipped) and post-publish
detection (so the first audit pass missed them). This widens both.

## Scope (this PR)

1. Add five patterns to `_ORPHAN_DISCLAIMER_RES` covering the
   "references X rather than CRM/software", "does not (directly) reference",
   "data noise", "lacked <X>-specific context", and "not every mention of ...
   refers to" disclaimer shapes.
2. Add two regression tests: one asserting the new variants are swept with
   their stripped block, one false-positive guard that non-disclaimer prose
   containing "rather than" adjacent to a stripped block is preserved.
3. Sync the change into the `extracted_content_pipeline` mirror.

The skill-side audit patterns (`detectAcknowledgedMisattribution` in
`scripts/audit-published-posts.js`) are widened in tandem so the two stay in
sync; that change lives in the skill repo, not this PR.

### Files touched

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`
- `plans/PR-Blog-Harden-Orphan-Disclaimer-Patterns.md`

## Mechanism

The new patterns are appended to the existing tuple, so the
`_looks_like_orphan_disclaimer` guard (skip lines starting with `>`, `#`,
`-`, `*`, `+`, or an ordered-list marker) still applies -- the additions only
change *which* prose lines count as disclaimers, not the structural safety
rails. They fire only on a paragraph adjacent to an already-stripped block, so
their blast radius is bounded by the existing block-strip logic.

Pattern tightness was validated against the published corpus (79 posts): the
five additions match only the genuine disclaimers (the repaired
`real-cost-of-copper` spots and the one remaining `close-deep-dive`
occurrence) with zero false positives on legitimate prose. The
`references ... rather than ...` pattern is anchored to a trailing
`crm|software|the product|the platform`, and `data noise` is a specific
two-word phrase, so ordinary "X rather than Y" prose is not swept.

## Intentional

- **Patterns appended, not rewritten.** The original four stay verbatim; the
  additions are a clearly-commented second group, so the diff is auditable and
  the existing behavior is unchanged.
- **`does not (directly) reference` left fairly broad.** It is a strong
  disclaimer signal and only fires adjacent to a stripped block; the corpus
  validation showed no false positives.
- **Skill audit widened separately.** The generator (this PR) and the skill
  audit must carry the same patterns, but they live in different repos. This
  PR notes the dependency; the skill edit ships alongside.

## Deferred

- **`close-deep-dive` data repair.** The hardened audit now flags its one
  remaining off-topic financial-planning quote; repairing that post is a
  data-side follow-up (same surgical approach as `real-cost-of-copper`).

## Verification

- `python -m pytest tests/test_b2b_blog_post_generation_quote_gate.py -q` ->
  `83 passed` (was 81; +2 new tests).
- Corpus scan with the five candidate patterns over all 79 published posts ->
  matched only `close-deep-dive` (the one unrepaired disclaimer), zero matches
  on the other 78, confirming no false positives.
- `extracted_content_pipeline` re-synced via
  `extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`
  (`refreshed from atlas_brain sources (43 files)`); mirror carries the new
  patterns.
- `scripts/local_pr_review.sh` -> `local PR review passed` (plan shape,
  extracted manifest sync, ASCII Python policy, plan/code consistency,
  `git diff --check`).

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `b2b_blog_post_generation.py` (5 patterns + comment) | ~23 |
| `extracted_content_pipeline` mirror | ~23 |
| `test_b2b_blog_post_generation_quote_gate.py` (2 tests) | ~41 |
| Plan doc | ~120 |
| **Total** | **~207** |
