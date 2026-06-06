# PR-Orphan-Detector-Question-Stacked

Ownership lane: `content-ops/blog-orphan-detector`

## Why this slice exists

PR #745 (D1 form-prompt removal) surfaced two false-NEGATIVES in the
orphaned-quote-reference detector -- it reported `0/0` while real danglers
remained:

1. **"question" back-references.** G2 form-PROMPTS *are* questions ("What do
   you like best about X?"). When those blockquotes were stripped, follow-ons
   like *"This open-ended question from a verified review ..."*
   (switch-to-woocommerce) and *"This quote is a question rather than a
   statement ..."* (power-bi) dangled. The detector only matched
   "quote"/"excerpt" shapes.

2. **Stacked references.** Only the FIRST `<p>` after a blockquote legitimately
   references it. A second consecutive ref-`<p>` points at a *different*
   (stripped) quote, but a nearby already-referenced blockquote suppressed the
   flag -- the power-bi case: two "This quote ..." paragraphs under one block,
   the second calling it "a question rather than a statement".

The fix was deferred from #745 to "a separate skill change". This slice does it
in both the skill's audit detector and the generator detector kept in sync.

## Scope (this PR)

Two detectors, "kept in sync" per their own comments:

- **Generator** (`_ORPHAN_QUOTE_REF_RE` / `_looks_like_orphan_quote_reference`,
  both byte-identical `b2b_blog_post_generation.py` copies) -- gets (a) the
  question-artifact shapes. (b) does NOT apply: the generator only consults the
  detector on the single paragraph immediately after a *stripped* block
  (`_remove_unmatched_quote_lines` forward-widen), so the "two refs after a
  kept block" shape structurally cannot arise. Documented in a code comment.
- **Audit** (`detectOrphanedQuoteReference` in the seo-geo-aeo-blog-post skill's
  `scripts/audit-published-posts.js`) -- gets (a) AND (b). The audit scans
  finished documents, where stacked refs occur. This file is an untracked local
  skill script (NOT in this repo); the change is applied in place and noted
  here. A `--self-test` regression harness was added since the file has no test
  runner.

### Files touched

- `plans/PR-Orphan-Detector-Question-Stacked.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`

### Out of repo (coordinated)

- `~/.claude/skills/seo-geo-aeo-blog-post/scripts/audit-published-posts.js`
  -- untracked skill script; (a) + (b) + `--self-test`.

## Mechanism

**Generator (Python).** `_ORPHAN_QUOTE_REF_RE` gains two quoted-question
alternatives -- `open[- ]ended\s+question` and `question\s+format` -- alongside
the existing `quote`/`excerpt` shapes, all anchored to a leading
`(?:that|this|the)`. Bare "question" is deliberately excluded. Applied to both
byte-identical `b2b_blog_post_generation.py` copies (edit one, `cp` to the
mirror; verified identical).

**Audit (JS, out of repo).** `detectOrphanedQuoteReference` gets the same
narrowed `flag` regex (a), plus the stacked-reference fix (b): the backward
look-back that decides whether a ref is "claimed" by a nearby blockquote now
stops at a *prior* orphan-ref `<p>` (extracted into an `isRef()` helper), so the
second of two stacked refs is flagged even though a blockquote sits within the
window. A `--self-test` flag (early-exit before repo resolution, relying on
hoisted function declarations) runs seven fixtures, including the power-bi
stacked case and the two rhetorical false-positive guards.

Narrowing was driven empirically: run against the live corpus, a bare
"question" pattern flagged two rhetorical paragraphs ("The question is not which
vendor ...") that reference no quote. The narrowed pattern drops both while
still catching the real danglers and the one stacked orphan.

## Intentional

- **Narrow "question" matching.** Only quoted-question artifacts
  ("open-ended question", "question format") -- NOT a bare "question". A bare
  pattern false-positived on rhetorical author prose ("The question is not which
  vendor ...", "The question isn't whether ...") when run against the live
  corpus; narrowing removed those FPs while keeping the real danglers.
- **(b) lives only in the audit.** The generator's single-paragraph,
  after-stripped-block sweep cannot produce the stacked shape; forcing the logic
  in would add risk with no coverage.

## Deferred

- **One real orphan the hardened audit newly surfaced (DATA, not detector):**
  `project-management-landscape-2026-04` L266 ("The quote reflects Notion's
  appeal ...") sits under a Wrike blockquote, after another ref -- a genuine
  stacked dangler the old detector missed. Belongs in a follow-up data PR
  (like #723/#745), not here.
- Versioning the skill script (still untracked, edited in place).

## Verification

- New Python regression tests fail pre-fix, pass after: question-artifact match,
  swept-with-block, survives-after-kept-block, and rhetorical-FP negatives.
- `pytest` quote-gate + generation + blog-ts suites -> 189 passed.
- Both `b2b_blog_post_generation.py` copies byte-identical after edit.
- JS `--self-test` -> 9/9 PASS.
- Live audit (`atlas-churn-ui`, 78 posts): `Orphaned quote reference` 3 -> 1
  after narrowing -- the 2 rhetorical FPs gone, the 1 real stacked orphan
  (project-management-landscape) correctly retained. `Form-prompt-as-quote 0`.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (regex + comments) | ~60 |
| Tests | ~75 |
| Plan doc | ~95 |
| **Total** | **~230** |
| Skill JS (out of repo, not counted above) | ~60 |
