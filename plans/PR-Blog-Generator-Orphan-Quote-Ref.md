# PR-Blog-Generator-Orphan-Quote-Ref

## Why this slice exists

The generator-prevention phase of #706. The data fix (#723) cleaned the 7
orphaned quote references already live in `main`; this closes the upstream so
future generations cannot reintroduce them.

When `_remove_unmatched_quote_lines` strips an ungrounded/empty blockquote it
already sweeps an adjacent orphan **intro** (lead-in ending ":") and orphan
**disclaimer** paragraph. It did NOT sweep a following paragraph that
back-references the stripped quote ("This quote captures...", "The excerpt
cuts off...", "quoted earlier") -- exactly the shape that shipped into the 5
posts the data fix repaired. This widens the forward-sweep to catch that shape
too, keeping the generator and the skill's `orphaned_quote_reference` detector
in sync.

## Scope (this PR)

1. Add `_looks_like_orphan_quote_reference(line)` -- mirrors the skill
   detector: matches a line starting with "That/This quote", "This/The
   excerpt", or containing "quoted earlier"; same structural guards as
   `_looks_like_orphan_disclaimer`. The witness guard excludes only the
   AGGREGATE-noun forms ("the witness evidence/data/highlights/..."), so a
   genuine orphan ref like "The witness quoted earlier ..." is still caught
   (per Codex review on #728).
2. Extend the forward-sweep in `_remove_unmatched_quote_lines` to widen across
   an orphan quote-reference paragraph as well as a disclaimer.
3. Tests: the sweep fires on the back-reference shapes (incl. "The witness
   quoted earlier ..."); a false-positive guard that a generic follow-on and
   an aggregate "The witness evidence ..." reference are preserved; and a
   quote-reference-shaped follow-on after a KEPT (grounded) block survives
   (the legitimate #723 case -- the line matches the matcher, so the
   stripped-block gate is the only protection, and this pins it).
4. Sync the change into the `extracted_content_pipeline` mirror.

### Files touched

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`
- `plans/PR-Blog-Generator-Orphan-Quote-Ref.md`

## Mechanism

The forward-sweep only fires on a STRIPPED block, so the blast radius is
bounded by the existing block-strip logic. The new matcher reuses the
disclaimer guards, so a fresh blockquote / heading / list item that happens to
start with "this quote" is not swept. Patterns are ASCII-only (Python Unicode
policy) and are pattern definitions (like `_ORPHAN_DISCLAIMER_RES`), not
configurable values. `_remove_unmatched_quote_lines`'s signature is unchanged.

## Intentional

- **Explicit back-reference only.** Generic follow-ons ("This pattern recurs")
  are accepted behavior and carry real analysis -- they are NOT swept. Only
  paragraphs that name "this/that quote", "this/the excerpt", or "quoted
  earlier" are.
- **Whole-paragraph sweep (generator), surgical edit (data fix).** New
  generation removes the dangling follow-on outright (the quote it cites is
  gone); the published-data fix (#723) was surgical to preserve already-shipped
  analysis. Different tools, different appropriate behavior.
- **Kept in sync with the skill detector**, as with the disclaimer patterns
  (#703).

## Deferred

- None; this completes #706 alongside #723.

## Verification

- `tests/test_b2b_blog_post_generation_quote_gate.py` run via pytest ->
  `86 passed` (orphan-quote-reference swept incl. "The witness quoted
  earlier ..."; generic/aggregate-witness false-positive guard; and the
  kept-block-survives case from the #728 review).
- `extracted_content_pipeline` re-synced via
  `extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`
  (`refreshed from atlas_brain sources (43 files)`); mirror carries the new
  function.
- `scripts/local_pr_review.sh` -> plan shape, plan/code consistency, extracted
  manifest sync, ASCII Python policy, `git diff --check`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `b2b_blog_post_generation.py` (matcher + sweep) | ~55 |
| `extracted_content_pipeline` mirror | ~55 |
| `test_b2b_blog_post_generation_quote_gate.py` (tests) | ~55 |
| Plan doc | ~90 |
| **Total** | **~255** |
