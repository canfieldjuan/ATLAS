# PR-Blog-Orphaned-Quote-Reference-Cleanup

## Why this slice exists

The blog corpus integrity cleanup (#697) stripped empty blockquotes and their
lead-in/disclaimer prose, but left a third shape behind: **follow-on
paragraphs that explicitly back-reference a now-deleted quote** -- e.g.
*"That quote..."*, *"This quote captures..."*, *"The quote illustrates..."*,
*"...quoted earlier"*. These shipped to `main` when #697 merged with the
Codex P2 comments open, so they are live in production now.

The SEO audit does not catch this shape -- it detects empty blockquotes,
markdown-in-HTML, and acknowledged-misattribution disclaimers, but not an
orphaned analysis sentence whose subject quote no longer exists. So the class
is both **live** and **invisible to the existing tooling**.

Two scans bound the problem but neither is authoritative:
- Codex inline-flagged 6 posts (notably `power-bi-deep-dive`, with several).
- A rough local scan flagged ~12 references across 9 posts, but mixed in
  false positives: *"The witness evidence shows..."* references aggregate
  witness data, not a deleted quote, and is legitimate.

The true set is the union, minus the false positives -- which is exactly why
this needs a validated detector before any prose is touched.

## Scope (this PR)

This PR checks in the plan only. The work is phased across follow-up PRs so
the detector is proven before production prose is edited:

1. **This plan doc**, `plans/PR-Blog-Orphaned-Quote-Reference-Cleanup.md`.

The implementation phases (each its own PR) are specified below under
Mechanism so a builder can pick them up directly.

### Files touched

- `plans/PR-Blog-Orphaned-Quote-Reference-Cleanup.md`

## Mechanism

**Phase 1 -- precise detector (skill audit).** Add an
`orphaned_quote_reference` detector to the skill analyzer
(`~/.claude/skills/seo-geo-aeo-blog-post/scripts/audit-published-posts.js`):
flag a `<p>` that back-references a specific quote when **no `<blockquote>`
appears within the preceding ~3 non-blank lines**.
- **Flag:** paragraphs whose visible text starts with `That quote`,
  `This quote (captures|illustrates|reflects|shows|reveals)`,
  `The quote (above|illustrates|...)`, `This excerpt`, `The excerpt`, or
  contains `quoted earlier` / `quote above`.
- **Do not flag:** `The witness evidence/data/highlights ...` (aggregate
  references, legitimate), and generic analysis with no explicit quote
  back-reference.
- Tune until it catches the Codex-flagged set (`power-bi-deep-dive` x5, etc.)
  AND the genuine ones the rough scan surfaced (`close-vs-zoho-crm`,
  `marketing-automation-landscape`, `why-teams-leave-slack`,
  `top-complaint-every-project-management`) with **zero** false positives on
  "witness evidence" prose, validated across all published posts.

**Phase 2 -- data fix (its own PR; the references are live in main).** Run
the validated detector to get the authoritative set (estimate ~10-15
references across ~10 posts). Repair each with the same judgment used for the
lead-in trims and the copper repair: drop the orphaned sentence, or rephrase
to a standalone statement where it carries real substance. No fabrication, no
invented quotes. Verify: detector -> 0, `npm run build` clean, HTML
well-formedness holds.

**Phase 3 -- prevent recurrence at the source (its own PR).** Mirror the
detector's back-reference shape into the generator's orphan-prose stripper
(`_remove_unmatched_quote_lines` in
`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`) so a stripped
block's `That quote...` / `This quote captures...` follow-on is swept like the
lead-in and disclaimer already are -- but only the explicit-back-reference
shape, never generic follow-ons. Add regression tests (including a
false-positive guard) in `tests/test_b2b_blog_post_generation_quote_gate.py`,
and sync the change into
`extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`.
Keep the skill detector and the generator patterns in sync, as was done for
the disclaimer hardening (#703).

## Intentional

- **Detector before data.** The "this quote" back-reference vs "the witness
  evidence" aggregate-reference distinction is precisely where a naive strip
  would either miss real orphans or delete legitimate analysis. Proving the
  detector at zero false positives is the gate for editing any production
  prose.
- **Three phases, separate PRs.** Phase 1 (detector) lands with Phase 2 (the
  data fix it authorizes) so the detector output is the data fix's
  verification surface; Phase 3 (generator) is separate, like the #703
  hardening, because it is a producer-behavior change with its own
  false-positive risk.
- **Explicit back-reference only.** The generator and detector target only
  paragraphs that name "that/this/the quote" or "quoted earlier" -- not
  generic follow-ons ("This pattern recurs:"), which are accepted behavior
  and carry real analysis.

## Deferred

- **Common-word vendor contamination** (the upstream review matcher pulling
  off-topic reviews for "Close"/"Copper"). Separate, larger fix tracked
  outside this slice; needed before regenerating those vendors. Orphaned
  references are a downstream symptom of the same stripping work, not of the
  matching gap, so they are handled independently here.
- **Regenerating the pulled `close-deep-dive`** post -- gated on the
  common-word vendor fix.

## Verification

- This PR: `scripts/local_pr_review.sh` -> plan shape, plan/code consistency,
  `git diff --check`.
- Phase 1/2 PR: detector run over the corpus produces the authoritative
  orphan list, then `0` after the data fix; `npm run build` clean; HTML
  well-formedness validator -> 0 issues.
- Phase 3 PR: `python -m pytest tests/test_b2b_blog_post_generation_quote_gate.py`
  with the new sweep + false-positive-guard tests; extracted mirror re-synced.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~135 |
| **Total (this PR)** | **~135** |
