# PR-Blog-Generator-D1-D9

Ownership lane: content-ops/blog-generator-d1-d9

## Why this slice exists

A per-post correctness audit (catalogued in `reports/blog-audit-findings.md`)
found two defect classes that the data-cleanup PRs (#741, #745) removed from
published posts but that the **generator still produces**, so they would
recur:

- **D1 -- form-prompt-as-quote.** G2 review-FORM PROMPTS ("What do you like
  best about <X>?") get scraped into `review_text` as boilerplate and pass the
  quote gate as verbatim phrases, surfacing as fake reviewer quotes. 47
  occurrences across 31 posts before cleanup.
- **D9 -- markdown-in-`<p>`.** A bullet list tightly coupled to a label (no
  blank line, e.g. `**Top pain points:**\n- Pricing`) is left un-converted by
  the markdown step, rendering as literal "- item" text. 61 occurrences across
  10 posts before cleanup.

This PR fixes both at the source so new posts are clean, and the skill's
`form_prompt_quote` / `markdown_in_html` detectors stay at 0.

## Scope (this PR)

- **D1:** add `_is_form_prompt` + a reject filter in
  `_split_and_gate_blog_quotes` (the gate every blueprint path uses to build
  `quotable_phrases`), so form prompts are dropped from the quote pool.
- **D9:** add `_ensure_blank_line_before_lists` and apply it just before the
  markdown->HTML conversion in `_blog_ts.py` (`build_post_ts` /
  `_render_markdown`), so tightly-coupled lists convert to a real `<ul>`.
- Tests for both; sync the change into `extracted_content_pipeline`. The
  `b2b_blog_post_generation.py` mirror is source-mapped (auto-synced); the
  `_blog_ts.py` mirror is a target-only (divergent) copy with the same bug, so
  the D9 fix is applied to it directly.
- Includes the audit catalog `reports/blog-audit-findings.md` as the rationale
  + remaining-work tracker (D2/D3/D4/D7/D8 still open).

### Files touched

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `atlas_brain/autonomous/tasks/_blog_ts.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/_blog_ts.py`
- `tests/test_b2b_blog_post_generation_quote_gate.py`
- `tests/test_blog_ts_publish.py`
- `reports/blog-audit-findings.md`
- `plans/PR-Blog-Generator-D1-D9.md`

## Mechanism

**D1.** `_FORM_PROMPT_RE` matches the G2 prompt shapes ("what do you like
best/dislike about", "what problems ... solving", "recommendations to others
considering", "what benefits have you realized"). After
`_split_and_gate_blog_quotes` assembles `combined`, any phrase matching the
pattern is dropped. Kept in sync with the skill's `form_prompt_quote`
detector. ASCII-only.

**D9.** `_ensure_blank_line_before_lists` inserts a blank line before a bullet
line (`- `/`* `) that immediately follows a non-blank, non-bullet line.
python-markdown requires that blank line to start a `<ul>`; without it the
list is absorbed into the preceding `<p>` and emitted as raw text. Applied at
the single md->html boundary so it covers every generation path. ASCII-only;
`build_post_ts`'s signature is unchanged.

## Intentional

- **Filter at the shared gate, not per-blueprint.** `_split_and_gate_blog_quotes`
  is called by every topic-type producer, so one filter covers all paths.
- **Fix D9 at the conversion boundary**, not by rewriting LLM output -- the
  blank-line normalization is deterministic and source-agnostic.
- **Mirror D9 into the target-only extracted copy by hand**, since it is not
  source-mapped but carries the same bug (and the extracted pipeline renders
  blogs through it).
- **No behavior change to real quotes / real lists** -- only form prompts are
  dropped, only tightly-coupled lists get a blank line.

## Deferred

- **D2/D3/D4/D7/D8** (count-vs-list, prose-vs-chart, strength/weakness
  mislabel, corpus-number overstatement, vendor-count) -- catalogued; need
  blueprint/data-layer changes + DB cross-checks, not regex.
- **Detector hardening** for "question"/stacked orphan-quote references (a
  skill change, tracked from the #745 review).

## Verification

- `tests/test_b2b_blog_post_generation_quote_gate.py` + `tests/test_blog_ts_publish.py`
  -> `91 passed` (added: form-prompt detection, gate-drops-form-prompt, and
  tight-bullet-list -> `<ul>`).
- `tests/test_extracted_blog_post_export.py` + `tests/test_extracted_blog_generation.py`
  -> `36 passed` (mirror change safe).
- Functional check: `_split_and_gate_blog_quotes` drops "What do you like best
  about Zoho CRM", keeps "Support never responded for weeks"; a tight list
  renders `<ul><li>Pricing</li><li>Support</li></ul>`.
- `extracted_content_pipeline` re-synced; manifest sync + ASCII Python policy
  PASS.
- `scripts/local_pr_review.sh` -> `local PR review passed` (plan shape,
  plan/code consistency 13/13 path claims, manifest sync, ASCII Python policy,
  cross-session ownership lane, diff drift ~6%, `git diff --check`).

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `b2b_blog_post_generation.py` + mirror (D1) | ~55 |
| `_blog_ts.py` + mirror (D9) | ~55 |
| tests | ~50 |
| `reports/blog-audit-findings.md` (audit catalog) | ~200 |
| Plan doc | ~110 |
| **Total** | **~470** |
