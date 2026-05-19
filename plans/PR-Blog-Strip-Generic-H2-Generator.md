# PR-Blog-Strip-Generic-H2-Generator

## Why this slice exists

Companion to PR #649 (which strips generic `<h2>Introduction</h2>`
and `<h2>Conclusion</h2>` from the 77 already-published blog posts).
That PR cleans the data; this slice closes the upstream so future
generations don't reintroduce the anti-pattern.

The pattern source is the 10 blueprint functions in
`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`. Each
defines a first `SectionSpec(heading="Introduction", ...)` as the
post's opening "hook" section. The LLM follows that hint and emits
`<h2 id="introduction">Introduction</h2>` in the rendered HTML. AI
engines (ChatGPT, Perplexity, Google AI Overviews) skip sections
whose heading doesn't predict the content underneath, so the
opening section's hook content -- which is exactly what should be
extracted for AI citations and featured snippets -- becomes
extraction-hostile because of the boilerplate header above it.

Two ways to fix:

1. Rename the heading per blueprint to a topic-aware default.
   Better UX (still scannable) but requires per-blueprint copy
   decisions and writes prose into the producer.
2. Strip the literal anti-pattern lines from rendered content at
   the quality-gate step.

Choosing (2) for symmetry with how other generator anti-patterns
get handled (`_remove_unmatched_quote_lines`,
`_apply_specificity_anchor_repair`), and because the prose
underneath the H2 already serves as the post's opening hook -- the
header was always boilerplate, not load-bearing.

## Scope (this PR)

1. Add `_GENERIC_HEADING_LABELS` constant listing the literal
   anti-pattern heading labels: `Introduction`, `Conclusion`,
   `Overview`, `Summary`, `Background`, `TL;DR`.
2. Add `_GENERIC_SECTION_HEADING_RE` -- a multi-line, case-
   insensitive regex matching `<h2 [...attrs]>LABEL</h2>` followed
   by an optional newline. Matches HTML form only; markdown
   `## Introduction` is left alone because the rendered output
   the audit cares about is HTML.
3. Extend `_sanitize_blog_markdown` to also strip generic H2s,
   record the count, and surface it via a new `fixes_applied`
   entry: `removed_generic_section_heading:N`.
4. Add two tests: one verifies the strip fires + leaves
   substantive content untouched; one verifies topic-specific H2s
   are NOT touched.
5. Sync the change into `extracted_content_pipeline`.

### Files touched

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_quality_gate.py`
- `plans/PR-Blog-Strip-Generic-H2-Generator.md`

## Mechanism

The strip lives in `_sanitize_blog_markdown`, which already runs
the deterministic-cleanup pass alongside the `Answer:` prefix
strip. Same pattern: detect, replace, count, emit a fix label.

The regex is HTML-anchored (`<h2...>`) so markdown headings (`##
Introduction`) are unaffected. The label list is literal; future
generic labels (e.g., `Foreword`, `Preface`) can be added to
`_GENERIC_HEADING_LABELS` without other changes.

Fix label flows through to the report's `fixes_applied` array,
making the action observable in the quality-pipeline telemetry the
same way `removed_answer_prefix:N` and `removed_unmatched_quotes:N`
already are.

## Intentional

- HTML-anchored regex only. Markdown `## Introduction` inside
  intermediate generation is fine; the strip only operates on the
  rendered HTML that ships. If a future producer emits markdown
  directly, that path needs its own filter.
- Label list is explicit and short. Generic-heading detection
  could be heuristic ("any single-word noun heading at section
  start") but a literal list keeps the contract auditable and
  avoids over-stripping legitimate single-word vendor or category
  H2s.
- No change to the blueprint functions themselves. The
  `heading="Introduction"` defaults are technically dead from a
  rendering standpoint after this strip, but the SectionSpec
  contract still requires a heading string and downstream prompt
  logic uses the heading as a hint for what the section is
  *about* (not literally what to render). Updating the blueprints
  to use topic-aware defaults is a follow-up.

## Deferred

- Per-blueprint topic-aware heading defaults (`heading=f"Why
  {vendor} Reviewers Concentrate the Most Friction"` etc.). A
  larger refactor that touches 10 blueprint functions and
  requires copy decisions per topic_type. The current strip-only
  approach is sufficient for the AEO + featured-snippet goal.
- Markdown-form generic headings (`## Introduction` outside
  HTML). Not currently observed in production; cleanup-when-
  needed.
- Telemetry alerting when the strip fires unusually often
  (could indicate a regression in the LLM prompt). Routine
  observability work, not blocking.

## Verification

- `tests/test_b2b_blog_quality_gate.py` -> 212 tests passed via
  `pytest` (was 210, +2 new tests covering the strip).
- Two updated existing tests
  (`test_quality_gate_sanitizes_answer_prefix_and_drops_unsourced_quotes`,
  `test_quality_gate_drops_quotes_from_unexpected_vendors`) had
  their fixture-content blockquotes separated into independent
  paragraphs to satisfy the all-quote-lines-must-ground contract
  introduced in PR #625; the assertions are unchanged.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> expected to pass.
- `extracted_content_pipeline` re-synced via the documented
  script.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `b2b_blog_post_generation.py` (constant + regex + sanitizer call) | ~35 |
| `extracted_content_pipeline` mirror of same change | ~35 |
| `test_b2b_blog_quality_gate.py` (2 new tests + 2 fixture tweaks) | ~75 |
| Plan doc | ~125 |
| **Total** | **~270** |
