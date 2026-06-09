# PR-Deflection-Inline-Html-Strip-Before-Clustering

## Why this slice exists

PR #1432 closed the broad provider-HTML strip path for support-ticket
clustering, but review found the narrowing fix swung too far: with the
catch-all paired-tag detector removed, common inline provider tags such as
`<a href="...">` and `<b>` no longer count as HTML. On `origin/main`, a
link-heavy ticket still produces cluster tokens like `href`, `http`, `token`,
and URL fragments, which is exactly the #1416 pollution this lane exists to
remove.

This follow-up fixes the class rather than only the cited examples by auditing
the inline provider tags that commonly appear in help-desk exports and pinning
both strip-direction and preserve-direction behavior in tests.

## Scope (this PR)

Ownership lane: deflection/clustering-html-normalization
Slice phase: Production hardening

1. Expand support-ticket HTML detection to include common inline provider tags
   that can carry text or URL attributes into deterministic cluster tokens.
2. Keep the #1432 fail-safe intact: arbitrary paired non-HTML/XML-like ticket
   tags such as `<email>...</email>` and `<config>...</config>` must still
   bypass `HTMLParser`.
3. Add regression coverage proving inline links/bold markup strip from
   readable text and do not leak attribute/URL fragments into
   `support_ticket_tokens`.
4. Keep the change scoped to the shared support-ticket normalization helper and
   its extracted package tests; no LLM, no provider-specific parser, no UI
   surface.

### Review Contract

- Acceptance criteria:
  - [ ] `<a href="...">`, `<b>`, and other common inline provider tags are
        detected as HTML and stripped before source text and cluster tokens are
        produced.
  - [ ] URL and attribute fragments from links, such as `href`, `http`,
        domains, query keys, or reset tokens, do not appear in
        `support_ticket_tokens`.
  - [ ] Literal paired non-HTML/XML-like tags such as
        `<email>user@example.test</email>` and `<config>retries=3</config>`
        remain preserved as non-HTML.
  - [ ] Existing structural/custom provider HTML detection from #1432 remains
        covered.
- Affected surfaces: support-ticket text normalization and deterministic
  clustering in `extracted_content_pipeline`.
- Risk areas: under-stripping inline HTML, over-stripping XML/code-like ticket
  content, token pollution from URL attributes, review-example-only fixes.
- Reviewer rules triggered: R1, R2, R10, R13.

### Files touched

- `extracted_content_pipeline/support_ticket_clustering.py`
- `plans/PR-Deflection-Inline-Html-Strip-Before-Clustering.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The support-ticket plain-text helper already owns the shared normalization path used
by source material, customer wording, FAQ questions, resolution evidence, and
cluster tokens. This PR only adjusts `_HTML_SIGNAL_RE` so common inline HTML
tags emitted by help-desk rich-text exports are treated as real HTML signals.

When an inline tag is detected, the existing `HTMLParser` extractor removes the
tag and attributes, keeps readable text, skips script/style bodies, and then
compacts whitespace. The custom hyphenated provider-tag detector remains for
wrapper components, and arbitrary non-hyphenated paired tags that are not in
the known HTML set still bypass the parser.

## Intentional

- This PR keeps the #1432 known-tag approach instead of restoring the
  catch-all paired-tag regex. The catch-all caused content loss for legitimate
  XML/code-like support-ticket text.
- This PR does not add a dependency such as BeautifulSoup/lxml. The existing
  Python `HTMLParser` path is sufficient once detection recognizes common
  provider inline tags.
- This PR does not claim a full sanitized provider fixture sweep. It fixes and
  tests the inline HTML class that review found after #1432.

## Deferred

- #1384 follow-up: sanitized real Zendesk/Intercom/Help Scout/Freshdesk
  provider fixtures for the full parse-to-cluster path.

Parked hardening: none.

## Verification

- Focused support-ticket input pytest:
  python -m pytest tests/test_extracted_support_ticket_input_package.py -q
  - Result after review-thread fix: `40 passed in 0.31s`.
- Extracted content package validation:
  bash scripts/validate_extracted_content_pipeline.sh
  - Result: passed.
- Reasoning-import audit:
  python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Result: clean.
- Extracted standalone audit:
  python scripts/audit_extracted_standalone.py --fail-on-debt
  - Result: `Atlas runtime import findings: 0`.
- ASCII policy:
  bash scripts/check_ascii_python.sh
  - Result: passed.
- Full extracted pipeline CI mirror:
  bash scripts/run_extracted_pipeline_checks.sh
  - Result after review-thread fix: `3563 passed, 10 skipped, 1 warning in 55.14s`.
- Pending before push: local PR review via scripts/push_pr.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_clustering.py` | 14 |
| `plans/PR-Deflection-Inline-Html-Strip-Before-Clustering.md` | 120 |
| `tests/test_extracted_support_ticket_input_package.py` | 73 |
| **Total** | **207** |
