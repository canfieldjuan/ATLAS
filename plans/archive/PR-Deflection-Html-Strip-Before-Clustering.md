# PR-Deflection-Html-Strip-Before-Clustering

## Why this slice exists

Issue #1384 tracks pre-launch robustness for provider support-ticket exports.
PR #1391 already handled the P0 CSV parser hardening. This slice takes the
remaining narrow P1 item from #1384: HTML-heavy Zendesk/Help Scout/Freshdesk
ticket bodies should be converted to readable customer wording before
deterministic clustering, examples, FAQ questions, and resolution evidence are
derived.

The public portfolio intake was also checked read-only: `/api/gap-report-intake/record`
calls `submitDeflectionReportCsv()`, which reads the private Blob and forwards
raw CSV bytes to ATLAS. There is no separate portfolio-side clustering parser
for this path, so the ATLAS normalization helper is on the public funnel.

## Scope (this PR)

Ownership lane: go-live-deflection-cleanup
Slice phase: Production hardening

1. Broaden support-ticket HTML detection beyond the original common-tag list
   to real provider/custom wrapper tags.
2. Strip known HTML tags and custom hyphenated provider tags before source
   text, customer wording examples, FAQ questions, resolution evidence, and
   cluster tokens are produced.
3. Preserve non-HTML angle-bracket text such as `total < 10` and decoded
   literal markers such as `<manual_review>`.
4. Preserve raw paired non-HTML/XML-like ticket text such as
   `<email>user@example.test</email>`.
5. Add a regression test covering subject, description, comments, resolution
   text, script removal, cluster-label contamination, and customer wording.

### Review Contract

- Acceptance criteria:
  - [ ] Generic provider HTML tags such as `<section>`, `<article>`,
        `<custom-widget>`, and `<answer-card>` are removed before source text,
        customer wording examples, FAQ questions, resolution evidence, and
        cluster tokens are produced.
  - [ ] Script/style body content remains excluded from support-ticket text.
  - [ ] Plain non-HTML text containing `<` remains intact.
  - [ ] Decoded literal markers like `&lt;manual_review&gt;` are not
        accidentally treated as HTML.
  - [ ] Raw paired non-HTML/XML-like tags such as `<email>...</email>` remain
        intact rather than being routed through `HTMLParser`.
  - [ ] Existing common-tag stripping and messy untagged clustering behavior
        remain covered.
- Affected surfaces: support-ticket text normalization and deterministic
  clustering in `extracted_content_pipeline`.
- Risk areas: false-positive tag stripping, under-stripping custom provider
  tags, text spacing drift, clustering label drift.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/support_ticket_clustering.py`
- `plans/PR-Deflection-Html-Strip-Before-Clustering.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The support-ticket plain-text helper remains the single normalization path used by
the input package and clustering path. The detector now treats text as HTML
only when it sees a common real tag or a custom hyphenated provider tag. Plain
text bypasses the parser even if it contains an ordinary less-than comparison
or raw paired non-HTML/XML-like tags.

When HTML is detected, the existing `HTMLParser` boundary collects readable
text, skips script/style content, decodes entities, and compacts whitespace.
The support-ticket input package already routes subject, description, comments,
and resolution text through this helper before assigning clusters, so the
normalization happens before clustering and every downstream report surface
receives the same plain-text ticket body.

The regression test builds a support-ticket package with HTML in subject,
description, comments, and resolution text. It asserts that normalized source
text and customer wording are stripped, script content is gone, tag names do
not leak into the cluster label, a separate row containing `total < 10` keeps
the literal comparison text, a decoded `<manual_review>` marker stays literal,
and raw paired non-HTML tags stay intact.

## Intentional

- This slice does not add BeautifulSoup/lxml or any new dependency. Python's
  `HTMLParser` is already used here and is enough for provider export cleanup.
- This is not the full #1384 real-provider fixture or inspect-preview gate. It
  only hardens the HTML normalization item now that the P0 CSV parser work is
  already merged.
- Exact cluster labels are not newly pinned for the HTML-heavy row; the test
  pins the behavioral contract that tag text does not leak into clustering.

## Deferred

- #1384 follow-up: sanitized real Zendesk/Intercom/Help Scout/Freshdesk
  provider fixtures for the full parse-to-cluster path.
- #1384 follow-up: make inspect a true pre-payment preview/validation gate.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_support_ticket_input_package.py -q`
  - Result: `29 passed in 0.30s`.
- `scripts/run_extracted_pipeline_checks.sh` via bash
  - Result: `3552 passed, 10 skipped, 1 warning in 53.82s`; wrapper completed.
- `scripts/local_pr_review.sh` via bash with current body file
  `tmp/pr-body-deflection-html-strip-before-clustering.md`
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_clustering.py` | 30 |
| `plans/PR-Deflection-Html-Strip-Before-Clustering.md` | 118 |
| `tests/test_extracted_support_ticket_input_package.py` | 71 |
| **Total** | **219** |
