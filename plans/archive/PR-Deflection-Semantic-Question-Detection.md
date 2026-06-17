# PR-Deflection-Semantic-Question-Detection

## Why this slice exists

#1463's remaining concrete P2 parser item is semantic question detection. The
issue names three customer phrasings that still reproduce on current `main`:
`Unable to reset password`, `The export keeps failing`, and `please reset my
password` all return no question from `_question_text(...)`, so they fall back
to weaker generated labels even though they clearly express a support question.

Root cause: `_question_text_matching(...)` only recognizes explicit
question-starts and first-person issue forms. It does not have a conservative
template for short support-action complaints without "I/we", so useful customer
phrasing is dropped before FAQ labels are selected.

This PR fixes the root in the question extraction layer, not downstream in
rendering. It adds a bounded semantic support-issue template for the exact class
and tests both cited cases and allowed near-misses.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser
Slice phase: Product polish

1. Add a conservative semantic support-issue question helper in
   `ticket_faq_markdown.py`.
2. Recognize support-action variants of:
   - `Unable to ...` / `Cannot ...`
   - `please <support action> ...`
   - `<thing> keeps failing`
3. Add focused tests for the #1463 cited examples plus same-class held-out
   variants and near-misses that must stay unrecognized.

### Review Contract

Acceptance criteria:
- The three #1463 examples become usable questions.
- Same-class held-out support examples become usable questions.
- Near-misses such as generic "please see attached logs" do not become FAQ
  questions.
- Existing explicit question and first-person behavior remains unchanged.

Affected surfaces:
- Extracted support-ticket FAQ question-label extraction.

Risk areas:
- Over-capturing generic statements into buyer-facing FAQ questions.
- Fixing only the cited examples rather than the class.

Reviewer rules triggered: R1, R2, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Semantic-Question-Detection.md`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

`_question_text_matching(...)` already routes through explicit question starts
and first-person issue templates. This PR adds one more helper after those
existing branches: `_semantic_support_issue_question_text(...)`.

The helper is intentionally small. It only templates support-action terms such
as reset/export/update/download/access/login/invite/payment/billing, and it has
separate shapes for `unable/cannot`, `please <action>`, and `keeps failing`.
Everything still passes through the existing `_normalize_question_text(...)` and
predicate check before it can become a label.

## Intentional

- This is not a broad NLP parser or LLM repair. It is a deterministic template
  for a known support-ticket phrasing class.
- Generic polite statements remain unrecognized unless they contain a supported
  action term.
- The stemmer/synonym-map and HTML bullets in #1463 are left for reconciliation
  or separate slices; this PR only takes the proven semantic-question residual.

## Deferred

- Reconcile the remaining #1463 bullets after this PR: some are already landed,
  some are intentionally superseded, and any true residual should get its own
  narrow slice.

Parked hardening: none.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py -q -- 418 passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 185 matching
  tests are enrolled.
- bash scripts/run_extracted_pipeline_checks.sh -- extracted reasoning core 295
  passed; extracted content pipeline 4659 passed, 10 skipped, 1 warning.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 118 |
| `plans/PR-Deflection-Semantic-Question-Detection.md` | 102 |
| `tests/test_extracted_ticket_faq_markdown.py` | 51 |
| **Total** | **271** |
