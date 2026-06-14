# PR-Deflection-Customer-Wording-PII-Headings

## Why this slice exists

The next deflection/clustering hardening target started as the parked security
item `Customer-wording FAQ headings can publish raw PII`. Review showed heading
admission was only one path to the real defect: raw customer-derived free text
also reached buyer-facing output through fallback topics, evidence quotes,
source titles, summaries, and action context.

Root cause: the FAQ renderer has no shared "published customer text" boundary.
Different render surfaces independently decide whether to emit raw uploaded
text. A heading-only denylist fixes one symptom while leaving the true upstream
cause in place.

This fixes the root inside this renderer with one shared buyer-facing
customer-text gate before Markdown-facing fields are built. It is not global
ingestion redaction; other products that publish raw support-ticket text need
their own upstream privacy issue.

## Scope (this PR)

Ownership lane: deflection/clustering
Slice phase: Production hardening

1. Add a shared published-customer-text gate for FAQ Markdown-facing fields.
2. Block emails, phone-shaped numbers, redaction artifacts, and long numbers
   only in identifier contexts such as account, case, order, reference, or
   ticket.
3. Apply the gate to customer-wording question selection, source-title topic
   fallback, source labels, evidence quotes, summaries, and action context.
4. Add focused tests for unsafe heading/body text, safe customer wording, safe
   form/year numbers, and fallback stability.
5. Remove the closed renderer PII hardening item from `HARDENING.md`.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Customer-Wording-PII-Headings.md`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] Customer-wording questions containing emails, identifier-context long
        numbers, phone-shaped numbers, or redaction artifacts are not selected.
  - [ ] Safe customer wording still renders as customer wording, including
        non-identifier four-digit terms such as tax forms or years.
  - [ ] Unsafe customer wording falls through to existing source-policy or
        generic fallback rather than being rewritten.
  - [ ] Unsafe source titles cannot become fallback topic headings.
  - [ ] Unsafe source titles and evidence snippets do not render in buyer-facing
        Markdown source lines.
  - [ ] The representative-label behavior from the prior slice is unchanged.
- Affected surfaces: ticket FAQ Markdown item question selection and focused
  extracted-package tests.
- Risk areas: buyer-facing FAQ headings, source-policy fallback stability,
  customer-vocabulary retention, source-line privacy, and false positives on
  normal account/login/year/form wording.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14.

## Mechanism

The renderer now has one helper that returns customer-derived text only when it
is safe for buyer-facing output. The question resolver uses it at candidate
admission time and checks source context before punctuation splitting, so an
email cannot be reduced to a clean-looking fragment. Source-title fallback
topics, source labels, evidence quote text, summary examples, and action context
use the same helper.

The safety check blocks emails, phone-shaped numbers, redaction artifacts, and
long numbers in identifier contexts. Generic `account`, `email`, `login`, and
`password` wording remains allowed, as do non-identifier numbers such as `1099`
or `2024`.

If every customer-wording candidate is skipped, existing source-policy fallback
handles the question. Unsafe evidence/source titles render as source ID plus a
privacy placeholder.

## Intentional

- **No per-surface heading scrub.** This is a shared renderer boundary, not a
  one-off post-render heading rewrite.
- **No broad ban on customer vocabulary.** Safe terms and bare long numbers
  without identifier context stay valid by accepted precision trade.
- **No global ingestion redaction.** This slice owns the FAQ Markdown renderer;
  other raw-text publishing products need separate privacy follow-up.
- **No duplicate-label work.** That is a controlled-vocabulary collision issue,
  not a reason to merge groups again.
- **No metric/copy alignment work.** #1533 already landed that item.
- **No embedding-booster work.** The operator stopped further embedding-booster
  buildout unless a new explicit enablement/calibration decision is made.

## Deferred

- Safe-vocabulary representative label collision handling. Root cause: distinct
  kept subclusters can resolve to the same controlled-vocabulary label because
  the labeler has no disambiguation term after privacy-safe token selection.

Parked hardening: `Safe-vocabulary representative label collisions render
duplicate FAQ headings`.

## Verification

- Command passed: pytest targeted PII/body and nearby representative-label regressions -- 11 passed.
- Command passed: python -m py_compile for the touched Python files.
- Command passed: pytest tests/test_extracted_ticket_faq_markdown.py -q -- 261 passed.
- Command passed: bash scripts/check_ascii_python.sh.
- Command passed: git diff --check.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4143 passed, 10 skipped, 1 warning.
- Command passed: bash scripts/local_pr_review.sh --current-pr-body-file tmp/deflection_customer_wording_pii_headings_pr_body.md.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 119 |
| `plans/PR-Deflection-Customer-Wording-PII-Headings.md` | 122 |
| `tests/test_extracted_ticket_faq_markdown.py` | 149 |
| **Total** | **399** |
