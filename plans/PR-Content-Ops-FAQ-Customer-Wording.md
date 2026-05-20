# Content Ops FAQ Customer Wording

## Why this slice exists

The FAQ Markdown builder is now date-windowed, clustered by deterministic
intent, and exposes total ticket volume. The next output-quality gap is the FAQ
question text itself: `_item(...)` still renders a generic topic question such
as "What are customers asking about email and profile updates?" even when the
source ticket contains a direct customer question like "How do I change my login
email?".

This weakens the "Extract Customer Wording" workflow claim. This slice derives
FAQ questions from the source ticket wording when a concise question-like text
is present, while keeping the current generic fallback.

## Scope (this PR)

1. Extract a customer-worded FAQ question from displayed evidence rows when one
   is present.
2. Preserve the existing topic-based fallback when source text is not a usable
   question.
3. Add item metadata that records whether the question came from customer
   wording or the fallback.
4. Add regression coverage for customer wording and fallback behavior.
5. Replace the stale FAQ volume-count in-flight row with this active slice.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-FAQ-Customer-Wording.md` | Plan doc for this slice. |
| `docs/extraction/coordination/inflight.md` | Replace stale FAQ volume-count claim with this active slice. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Derive FAQ question text from source ticket wording when available. |
| `tests/test_extracted_ticket_faq_markdown.py` | Regression coverage for customer-worded questions and fallback. |

## Mechanism

`_item(...)` already receives the displayed evidence rows used for snippets.
This PR adds a small `_question(...)` helper that scans those rows in order,
normalizes whitespace, and returns the first concise question-like sentence. If
no such sentence exists, it returns the existing topic-based fallback. The item
also records `question_source` as either `customer_wording` or `topic_fallback`.
Question extraction strips URLs, ignores explicit agent/support prompts, accepts
leading unlabeled customer text before an agent label, treats speaker labels as
turn-boundary markers instead of arbitrary inline prose, and applies the same
length guard to both punctuated and normalized question-start fallbacks.

## Intentional

- No LLM rewrite or paraphrase. The customer-worded question is extractive.
- No change to answer generation, action items, clustering, or date filtering.
- The helper only uses displayed rows so the rendered question is traceable to
  visible evidence.

## Deferred

- Semantic question synthesis across many tickets remains deferred.
- UI edit/publish workflow remains separate from Markdown generation.
- Broader tone polish for answers remains a future output-quality slice.

## Verification

- pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_ops_execution_smoke.py::test_content_ops_execution_smoke_cli_runs_faq_markdown_json - 36 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py - passed
- git diff --check - passed
- bash scripts/run_extracted_pipeline_checks.sh - 1516 passed, 1 existing torch/pynvml warning

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Customer-Wording.md` | +75 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | +81 / -1 |
| `tests/test_extracted_ticket_faq_markdown.py` | +177 |
| Total | ~338 |
