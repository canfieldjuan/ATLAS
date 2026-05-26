# PR-FAQ-Answer-Step-Grounding

## Why this slice exists

The FAQ generator is deterministic and grounded in source-ticket language, but
its "What to do next" steps are currently chosen from intent templates even
when the upload contains no resolved-answer evidence. That is acceptable for a
review draft only if the artifact says so clearly. Without that contract, the
FAQ wedge can imply verified support instructions from tickets that only
describe problems.

This slice is production hardening for the FAQ lane: concrete generated steps
should be grounded in uploaded resolution evidence. When that evidence is
absent, the artifact should degrade to draft/review guidance instead of
presenting template steps as verified answers.

Slice size: **medium**. This changes the generated FAQ output contract and its
focused tests, but it stays inside the FAQ renderer and does not touch hosted
routes, persistence, DB migrations, or landing/blog generation.

The final diff is expected to exceed the usual 400 LOC target after review
because the same slice now includes the stale CFPB smoke consumer assertion, the
negative disposition-key fixture, and the render-cap/source-count regression
fixtures needed to close the hardening item honestly.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

Slice phase: Production hardening.

1. Preserve resolution-style fields from source rows when grouping FAQ evidence.
2. Use resolution evidence to build concrete FAQ action steps when present.
3. Render explicit draft/review guidance when resolution evidence is absent.
4. Expose compact per-item metadata so callers can tell which mode was used.
5. Remove the now-closed FAQ action-step hardening entry.
6. Update the CFPB FAQ smoke consumer to the new no-resolution draft contract.
7. Add focused regression tests for grounded, no-resolution, disposition-key,
   render-cap, and source-count behavior.

### Files touched

- `plans/PR-FAQ-Answer-Step-Grounding.md`
- `HARDENING.md`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_cfpb_faq_markdown.py`

## Mechanism

The FAQ grouping path will carry a normalized `resolution_text` field from
explicit resolution-text fields on evidence/opportunity rows. `_item(...)` will classify each FAQ item into one of
two modes:

```python
if resolution_texts:
    answer_evidence_status = "resolution_evidence"
    steps = _resolution_article_steps(resolution_texts, support_contact=...)
else:
    answer_evidence_status = "draft_needs_review"
    steps = _draft_review_steps(support_contact=...)
```

The Markdown still keeps the same "What to do next" section so downstream
renderers do not need a shape change, but no-resolution output will explicitly
say it is draft guidance that must be reviewed against the team's policy or
runbook before publishing.

Resolution grounding uses all rows in a grouped FAQ item, not only the
displayed evidence rows, so `max_evidence_per_item` cannot hide resolution
evidence. `resolution_source_count` counts distinct resolution-backed source
IDs instead of deduplicated macro text.

## Intentional

- No LLM call, summarizer, or policy inference is introduced. Resolution
  evidence is copied/excerpted from uploaded structured fields.
- Existing topic, scoring, vocabulary-gap, and source-id behavior stays
  unchanged.
- The no-resolution path still has actionable review steps so existing
  `has_action_items` output checks remain meaningful, but those steps are about
  authoring/reviewing the answer rather than instructing an end user to perform
  unsupported product-specific actions.
- Bare upload columns like `resolution`, `answer`, and `solution` are not
  treated as resolution evidence because they commonly contain disposition or
  survey values, not verified support-answer text.

## Deferred

- Rich resolution parsing from long chat transcripts is deferred; this slice
  only uses structured resolution-style fields already present on rows.
- Hosted UI display for `answer_evidence_status` is deferred until the frontend
  consumes FAQ item metadata.
- The older FAQ scale/backpressure item remains parked; this slice only closes
  the answer-step grounding item.

## Verification

- `python -m pytest tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_cfpb_faq_markdown.py -q` - 145 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` - 2421 passed, 5 skipped, 1 warning.
- `python -m py_compile extracted_content_pipeline/ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py tests/test_smoke_content_ops_cfpb_faq_markdown.py` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Answer-Step-Grounding.md` - passed.
- `git diff --check` - passed.
- `bash scripts/validate_extracted_content_pipeline.sh` - passed.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - passed.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - passed.
- `bash scripts/check_ascii_python.sh` - passed.
- `bash scripts/local_pr_review.sh` - pending clean-branch run after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 119 |
| Hardening closeout | 11 |
| FAQ renderer | 160 |
| FAQ tests | 175 |
| CFPB smoke consumer | 2 |
| **Total** | **467** |
