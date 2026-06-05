# Support Ticket Generated Content Quality Eval

## Why this slice exists

PR #950 proves the support-ticket source context survives into saved-draft
exports for landing pages and blog posts. That closes the artifact-truth gap,
but it does not prove the generated copy actually uses the ticket facts well.

The next validation gap is content quality on the generated artifact: when a
live smoke writes an exported draft JSON, operators need a deterministic way to
spot obvious drift such as missing support-ticket context, stale benchmark
numbers, absent customer wording, or generated text that ignores the observed
ticket clusters.

This slice stays in the support-ticket provider lane and evaluates exported
drafts after generation. It does not change generation prompts, FAQ generation,
or hosted upload flow.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add an offline support-ticket generated-content evaluator for saved-draft
   export JSON files.
2. Support both `landing_page` exports and `blog_post` exports.
3. Check that the exported draft still has support-ticket context in the
   expected location.
4. Check generated text for concrete signs that it used the source facts:
   source counts where present, top ticket clusters, customer questions or
   wording examples, and support-ticket/FAQ framing.
5. Flag stale generic benchmark numbers from prior smoke drift, including
   `186`, `78`, and `42%`, when those values are not present in the source
   context.
6. Add focused tests with representative passing and failing landing/blog
   export artifacts.

### Files touched

- `plans/PR-Support-Ticket-Generated-Content-Quality-Eval.md`
- `scripts/evaluate_support_ticket_generated_content.py`
- `tests/test_evaluate_support_ticket_generated_content.py`

## Mechanism

The evaluator will read a JSON export produced by
`scripts/smoke_content_ops_live_generation.py --export-saved-draft`. The caller
will pass `--output landing_page` or `--output blog_post`.

For landing pages, the evaluator reads source context from
`rows[0].metadata.source_context` and generated text from title, slug, hero,
sections, CTA, and meta fields.

For blog posts, the evaluator reads source context from `rows[0].data_context`
and generated text from title, description, content, tags, and charts.

The command returns machine-readable JSON with:

- `ok`
- `errors`
- `warnings`
- `checks`
- `source_context_summary`

Required failures should cover missing export rows, missing support-ticket
context, stale benchmark numbers that are not source-backed, and generated text
that does not mention any observed cluster/question/customer wording. Softer
quality concerns can be warnings so this first version helps inspection without
pretending to be a full editorial judge.

## Intentional

- No LLM judge. This is deterministic and cheap enough for local/live-smoke
  validation.
- No prompt changes. If the evaluator exposes quality problems, prompt or
  generator fixes should be follow-up slices with concrete evidence.
- No FAQ generator or FAQ article evaluation. FAQ output remains owned by the
  parallel FAQ session.
- No hosted route changes. This reads export JSON files produced by existing
  smoke tooling.

## Deferred

- Future PR: wire this evaluator into the live smoke command as an optional
  `--evaluate-generated-content` flag after the standalone evaluator is proven.
- Future PR: use real exported CBFS/support-ticket artifacts as regression
  fixtures if we can commit safe redacted samples.
- Future PR: prompt/generator adjustments if the evaluator finds consistent
  content drift in live Haiku outputs.
- Parked hardening: none added by this slice. Existing `FAQSCALE-1` remains
  owned by `content-ops/faq-generation-scale`.

## Verification

- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - 9 passed.
- Py compile for `scripts/evaluate_support_ticket_generated_content.py` and
  `tests/test_evaluate_support_ticket_generated_content.py`
  - Passed.
- Local PR review wrapper
  - Passed.
- Review comment fix: stale benchmark numbers now match whole tokens, so
  larger values like `2186` do not collide with stale `186`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~110 |
| Evaluator script | ~415 |
| Tests | ~225 |
| **Total** | **~750** |

This exceeds the 400 LOC soft target because the slice needs both asset shapes,
CLI behavior, and failure coverage to be useful. The evaluator remains
standalone and does not touch production routes or generation code.
