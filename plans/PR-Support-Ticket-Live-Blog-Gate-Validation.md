# Support Ticket Live Blog Gate Validation

## Why this slice exists

PR #961 moved support-ticket generated-content evaluation into the blog
save-time quality path. The deterministic route tests prove the gate can block
fabricated support-ticket claims, but the next required proof is a live Haiku
blog run through the real Content Ops smoke: support-ticket CSV input provider,
blog blueprint seeding, pipeline-routed LLM, saved draft persistence, export,
and generated-content evaluation.

This slice stays in the support-ticket provider lane and validates the merged
gate before adding new product behavior. The live run exposed one more
truthfulness gap: an undated uploaded-ticket CSV was converted into "per week"
cadence language. That would be a false green, so this slice closes the
cadence detector and reruns the live smoke until the saved draft passes.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Run a live Haiku blog-post smoke with support-ticket CSV packaging, saved
   draft export, and generated-content evaluation enabled.
2. Add an evaluator check for unsupported recurring cadence language when the
   source period is only `Uploaded support tickets`.
3. Update the blog generation prompt so uploaded-ticket posts do not invent
   "per week", "weekly", "per month", or similar cadence language unless that
   cadence exists in the input context.
4. Record the live failing/passing draft ids and evaluator results in a
   validation document.
5. Fix the generated-content evaluator CLI wrapper so direct script invocation
   works without manually setting `PYTHONPATH`.

### Files touched

- `plans/PR-Support-Ticket-Live-Blog-Gate-Validation.md`
- `docs/extraction/validation/support_ticket_live_blog_gate_validation_2026-05-25.md`
- `extracted_content_pipeline/support_ticket_generated_content_eval.py`
- `scripts/evaluate_support_ticket_generated_content.py`
- `tests/test_evaluate_support_ticket_generated_content.py`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`

## Mechanism

The evaluator already treats neutral uploaded-ticket sources differently from
dated support-ticket windows. This slice adds a second uploaded-ticket-only
truthfulness check for recurring cadence terms such as `per week`, `weekly`,
`per month`, and `monthly`.

The check only applies when `source_period` is `Uploaded support tickets` or
`review_period` is `uploaded tickets`. It records an
`uploaded_ticket_cadence_truthful` check and emits a blocker when the generated
text invents cadence from an undated upload.

The blog prompt is updated at the mapped Atlas source and synced into the
extracted skill copy. The evaluator CLI wrapper inserts the repo root onto
`sys.path` before importing the package module, which makes
`python scripts/evaluate_support_ticket_generated_content.py ...` work from a
normal checkout.

## Intentional

- No FAQ generator changes. FAQ generation and standalone FAQ article shape
  remain owned by the parallel FAQ lane.
- No new landing-page behavior. The cadence check applies to landing-page and
  blog exports because both can use uploaded support-ticket context, but this
  validation run is specifically the blog path that #961 hardened.
- No Sonnet spend. The live runs use a temporary Haiku override env file.
- Cross-layer caller hints will be inspected by local review. The prompt file
  is mapped through the extracted manifest, so the Atlas source is edited and
  synced to the extracted target.

## Deferred

- Future PR: centralize support-ticket data-context marker constants if the
  smoke seed, executor injection, and evaluator marker names drift again.
- Future PR: broader generated-copy quality audit for qualitative retention
  promises once the deterministic truthfulness gates stay green.
- Parked hardening: none added. Root `HARDENING.md` was scanned; the active
  FAQ scale/backpressure entry belongs to the FAQ generation lane and is not
  required for this support-ticket blog validation.

## Verification

- Live Haiku blog smoke before cadence fix
  - Passed the old evaluator, saved draft
    `a90aeb06-8b71-4a02-b31a-4c329503885f`, but generated unsupported
    `per week` language from undated uploaded tickets.
- Updated evaluator against that saved draft
  - Failed as expected with
    `generated text claims a recurring cadence for an undated uploaded-ticket source: per week`.
- Live Haiku blog smoke after cadence fix
  - Passed; saved draft `ea0a3333-e952-4cbd-9f9b-57c9825b3470`;
    SEO/AEO ready; GEO ready; generated-content evaluation passed.
- Direct evaluator CLI against the final saved draft
  - Passed without manual `PYTHONPATH`.
- Direct evaluator CLI help output
  - Passed without manual `PYTHONPATH`.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
  - 21 passed.
- Combined pytest over `tests/test_smoke_content_ops_live_generation.py`,
  `tests/test_evaluate_support_ticket_generated_content.py`,
  `tests/test_extracted_blog_generation.py`, and
  `tests/test_extracted_content_ops_live_execute_harness.py`
  - 118 passed.
- Extracted manifest validation at
  `extracted/_shared/scripts/validate_extracted.sh extracted_content_pipeline`
  - Passed.
- Py compile for changed evaluator/script/test files
  - Passed.
- Local PR review script
  - Passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan and validation report | ~230 |
| Evaluator and CLI wrapper | ~55 |
| Prompt sync | ~2 |
| Tests | ~75 |
| **Total** | **~362** |
