# Support Ticket Outcome Live Validation

## Why this slice exists

#970 added the stricter `support_ticket_outcome_claims_grounded` evaluator check
and prompt guidance after saved Haiku artifacts proved generated support-ticket
copy could make unsupported outcome promises. Fresh live Haiku runs through the
real Content Ops landing and blog paths then exposed softer unsupported outcome
phrases that the first detector missed. This slice turns those live false greens
into source checks, tests, and prompt guidance.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Functional validation

1. Rerun support-ticket landing-page generation through the live Content Ops
   smoke using the Haiku override.
2. Rerun support-ticket blog-post generation through the live Content Ops smoke
   using the Haiku override.
3. Export saved drafts and run generated-content evaluation on both.
4. Broaden the support-ticket outcome detector when live/manual review finds
   unsupported outcome claims around support volume, retention guarantees,
   capacity, instant results, no-ticket promises, or unsupported time savings
   that still pass.
5. Tighten landing/blog prompt guidance so new drafts avoid those claims before
   the evaluator has to block them.
6. Document pass/fail results, saved draft ids, readiness states, and remaining
   product-copy blockers.
7. Run one Sonnet comparison after review to separate model-specific behavior
   from the support-ticket generation contract problem.

### Files touched

- `plans/PR-Support-Ticket-Outcome-Live-Validation.md`
- `docs/extraction/validation/support_ticket_outcome_live_validation_2026-05-25.md`
- `extracted_content_pipeline/support_ticket_generated_content_eval.py`
- `tests/test_evaluate_support_ticket_generated_content.py`
- `atlas_brain/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/blog_post_generation.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `ATLAS-HARDENING.md`

## Mechanism

This slice uses the existing live smoke harness:

```bash
python scripts/smoke_content_ops_live_generation.py \
  --support-ticket-csv \
  --env-file /home/juan-canfield/Desktop/Atlas/.env \
  --env-file /home/juan-canfield/Desktop/Atlas/.env.local \
  --env-file tmp/support_ticket_live_haiku_eval_20260525/haiku.env \
  --export-saved-draft <draft-path> \
  --output-result <result-path> \
  --evaluate-generated-content \
  --json
```

The `haiku.env` file pins OpenRouter reasoning generation to the Claude Haiku
family so this validation does not use Sonnet for test generation.

A follow-up comparison used the same smoke harness with an OpenRouter Sonnet
override (`anthropic/claude-sonnet-4-5`) to check whether the remaining failure
shape was model-specific.

## Intentional

- This began as a validation slice, but the live run exposed detector false
  greens that must be fixed inline because the gate is load-bearing.
- The support-ticket CSV remains the packaged example file so results are
  comparable to the prior validation runs.
- No FAQ generator internals or standalone FAQ article shape changes.

## Deferred

- Parked hardening: support-ticket FAQ drafts can invent procedural answer steps
  when the uploaded tickets include customer questions but no support-resolution
  fields. This is tracked in `ATLAS-HARDENING.md` with owner/session
  `content-ops/support-ticket-outcome-live-validation`.
- Future PR: add a first-class "resolution evidence present" contract so the
  generator can emit review-needed FAQ answer placeholders instead of concrete
  product steps when the ticket export lacks support resolutions.

## Verification

- Landing live smoke with `--evaluate-generated-content`: passed and saved
  draft `030c08da-3f01-41f6-afa3-474d600bfa6d`.
- Blog live smokes exposed multiple false greens; stale drafts now fail after
  detector broadening. The latest known saved draft before the final hardening
  was `66fc02f9-82cf-400c-ad31-b5491ee3647f`, and rerunning the evaluator after
  hardening fails it for unsupported instant-result, support-capacity,
  no-ticket, and support-volume claims while allowing neutral churn/retention
  context and disclaimers.
- Sonnet comparison with
  `acct_support_ticket_outcome_sonnet_eval_20260525_blog1` did not save a draft:
  it failed length/GEO checks and copied cautionary prompt guidance into the
  article. That supports the contract diagnosis: model choice changes the leak
  shape but does not supply missing outcomes, dates, or resolutions.
- `python -m pytest tests/test_evaluate_support_ticket_generated_content.py -q`
- `bash scripts/local_pr_review.sh`

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan/doc/hardening | ~220 |
| Evaluator/tests | ~445 |
| Prompt guidance | ~15 |
| **Total** | **~680** |
