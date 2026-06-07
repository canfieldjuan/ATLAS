# PR: Content Ops Output Variations

## Why this slice exists

Marketers want *options to choose from*, not a single take-it-or-leave-it
draft. The content-ops engine generates exactly one draft per output today:
`execute_content_ops_request` (`extracted_content_pipeline/content_ops_execution.py:257`)
fans `asyncio.gather` over `plan.steps`, one step per output, one draft per
step. The existing `limit` field is not a variations knob -- it reads `limit`
distinct blueprints/topics (`blog_generation.py:466` reads `limit` rows from
`read_blog_blueprints`) and emits one post per *distinct topic*. That is
quantity of different assets, not alternative renderings of the same spec.

This slice adds genuine variations: from one run spec, produce N alternatives
of the same output that differ by *angle* (pain-led vs outcome-led vs
social-proof hook), filter out anything the safety gate hard-blocks, and return
the survivors together so the marketer picks the best in the existing review
queue. It is the first layer of the requested variations -> batching -> A/B arc;
batching and A/B are scoped in **Deferred**.

## Scope (this PR)

Ownership lane: content-ops/output-variations
Slice phase: Workflow/process

This is a plan-doc-only PR. The plan specs the implementation slice below; the
only tracked artifact in this PR is the plan doc.

1. Add a `variant_count` knob to the run request (`ContentOpsRequestModel`,
   `extracted_content_pipeline/api/control_surfaces.py`) and thread it through
   `request_from_mapping` into the `ContentOpsRequest` dataclass
   (`control_surfaces.py`). Cap it to the angle-catalogue size.
2. Add a small deterministic `VARIANT_ANGLES` catalogue (e.g. pain-led,
   outcome-led, social-proof-led, objection-handling, urgency-led).
   `variant_count` selects the first N angles -- deterministic, no RNG.
3. Add an optional `variant_angle: str | None = None` param to the three
   matrix generators' `generate()` (`blog_generation.py`,
   `landing_page_generation.py`, `sales_brief_generation.py`), injected into
   each generator's existing skill-prompt substitution context. `None`
   preserves today's behavior exactly (backward compatible).
4. Centralize the fan-out in the step-execution loop of
   `execute_content_ops_request`: when `variant_count > 1`, run the per-output
   dispatcher once per selected angle and aggregate the resulting drafts.
   Generators stay single-asset; the orchestrator does the looping.
5. Run the matching quality-gate evaluator on each variant
   (`evaluate_blog_post` in `extracted_quality_gate/blog_pack.py`,
   `evaluate_landing_page` in `extracted_quality_gate/landing_page_pack.py`, and
   the sales-brief path, each returning `QualityReport`). Use the
   gate as a **filter**: drop any variant whose `decision` is hard-BLOCK; keep
   PASS / WARN / APPROVAL_REQUIRED. If all N are blocked, return zero survivors
   plus a clear warning rather than shipping garbage.
6. Return the surviving variants together in the execution result, each tagged
   with its angle label and saved draft id, so the existing review queue surfaces
   them as siblings and the human approves one.
7. Cost: multiply per-output cost by `variant_count` through the existing
   `estimate_cost_usd` (`control_surfaces.py:401`); the existing `max_cost_usd`
   ceiling and account-period budget check (the `_evaluate_account_usage_budget`
   gate at the `/execute` route in
   `extracted_content_pipeline/api/control_surfaces.py`) reject over-budget runs.
   No new budget guardrail.

### Files touched

- `plans/PR-Content-Ops-Output-Variations.md`

## Mechanism

The variant lever is *angle*, not temperature. Temperature jitter yields the
same hook reworded; marketers want different selling angles. Each selected angle
is passed to the generator as `variant_angle` and injected into the skill prompt
the generator already renders, so variants differ in framing.

The three matrix generators are not uniform: `blog_generation.generate` already
takes `topic`/`data_context`, but `landing_page_generation.generate` takes a
`campaign: MarketingCampaign` and `sales_brief_generation.generate` takes
`default_brief_type` -- neither has a free-form context param. So the angle lever
is added as one uniform optional `variant_angle` param across the three
`generate()` signatures (an additive capability behind the existing port, not a
fork of any generator), each wiring it into its own prompt-substitution dict.
When `variant_angle is None` the substitution is a no-op and behavior is
identical to today.

Fan-out lives in one place -- the step loop in `execute_content_ops_request` --
rather than edited into each `_dispatch_*` handler. For a step whose output is in
the variation-eligible set, the loop calls the dispatcher `variant_count` times,
once per angle, and concatenates the per-call drafts into the step result.

"Choose the best" is a human decision, not an auto-rank. The quality gate is a
compliance gate (PASS/WARN/BLOCK), not a persuasiveness ranker -- clean
marketing copy mostly returns zero blockers, so ranking by blocker count is
non-discriminating. The gate therefore only filters out hard-blocked variants;
the marketer picks the winner from the survivors in the review queue.

## Intentional

- Quality gate is a **filter**, not a ranker. Drop hard-BLOCK variants; the
  human picks. Auto-ranking is explicitly out (see Deferred) because
  blocker-count ranking does not measure which ad converts.
- Variants differ by **angle** (prompt-level), threaded via a uniform additive
  `variant_angle` param. `variant_angle=None` is byte-for-byte today's behavior.
- **No persistent variant-group column.** Variants come back together in the run
  result and the existing review-queue status workflow handles approve-one;
  revisiting a past run's variant set is Deferred.
- `variant_count` is capped to the angle-catalogue size to bound cost/latency,
  and every variant is cost-counted through the existing budget seam.
- All-blocked is surfaced as an explicit warning (zero survivors), never an
  empty silent run.

## Deferred

- **Batching** ("several batches for different ads") -- a thin orchestration
  layer over this variant engine: run multiple specs/audiences in one batch,
  aggregate cost through the same `max_cost_usd`/account-budget seam, return a
  batch summary. Specced once variations lands.
- **Auto-ranking / recommend-best** -- an LLM-as-judge scoring variants on
  persuasiveness (not compliance) to surface a recommended pick. Build only when
  there is a validated pull beyond "give me options."
- **A/B testing** -- depends on infrastructure absent today: a public
  variant-serving path that assigns variant A vs B to visitors, plus
  conversion/metric capture and winner selection. Design-aware note: variants
  already carry stable draft ids + angle metadata, so they are A/B-ready when
  serving + analytics exist. Build-when-validated.
- **Persistent variant grouping** -- a `variant_group_id` column + a
  "revisit a past run's variants" view, if/when re-opening old runs is needed.
- **Generator-parity fallback** -- if the single slice exceeds the 400 LOC cap,
  land blog-only variant generation first and split landing_page +
  sales_brief angle-lever parity into an immediate follow-up.
- Parked hardening: none.

## Verification

Commands run for this plan-doc-only PR:

- Command: python scripts/audit_plan_code_consistency.py
  plans/PR-Content-Ops-Output-Variations.md -- PASS.
- Command: git diff --check -- PASS.
- Command: bash scripts/local_pr_review.sh --current-pr-body-file
  /tmp/pr-1268-body.md -- PASS.

The implementation slice this plan specs will add focused pytest coverage for
variant fan-out count, angle diversity, BLOCK filtering, all-blocked warning,
cost scaling, and budget-gate rejection; it will also run the extracted package
gauntlet before PR.

## Estimated diff size

Estimated: ~150 LOC. This is a small workflow/process slice whose only tracked
artifact is the implementation plan. The implementation slice it specs targets
~375 LOC (variant_count threading, angle catalogue, uniform `variant_angle`
param across three generators, centralized fan-out + BLOCK-filter, cost scaling,
and focused tests), under the 400 LOC soft cap; the generator-parity fallback in
Deferred is the release valve if it runs over.

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~150 |
| **Total** | **~150** |
