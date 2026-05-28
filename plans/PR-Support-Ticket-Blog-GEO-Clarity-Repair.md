# PR: Support-Ticket Blog GEO Clarity Repair

## Why this slice exists

PR-Support-Ticket-SaaS-Demo-Blog-Accepted-Fixture recorded that the post-contract
36-row SaaS demo blog retry no longer failed on support-ticket outcome claims,
but it still failed before save on `geo_entity_clarity_missing` after two repair
attempts. The failed draft already had "Support Ticket FAQ Gaps" in the title
and target keyword, so the issue was not simply missing entity text.

The source gap is in the repair guidance. The quality gate returns
`geo_entity_clarity_missing` both when the entity is unclear and when the draft
contains vague H2s such as `## Summary`. The current repair guidance only tells
the model to repeat the target keyword, so repair can keep a vague H2 and fail
again.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-provider
Slice phase: Production hardening

1. Promote the parked GEO clarity hardening item from `HARDENING.md`.
2. Update blog quality repair guidance for `geo_entity_clarity_missing` so it
   also replaces vague H2s with specific question/answer headings.
3. Add a focused regression test for that repair guidance.
4. Do not run a new live Haiku retry in this source-fix slice.

### Files touched

- `plans/PR-Support-Ticket-Blog-GEO-Clarity-Repair.md` - Plan doc for this source fix.
- `HARDENING.md` - Remove the promoted GEO clarity parked item.
- `extracted_content_pipeline/blog_generation.py` - Expand GEO clarity repair guidance.
- `tests/test_extracted_blog_generation.py` - Assert repair guidance names vague H2 replacement.

## Mechanism

`_blog_quality_repair_guidance` already maps quality blocker codes into concrete
LLM repair instructions. This PR keeps the existing target-keyword instruction
for `geo_entity_clarity_missing`, and adds the missing instruction:

- replace vague H2s such as `Overview`, `Introduction`, `Conclusion`,
  `Summary`, `Final Thoughts`, and `Key Takeaways`;
- use specific question/answer headings that name the topic or target keyword.

That matches the quality gate's `_VAGUE_H2_RE` behavior without changing the
gate or weakening the GEO requirement.

## Intentional

- This does not change `extracted_quality_gate.blog_pack`; the gate is doing the
  right thing by blocking vague H2s.
- This does not run another live model call. The next validation slice should
  retry the SaaS demo blog path after this source fix lands.
- This does not alter the support-ticket outcome detector or FAQ ownership.

## Deferred

- Future PR: run one Haiku live retry against the 36-row SaaS demo blog path
  and commit an accepted fixture if it saves and passes.
- Future PR: add a scripted regression gate after an accepted fixture exists.
- Parked hardening:
  - LLM usage storage schema mismatch hides per-run cost telemetry.

## Verification

- Passed: python -m pytest tests/test_extracted_blog_generation.py -q (68 tests).
- Passed: bash scripts/validate_extracted_content_pipeline.sh.
- Passed: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline.
- Passed: python scripts/audit_extracted_standalone.py --fail-on-debt.
- Passed: bash scripts/check_ascii_python.sh.
- Passed: bash scripts/local_pr_review.sh --current-pr-body-file /tmp/support-ticket-blog-geo-clarity-repair-pr-body.md.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~80 |
| Repair guidance | ~8 |
| Test | ~4 |
| HARDENING cleanup | ~9 |
| **Total** | **~101** |
