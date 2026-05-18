# PR: Content Ops Review Evidence Asset Framing

## Why this slice exists

PR #593 fixed public review/source-row copy framing for campaign emails. Reports and sales briefs also consume normalized opportunity rows, so the same third-party review evidence can reach those generated assets. Their packaged prompts currently say to structure output from opportunity evidence but do not tell the model to avoid target-account intent overclaims when that evidence is public review/source-row material.

## Scope (this PR)

1. Add review/source-row market-framing guidance to the packaged report prompt.
2. Add the same guidance to the packaged sales-brief prompt.
3. Add packaged skill-registry assertions so the prompt contract remains visible in tests.

### Files touched

- `extracted_content_pipeline/skills/digest/report_generation.md`
- `extracted_content_pipeline/skills/digest/sales_brief_generation.md`
- `tests/test_extracted_campaign_skill_registry.py`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Evidence-Asset-Framing.md`

## Mechanism

- Keep the rule prompt-only because report and sales brief generation already pass the normalized opportunity payload through unchanged.
- Instruct the model to treat `source_type: "review"` or source-row evidence as third-party market evidence.
- Tell the model not to say the target account said, did, evaluated, or intends something unless account-specific reasoning or CRM evidence supports it.

## Intentional

- This does not change report or sales-brief runtime code.
- This does not touch campaign email framing, which shipped in PR #593.
- This does not touch landing pages or blog posts because their inputs are marketing-campaign and blueprint shaped, not normalized opportunity rows.

## Deferred

- Real provider-output quality review with pipeline LLM credentials.
- Source-type-specific copy policies for calls, meetings, CRM notes, support tickets, surveys, and contracts.
- Trustpilot v4 phrase-metadata re-enrichment.

## Verification

- Focused packaged skill registry tests -> 8 passed.
- Python compile check for `tests/test_extracted_campaign_skill_registry.py` -> passed.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Report prompt | ~5 |
| Sales brief prompt | ~5 |
| Tests | ~18 |
| Coordination and plan | ~61 |
| Total | ~91 |
