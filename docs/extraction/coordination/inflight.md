# In-Flight PRs

Last updated: 2026-05-06T18:07Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D21e, in flight) | PR-D21e: Wire validate_reasoning_output into MultiPassCampaignReasoningProvider (5/5 reasoning-core capabilities through the bridge) | EDIT: `extracted_content_pipeline/services/multi_pass_reasoning_provider.py` (+`output_policy` and `block_on_validation_failure` config knobs; validate after narrative-plan; surface report in `canonical_reasoning["validation"]`; pre-filter falsified claims when `drop_falsified=True`). EDIT: `tests/test_extracted_campaign_multi_pass_reasoning_provider.py` (+5 tests). Default no-op preserves D21d behavior. | claude-2026-05-03 | `extracted_content_pipeline/services/multi_pass_reasoning_provider.py`; `tests/test_extracted_campaign_multi_pass_reasoning_provider.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
