# In-Flight PRs

Last updated: 2026-05-06T18:15Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-D21f, in flight) | PR-D21f: Wire MultiPassCampaignReasoningProvider into the campaign-operations FastAPI router (final D21 slice) | EDIT: `extracted_content_pipeline/api/campaign_operations.py` (+5 multi-pass config fields, mutually exclusive with single-pass; multi-pass branch in `_generation_reasoning_context`; readiness now reports `multi_pass_configured` / `multi_pass_ready` / `mode="multi_pass"`). EDIT: `tests/test_extracted_campaign_api_operations.py` (+5 tests, +4 readiness-shape updates). Default no-op preserves D21e behavior. Hosts that need typed knobs (FalsificationPolicy, ReasoningPack, OutputPolicy) construct the provider themselves and pass via `reasoning_context_provider` FastAPI dep. | claude-2026-05-03 | `extracted_content_pipeline/api/campaign_operations.py`; `tests/test_extracted_campaign_api_operations.py` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
