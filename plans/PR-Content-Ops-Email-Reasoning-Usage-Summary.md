# PR: Content Ops Email Reasoning Usage Summary

## Goal

Expose whether email campaign generation actually consumed reasoning context,
not just whether a host reasoning provider was attached to the execution
bundle.

## Scope

- Add `reasoning_contexts_used` to `CampaignGenerationResult.as_dict()`.
- Count successful generated email campaign drafts whose prompt opportunity
  carried `campaign_reasoning_context`.
- Show the count in the Content Ops generated-asset summary when present.
- Update the frontend fixture and contract documentation.

## Non-Goals

- Do not expose the reasoning payload in the execution result.
- Do not change provider attachment or optional-host-context readiness badges.
- Do not add reasoning usage counters for blog, report, landing page, or sales
  brief generators in this slice.

## Verification

- `python -m py_compile extracted_content_pipeline/campaign_generation.py`
- `python -m pytest tests/test_extracted_campaign_generation.py -q`
- Frontend build from `atlas-intel-ui`
