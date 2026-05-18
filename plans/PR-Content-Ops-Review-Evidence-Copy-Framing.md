# PR: Content Ops Review Evidence Copy Framing

## Why this slice exists

A live G2 source-row smoke proved the export, inspection, and offline campaign generation path works, but the generated preview copy overstated third-party review evidence as direct target-account intent: "Acme Logistics appears to be weighing Slack." Public review rows should be framed as market evidence unless the opportunity carries account-specific reasoning.

## Scope (this PR)

1. Update the offline deterministic campaign LLM example to use market-level language when the opportunity evidence is sourced from public reviews.
2. Tighten the campaign generation skill rule so real LLM outputs follow the same policy for review/source-row evidence.
3. Add/adjust source-row CLI tests to lock the review-evidence copy contract.

### Files touched

- `extracted_content_pipeline/campaign_example.py`
- `atlas_brain/skills/digest/b2b_campaign_generation.md`
- `extracted_content_pipeline/skills/digest/b2b_campaign_generation.md`
- `tests/test_extracted_campaign_generation_example.py`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Review-Evidence-Copy-Framing.md`

## Mechanism

- Detect review-sourced evidence from the normalized opportunity evidence rows.
- Preserve existing account-specific copy for ordinary opportunity rows.
- Use neutral market framing for review-sourced rows: teams are reporting pain around the vendor; the target account can use that signal for outreach planning.

## Intentional

- This does not change ingestion normalization.
- This does not change LLM/provider wiring.
- This does not add a new source type or exporter.
- This does not make review sources visible in generated copy.

## Deferred

- Real provider-output quality review with pipeline LLM credentials.
- Broader source-type copy policies for calls, meetings, CRM notes, and support tickets.
- Trustpilot v4 phrase-metadata re-enrichment.

## Verification

- Focused campaign generation example tests -> 22 passed.
- Python compile check for `extracted_content_pipeline/campaign_example.py` and `tests/test_extracted_campaign_generation_example.py` -> passed.
- Live G2/Slack source-row offline generation smoke -> passed; body used market framing.
- `git diff --check` -> passed.
- `scripts/local_pr_review.sh` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Offline campaign example | ~31 |
| Campaign skill copy rule | ~6 |
| Tests | ~6 |
| Coordination and plan | ~63 |
| Total | ~108 |
