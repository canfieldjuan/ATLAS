# PR-Content-Ops-Support-Ticket-Copy-Policy

## Why this slice exists

PR #606 added CFPB complaint rows as support-ticket-like source evidence and
fixed the offline deterministic generator so the smoke does not claim the
target account is evaluating the complained-about vendor. The packaged LLM
prompts still mostly describe review/source-row evidence as market evidence,
and landing pages have no source-row copy policy at all. Real provider output
needs the same support-ticket guard as the deterministic smoke.

## Scope (this PR)

1. Add support-ticket/complaint source-row framing to packaged campaign,
   report, sales brief, and landing page prompts.
2. Lock the policy in skill-registry tests.
3. Remove the merged CFPB coordination row while claiming this slice.

### Files touched

- `extracted_content_pipeline/skills/digest/b2b_campaign_generation.md`
- `atlas_brain/skills/digest/b2b_campaign_generation.md`
- `extracted_content_pipeline/skills/digest/report_generation.md`
- `extracted_content_pipeline/skills/digest/sales_brief_generation.md`
- `extracted_content_pipeline/skills/digest/landing_page_generation.md`
- `tests/test_extracted_campaign_skill_registry.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Support-Ticket-Copy-Policy.md`

## Mechanism

Each affected prompt gets an explicit source-row evidence rule:

```text
support_ticket / complaint / case / conversation rows are service evidence,
not buying-intent evidence.
```

The rule tells the LLM to use service framing such as "support evidence points
to..." and to avoid claims that the target account is evaluating, buying,
switching, or considering a vendor unless account-specific CRM, call, or
meeting evidence supports that claim.

## Intentional

- This is prompt policy only. The source adapter and CFPB exporter already
  provide the `source_type` metadata needed by the prompts.
- `b2b_campaign_generation.md` is synced from the Atlas canonical skill file,
  so the Atlas source is edited and the extracted copy is refreshed by sync.
- Blog prompts stay out of scope because the current blog path is blueprint and
  review-intelligence oriented, not direct opportunity source-row generation.
- The merged CFPB coordination row is removed in this PR because #606 has
  landed.

## Deferred

- Prompt policy for future source families should be driven by real host
  exports or live provider failures, not speculative source taxonomy expansion.
- Shared Postgres smoke helper extraction remains a separate follow-up from
  PR #606.

## Verification

- `pytest tests/test_extracted_campaign_skill_registry.py -q` -> 12 passed.
- `python -m py_compile tests/test_extracted_campaign_skill_registry.py` -> passed.
- `grep -nP '[^\x00-\x7F]' tests/test_extracted_campaign_skill_registry.py` -> no matches.
- `bash scripts/local_pr_review.sh` -> pending after commit.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Prompt policy text | ~35 |
| Tests | ~35 |
| Plan + coordination | ~80 |
| **Total** | **~150** |
