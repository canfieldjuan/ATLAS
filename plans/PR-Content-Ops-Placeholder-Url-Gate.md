# PR-Content-Ops-Placeholder-Url-Gate

## Why this slice exists

The first live provider proof after the bridge-mode import fix generated and
persisted G2-grounded campaign drafts, but the LLM invented `example.com` CTA
and article URLs because the imported review-source row did not provide a
real selling asset URL. The evidence claims were grounded; the links were not.

AI Content Ops should fail closed on placeholder URLs before saving drafts.
This keeps live outputs demo-safe even when a host has not yet supplied a
real asset, blog post, or booking URL.

## Scope (this PR)

1. Add a small campaign-generation guard for placeholder URLs in generated
   `body` and `cta` fields.
2. Skip affected drafts with a clear `placeholder_url` error instead of
   persisting them.
3. Add focused service tests for body, CTA, scheme-less, punctuation-terminated,
   subdomain, and email-address false-positive behavior.
4. Register this slice in the coordination ledger.

### Files touched

- `extracted_content_pipeline/campaign_generation.py`
- `tests/test_extracted_campaign_generation.py`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Placeholder-Url-Gate.md`

## Mechanism

After campaign response parsing returns a candidate draft and before
quality revalidation / persistence, `CampaignGenerationService.generate()`
checks generated link-bearing fields for known placeholder hosts. If a
placeholder URL is present, the service increments `skipped`, records
`{"reason": "placeholder_url"}`, and continues without saving that draft.

## Intentional

- This is a hard runtime guard, not only a prompt edit. Prompt wording can
  reduce bad generations, but the persistence boundary is where fake links
  must be blocked.
- This does not require quality revalidation to be enabled. Placeholder URLs
  are unsafe regardless of the optional evidence-quality pass.
- The guard targets placeholder URLs, not all missing links. A draft with no
  URL can still be valid if the host did not supply an asset.

## Deferred

- Host-configured selling asset URLs for live smoke inputs remain a follow-up.
  That will let the same G2 live proof pass without fabricated links while
  preserving useful CTAs.
- Broader URL policy for non-campaign asset types is deferred until each
  live-output smoke is exercised.

## Verification

Local checks:

```bash
pytest tests/test_extracted_campaign_generation.py
# 44 passed

python -m py_compile extracted_content_pipeline/campaign_generation.py tests/test_extracted_campaign_generation.py
# passed

bash scripts/validate_extracted_content_pipeline.sh
# passed

python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
# passed

python scripts/audit_extracted_standalone.py --fail-on-debt
# passed

bash scripts/check_ascii_python.sh
# passed

python scripts/smoke_content_ops_review_source_postgres.py \
  --source g2 --vendor Slack --limit 1 \
  --account-id content_ops_live_g2_gate_20260519_125733 \
  --llm pipeline --min-drafts 2 --json
# expected fail-closed: 0 generated, 2 skipped with reason=placeholder_url

python scripts/export_extracted_campaign_drafts.py \
  --account-id content_ops_live_g2_gate_20260519_125733 --format json
# count=0
```

## Estimated diff size

| File | LOC churn (approx) |
|---|---:|
| `extracted_content_pipeline/campaign_generation.py` | 20 |
| `tests/test_extracted_campaign_generation.py` | 178 |
| `docs/extraction/coordination/inflight.md` | 3 |
| `plans/PR-Content-Ops-Placeholder-Url-Gate.md` | 101 |
| **Total** | **302** |

Below the 400 LOC review budget.
