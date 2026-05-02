# Remaining Productization Audit

Date: 2026-05-02

This audit follows the helper ownership pass that moved `_b2b_batch_utils`,
`_blog_matching`, `_campaign_sequence_context`, and `campaign_audit` into the
extracted content pipeline boundary.

## Current State

Standalone import debt is clean:

```bash
EXTRACTED_PIPELINE_STANDALONE=1 python scripts/audit_extracted_standalone.py --fail-on-debt
# Atlas runtime import findings: 0
```

The extracted runner is green after the helper pass:

```bash
EXTRACTED_PIPELINE_STANDALONE=1 bash scripts/run_extracted_pipeline_checks.sh
# 177 passed
```

The smoke import script now includes both campaign-core copied tasks after the
PR 1 and PR 2 seams made them importable. Direct import of all remaining
manifest-mapped Python files still shows two failing surfaces:

| Module | Standalone import | First blocker |
| --- | --- | --- |
| `autonomous.tasks.b2b_campaign_generation` | Passes | PR 2 seams present |
| `autonomous.tasks.b2b_vendor_briefing` | Passes | PR 1 seams present |
| `autonomous.tasks._b2b_pool_compression` | Fails | missing `autonomous.tasks._b2b_witnesses` |
| `autonomous.tasks.competitive_intelligence` | Fails | missing `services.brand_registry` |

Everything else still mapped from Atlas imports in standalone mode, but many
files remain Atlas-shaped and should not be product-owned as-is.

## Remaining Mapped Python Surface

| File | Lines | Classification |
| --- | ---: | --- |
| `_b2b_shared.py` | 19,875 | Monolith; split by required product seams |
| `b2b_blog_post_generation.py` | 9,613 | Blog product surface, not campaign-core |
| `b2b_campaign_generation.py` | 6,043 | Campaign-core copied task; currently not importable |
| `b2b_vendor_briefing.py` | 3,222 | Campaign/email adjacent; currently not importable |
| `_b2b_pool_compression.py` | 2,319 | Reasoning pool; blocked on witness extraction |
| `_b2b_reasoning_contracts.py` | 1,773 | Reasoning policy; importable but large |
| `_b2b_synthesis_reader.py` | 1,767 | Reasoning read model; importable but large |
| `blog_post_generation.py` | 1,758 | Consumer/blog sidecar |
| `competitive_intelligence.py` | 1,455 | Consumer intelligence sidecar; currently not importable |
| `_b2b_cross_vendor_synthesis.py` | 1,063 | Reasoning/synthesis helper; importable |
| `_b2b_specificity.py` | 772 | Specificity policy; importable |
| `complaint_analysis.py` | 527 | Consumer/complaint sidecar |
| `complaint_enrichment.py` | 493 | Consumer/complaint sidecar |
| `article_enrichment.py` | 431 | Consumer/blog sidecar |
| `complaint_content_generation.py` | 348 | Consumer/complaint sidecar |

## Missing Seams For Campaign-Core Imports

`b2b_campaign_generation.py` had these missing top-level dependencies, now
covered by product-owned compatibility seams:

- `services.b2b.account_opportunity_claims`
- `services.campaign_reasoning_context`
- `services.campaign_quality`
- `services.vendor_target_selection`
- `autonomous.visibility`
- `autonomous.tasks.campaign_suppression`

`b2b_vendor_briefing.py` had these missing top-level dependencies, now covered
by product-owned compatibility seams:

- `services.campaign_sender`
- `services.vendor_target_selection`
- `templates.email.vendor_briefing`
- `autonomous.tasks.campaign_suppression`

`_b2b_pool_compression.py` has one missing dependency:

- `autonomous.tasks._b2b_witnesses`

`competitive_intelligence.py` has one missing dependency:

- `services.brand_registry`

## Productization Interpretation

The sellable campaign product already has a cleaner product-owned spine:

- `campaign_ports.py`
- `campaign_generation.py`
- `campaign_send.py`
- `campaign_sequence_progression.py`
- `campaign_suppression.py`
- `campaign_sender.py`
- `campaign_webhooks.py`
- `campaign_analytics.py`
- `campaign_postgres.py`
- `campaign_llm_client.py`

The copied Atlas task files are still useful as extraction references, but the
next work should not make `b2b_campaign_generation.py` or `_b2b_shared.py`
product-owned in one move. The correct next move is to add narrow compatibility
seams that let the copied campaign task modules import, then migrate behavior
into the product-owned spine in smaller slices.

## Recommended Sequence

### PR 1: Vendor Briefing Import Seams

Status: implemented. `b2b_vendor_briefing.py` imports in standalone mode, and
the smoke script now covers it.

Goal: make `b2b_vendor_briefing.py` import in standalone mode without claiming
the copied 3,222-line task is product-owned.

Add minimal product-owned or compatibility modules:

- `services/vendor_target_selection.py`
  - Start with `dedupe_vendor_target_rows(...)`.
  - Keep it deterministic and data-only.
- `autonomous/tasks/campaign_suppression.py`
  - Compatibility wrapper over product `campaign_suppression.py`.
  - Provide copied task names `is_suppressed(...)`,
    `assign_recipient_to_sequence(...)`.
- `services/campaign_sender.py`
  - Compatibility wrapper around product `campaign_sender.create_campaign_sender`.
- `templates/email/vendor_briefing.py`
  - Extract the email renderer or provide a compatibility shim if the full
    template is not required for import-only readiness.

Acceptance criteria:

- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"`
- Local tests for each compatibility shim.

This slice may expand the smoke script to include `b2b_vendor_briefing.py`
once the direct import passes. Keep `b2b_campaign_generation.py` out of the
smoke list until PR 2 handles its separate missing imports.

### PR 2: Campaign Generation Import Seams

Status: implemented. `b2b_campaign_generation.py` imports in standalone mode,
and the smoke script now covers both campaign-core copied tasks.

Goal: make `b2b_campaign_generation.py` import in standalone mode without
claiming the copied 6,043-line task is product-owned.

Add:

- `services/b2b/account_opportunity_claims.py`
  - `account_opportunity_source_review_count(...)`
  - `build_account_opportunity_claim(...)`
  - `serialize_product_claim(...)`
- `services/campaign_reasoning_context.py`
  - `campaign_reasoning_scope_summary(...)`
  - `campaign_reasoning_atom_context(...)`
  - `campaign_reasoning_delta_summary(...)`
- `services/campaign_quality.py`
  - `campaign_quality_revalidation(...)`
- `autonomous/visibility.py`
  - Route `emit_event(...)` / `record_attempt(...)` through
    `pipelines.notify` or safe no-op defaults.

After this PR, expand `scripts/smoke_extracted_pipeline_imports.py` to include:

- `extracted_content_pipeline.autonomous.tasks.b2b_campaign_generation`
- `extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing`

Acceptance criteria:

- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_campaign_generation"`
- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"`
- New tests for each seam's local behavior.
- Smoke import script expanded to include both modules.

### PR 3: Pool Compression Decision

Goal: decide whether `_b2b_pool_compression.py` is part of the sellable campaign
product or a reasoning-product dependency.

The smoke script can add campaign-core modules after PR 2, but should not add
`_b2b_pool_compression.py` until this decision is made:

- `_b2b_pool_compression` only after witness extraction, or explicitly leave it
  out with a TODO in the smoke script.

Options:

1. Extract a minimal `_b2b_witnesses.py` compatibility helper and keep
   `_b2b_pool_compression.py` importable as a copied reasoning dependency.
2. Keep pool compression out of campaign-core and add a product-owned interface
   that accepts already-compressed reasoning/witness context from the host.

Recommendation: option 2 unless a campaign-core test requires pool compression
directly. It keeps the campaign product from owning the whole B2B reasoning
stack.

## Explicit Deferrals

These remain outside the immediate campaign-core import path:

- `competitive_intelligence.py` and `services.brand_registry`
  - Consumer intelligence sidecar, not required for the email/campaign product.
- Consumer/blog generation tasks
  - `blog_post_generation.py`, `article_enrichment.py`,
    `complaint_*` modules.
- Whole-file ownership of `_b2b_shared.py`
  - Too large to own as a unit; split only the helpers needed by product tests.
- Whole-file ownership of `b2b_blog_post_generation.py`
  - Blog/content product, separate from the campaign delivery product.

## Next Concrete Slice

Start with PR 1: vendor briefing import seams. It is smaller than
`b2b_campaign_generation.py`, exercises the same delivery-side compatibility
surface (`campaign_sender`, suppression, template rendering), and avoids the
campaign reasoning/claim seams until the next PR.

Acceptance criteria:

- `EXTRACTED_PIPELINE_STANDALONE=1 python -c "import extracted_content_pipeline.autonomous.tasks.b2b_vendor_briefing"`
- New tests for each shim's local behavior.
- Full extracted pipeline checks still pass.
