# AI Content Ops Control-Surface Preview API

Date: 2026-05-07

This document covers the pre-generation control surface for AI Content Ops.
It is intentionally separate from generation, sending, approval, and storage.
The point is to let a host UI ask what can run, what it will cost, and what is
missing before the product burns tokens like a tiny ceremonial bonfire.

## Why This Exists

The product can generate multiple content assets: email campaigns, blog posts,
reports, landing pages, sales briefs, and signal extraction. Not every run
should produce every asset. The UI needs a backend-owned contract for output
selection, presets, budget checks, missing input checks, and implementation
status.

The control-surface preview is that contract.

## Host Mount

Mount the router in the host FastAPI app and inject the same auth dependency the
host uses for other Content Ops routes:

```python
from fastapi import Depends

from extracted_content_pipeline.api.control_surfaces import (
    create_content_ops_control_surface_router,
)


app.include_router(
    create_content_ops_control_surface_router(
        dependencies=[Depends(require_content_ops_user)],
    )
)
```

The router is preflight-only. It does not call an LLM, read or write Postgres,
or start an autonomous task.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/content-ops/control-surfaces` | List output types, presets, required inputs, implementation status, cost estimates, and ingestion profiles. |
| `POST` | `/content-ops/preview` | Validate a requested preset/output selection and return cost, missing inputs, warnings, and blocked outputs. |

## Output Catalog

Current output ids:

| Output | Status | Notes |
|---|---|---|
| `email_campaign` | Implemented | Existing campaign draft path. |
| `blog_post` | Implemented | Existing content asset path. |
| `report` | Implemented | Structured report draft path. |
| `landing_page` | Not implemented | Included in catalog but blocked by default. |
| `sales_brief` | Not implemented | Included in catalog but blocked by default. |
| `signal_extraction` | Not implemented | Included in catalog but blocked by default. |

Future outputs should be added to the catalog first, then exposed through
presets only after the implementation and quality gate exist. Yes, this is less
exciting than shipping a toggle that lies to users. That is the point.

## Presets

Current preset ids:

| Preset | Outputs | Purpose |
|---|---|---|
| `email_only` | `email_campaign` | Lowest-cost outreach draft run. |
| `intelligence_report` | `report` | Reference-backed report generation. |
| `content_marketing` | `blog_post`, `report` | Blog plus report from the same evidence base. |
| `lead_gen_campaign` | `email_campaign`, `landing_page` | Outreach plus landing page. Landing page remains gated until implemented. |
| `full_campaign` | `email_campaign`, `blog_post`, `report`, `landing_page`, `sales_brief` | Full bundle. Expensive and partially gated. |

## Preview Payload

```json
{
  "outputs": ["email_campaign", "report"],
  "limit": 2,
  "max_cost_usd": 2.0,
  "inputs": {
    "target_account": "Acme",
    "offer": "Churn intelligence audit",
    "opportunity_id": "opp_123"
  },
  "ingestion_profile": "domain_specific",
  "require_quality_gates": true
}
```

A caller may use either `outputs` or `preset`. Explicit `outputs` win over a
preset.

## Preview Response

```json
{
  "can_run": true,
  "outputs": ["email_campaign", "report"],
  "estimated_cost_usd": 1.46,
  "missing_inputs": [],
  "blocked_outputs": [],
  "warnings": [],
  "normalized_request": {
    "target_mode": "b2b",
    "preset": null,
    "outputs": ["email_campaign", "report"],
    "limit": 2,
    "max_cost_usd": 2.0,
    "ingestion_profile": "domain_specific",
    "require_quality_gates": true,
    "allow_unimplemented_outputs": false
  }
}
```

Treat `can_run=false` as a hard stop for generation controls. The UI can still
show the selected plan, but it should not enable the generate button until
`missing_inputs`, `blocked_outputs`, and budget warnings are resolved.

## UI Contract

The UI should call `/content-ops/preview` whenever any of these change:

- selected preset
- selected outputs
- limit
- max budget
- source inputs
- ingestion profile

The first UI should be boring in the useful way:

1. choose a preset or outputs
2. add required inputs
3. show estimated cost and warnings
4. disable generation until `can_run=true`
5. pass the normalized request into the generation endpoint added in a later slice

Do not put prompt settings, retrieval knobs, model choices, or chunking strategy
in the first UI. Those are backend controls until there is a real product reason
to expose them.

## Cost Notes

The current cost estimates are conservative placeholders. They exist to create
the product contract now. Replace the placeholder estimates with real token or
provider accounting later without changing the UI contract.

## Ingestion Profiles

Supported values:

| Profile | Meaning |
|---|---|
| `domain_specific` | Default path. Use product/domain assumptions and existing evidence model. |
| `manual` | Caller supplies the inputs directly. |
| `existing_evidence` | Caller references already-ingested evidence or opportunities. |

Do not expose arbitrary ingestion-pipeline configuration in the first UI. Domain
specific ingestion is the sane default until the product has enough repeated use
to justify more knobs.
