# AI Content Ops Control-Surface Preview API

Date: 2026-05-07

This document covers the control surface for AI Content Ops. Preview and plan
routes are intentionally separate from generation, sending, approval, and
storage. The optional execute route runs only when a host injects generation
services. The point is to let a host UI ask what can run, what it will cost,
and what is missing before the product spends tokens.

## Why This Exists

The product can generate multiple content assets: email campaigns, blog posts,
reports, landing pages, sales briefs, and signal extraction. Not every run
should produce every asset. The UI needs a backend-owned contract for output
selection, presets, budget checks, missing input checks, and implementation
status.

The control-surface preview is that contract. The generation plan endpoint is
the next layer: it turns a passing preview into deterministic service steps
without executing them.

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

Preview and plan routes do not call an LLM, read or write Postgres, or start an
autonomous task. `/execute` is disabled unless the host injects execution
services; those services own any LLM, database, repository, and sender policy.
All request bodies are validated by the router before reaching the planning
layer: unknown top-level fields are rejected, `limit` is bounded to 1-1000,
`max_cost_usd` must be positive when supplied, and `inputs` has conservative
size and nesting limits.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/content-ops/control-surfaces` | List output types, presets, required inputs, implementation status, execution-service readiness, cost estimates, and ingestion profiles. |
| `POST` | `/content-ops/preview` | Validate a requested preset/output selection and return cost, missing inputs, warnings, and blocked outputs. |
| `POST` | `/content-ops/plan` | Convert a previewable request into deterministic generation steps. Does not execute generation. |
| `POST` | `/content-ops/execute` | Execute a runnable plan through host-injected services. Disabled unless the host configures execution services. |

## Output Catalog

Current output ids:

| Output | Status | Reasoning | Notes |
|---|---|---|---|
| `email_campaign` | Implemented | `optional_host_context` | Existing campaign draft path. |
| `blog_post` | Implemented | `absent` | Blog-post generation service path. |
| `report` | Implemented | `optional_host_context` | Structured report draft path. |
| `landing_page` | Implemented | `optional_host_context` | Landing page generation service path. |
| `sales_brief` | Implemented | `optional_host_context` | Sales brief generation service path. |
| `signal_extraction` | Implemented | `absent` | Deterministic source-row normalization into campaign opportunities; no LLM call. |

Future outputs should be added to the catalog first, then exposed through
presets only after the implementation and quality gate exist. Yes, this is less
exciting than shipping a toggle that lies to users. That is the point.
`optional_host_context` means the output can consume precomputed reasoning from
a host or separate reasoning product, but does not run synthesis internally.
`absent` means the output does not use the reasoning-provider path.

## Presets

Current preset ids:

| Preset | Outputs | Purpose |
|---|---|---|
| `email_only` | `email_campaign` | Lowest-cost outreach draft run. |
| `intelligence_report` | `report` | Reference-backed report generation. |
| `content_marketing` | `blog_post`, `report` | Blog plus report from the same evidence base. |
| `lead_gen_campaign` | `email_campaign`, `landing_page` | Outreach plus landing page. |
| `full_campaign` | `email_campaign`, `blog_post`, `report`, `landing_page`, `sales_brief` | Full generated-content bundle. |

The catalog endpoint exposes both `estimated_unit_cost_usd` and
`estimated_retry_adjusted_unit_cost_usd`. Use the retry-adjusted value for
budget UI and the preview response as the authoritative run estimate.

## Preview Payload

```json
{
  "outputs": ["email_campaign", "report"],
  "limit": 2,
  "max_cost_usd": 3.0,
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
  "estimated_cost_usd": 2.92,
  "missing_inputs": [],
  "blocked_outputs": [],
  "warnings": [],
  "normalized_request": {
    "target_mode": "vendor_retention",
    "preset": null,
    "outputs": ["email_campaign", "report"],
    "limit": 2,
    "max_cost_usd": 3.0,
    "ingestion_profile": "domain_specific",
    "require_quality_gates": true,
    "allow_unimplemented_outputs": false
  }
}
```

Treat `can_run=false` as a hard stop for generation controls. The UI can still
show the selected plan, but it should not enable the generate button until
`missing_inputs`, `blocked_outputs`, and budget warnings are resolved.
`estimated_cost_usd` is conservative: generated assets default to one parse
retry, so preview budgets include the worst-case retry attempt count.

> **Upgrade note (breaking):** Prior to 2026-05-08, `estimated_cost_usd`
> reflected a single LLM call per output. It now reflects worst-case retry
> attempts (default: 2 calls per generated asset). Operators with existing
> `max_cost_usd` budgets should multiply their previous limit by
> `default_parse_retry_attempts + 1` (default: x2).

## Plan Payload

`POST /content-ops/plan` accepts the same payload as `/content-ops/preview`.

## Plan Response

```json
{
  "can_execute": true,
  "target_mode": "vendor_retention",
  "limit": 2,
  "steps": [
    {
      "output": "email_campaign",
      "runner": "CampaignGenerationService.generate",
      "status": "runnable",
      "config": {
        "skill_name": "digest/b2b_campaign_generation",
        "channels": ["email_cold", "email_followup"],
        "limit": 2,
        "max_tokens": 1200,
        "temperature": 0.4,
        "quality_revalidation_enabled": true,
        "quality_prompt_proof_term_limit": 5,
        "parse_retry_attempts": 1,
        "parse_retry_response_excerpt_chars": 800
      },
      "reason": ""
    },
    {
      "output": "report",
      "runner": "ReportGenerationService.generate",
      "status": "runnable",
      "config": {
        "skill_name": "digest/report_generation",
        "default_report_type": "vendor_pressure",
        "limit": 2,
        "max_tokens": 4096,
        "temperature": 0.3
      },
      "reason": ""
    }
  ],
  "preview": {
    "can_run": true,
    "outputs": ["email_campaign", "report"],
    "estimated_cost_usd": 2.92,
    "missing_inputs": [],
    "blocked_outputs": [],
    "warnings": []
  }
}
```

`can_execute` is stricter than `preview.can_run`. It only becomes true when the
preview passes and every selected output maps to a runnable service-shaped step.
`signal_extraction` is deterministic and offline once the host supplies a
configured `SignalExtractionService`; without that service, preview and plan can
pass, but `/execute` reports the output as not configured.

## Execute Route

`POST /content-ops/execute` accepts the same payload as `/preview` and `/plan`.
The route is opt-in: hosts must pass a `ContentOpsExecutionServices` provider
when creating the router. The product does not construct database handles, LLM
clients, repositories, or senders inside the control-surface API.

Before wiring real providers, hosts can run the offline execution smoke:

```bash
python scripts/smoke_extracted_content_ops_execution.py
python scripts/smoke_extracted_content_ops_execution.py --outputs email_campaign,report --json
python scripts/smoke_extracted_content_ops_execution.py --outputs signal_extraction --source-vendor HubSpot --source-max-text-chars 400 --json
```

The smoke uses injected deterministic services and exercises the same
`execute_content_ops_from_mapping(...)` seam as the API route. It does not
open network, database, sender, or LLM handles. The signal extraction command
validates the deterministic source-material-to-opportunity path through the
same execution seam.

Runnable outputs dispatch to:

| Output | Service method |
|---|---|
| `email_campaign` | `CampaignGenerationService.generate(...)` |
| `blog_post` | `BlogPostGenerationService.generate(...)` |
| `report` | `ReportGenerationService.generate(...)` |
| `landing_page` | `LandingPageGenerationService.generate(...)` |
| `sales_brief` | `SalesBriefGenerationService.generate(...)` |
| `signal_extraction` | `SignalExtractionService.generate(...)` |

Non-executable plans return HTTP 400 with the blocked execution result. Missing
or failing execution/scope providers return HTTP 503. Service-level failures are
sanitized at the API boundary: internal exception messages are replaced with
stable reason codes. Partial executions return HTTP 207 with the execution
result in the response body; fully failed executions return HTTP 502 if the
executor reports `status="failed"`.

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
5. call `/content-ops/plan` to show the concrete steps before generation
6. pass the normalized request into the generation endpoint added in a later slice

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
