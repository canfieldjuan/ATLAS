# Standalone Productization Contract

The extracted package must be installable and runnable without the Atlas
monolith. The scaffold began as a staging mirror for extraction traceability;
the runtime import gate is now clean, but the package is not the customer
product until the minimal adapters are hardened and copied helper scope is
trimmed.

## Non-Negotiable Product Rule

Runtime product code must not import `atlas_brain`.

Allowed during extraction:

- `manifest.json` source paths that point at Atlas source-of-truth files.
- Documentation that references Atlas as the origin.
- Temporary compatibility shims in the staging scaffold.

Not allowed in the sellable module:

- `from atlas_brain... import ...`
- `import atlas_brain...`
- A package install that requires the Atlas repo on `PYTHONPATH`.
- API/task code that reaches directly into Atlas settings, DB pools, auth,
  visibility events, scheduled-task models, LLM registry, or B2B helper modules.

## Required Product Ports

Before the campaign system can be sold as its own module, Atlas infrastructure
must be replaced with product-owned interfaces:

- `CampaignRepository`: campaign, sequence, suppression, audit, analytics, and
  target persistence.
- `IntelligenceRepository`: account/vendor/category inputs used to build
  campaign blueprints.
- `LLMClient`: message completion, JSON cleanup, tracing metadata, batch replay.
- `SkillStore`: prompt contract lookup by name/version.
- `CampaignSender`: Resend/SES send abstraction plus provider message ids.
- `WebhookVerifier`: provider signature verification and event normalization.
- `AuditSink`: immutable campaign lifecycle events.
- `VisibilitySink`: progress/events for hosted dashboards.
- `TenantScope`: account ownership and permission filtering.
- `Clock`: deterministic send-window and delay calculations.

The first pass of these boundaries lives in
`extracted_content_pipeline/campaign_ports.py`. Refactors should move copied
campaign code toward those interfaces before adding more raw Atlas modules to
the manifest.

## Decoupling Order

1. Define product-local models, settings, and ports.
2. Move pure helpers first: sequence context, suppression decisions, quality
   validation, send-window logic, webhook event normalization, and analytics
   summaries.
3. Convert services to accept injected ports instead of importing Atlas
   settings, DB pools, LLM registry, or visibility helpers.
4. Convert tasks into orchestrators that receive a dependency bundle.
5. Convert APIs into an app/router factory that receives auth and repository
   adapters from the host app.
6. Keep Atlas adapters in a separate optional package or directory so Atlas can
   still run the product internally without contaminating the customer module.

## First Standalone Slice

`extracted_content_pipeline/campaign_suppression.py` is the first product-owned
campaign module in this scaffold. It implements normalized email/domain
suppression behavior against `SuppressionRepository` and does not import Atlas.
Use it as the pattern for the next slices: keep policy and orchestration in the
product package, and push persistence/provider concerns behind ports.

`extracted_content_pipeline/campaign_sequence_context.py` is the second
standalone slice. It keeps the sequence prompt/storage compaction behavior but
replaces Atlas settings reads with explicit `SequenceContextLimits`.

`extracted_content_pipeline/campaign_sender.py` is the third standalone slice.
It keeps Resend and SES provider behavior but uses explicit provider config and
the product `SendRequest`/`SendResult` dataclasses.

`extracted_content_pipeline/campaign_send.py` is the fourth standalone slice.
It orchestrates due sends through `CampaignRepository`,
`CampaignSuppressionService`, `CampaignSender`, `AuditSink`, and `Clock`.
Atlas-specific quality revalidation, fatigue SQL, sequence scheduling, and
visibility events remain later adapter/service work.

`extracted_content_pipeline/campaign_webhooks.py` is the fifth standalone
slice. It verifies Resend/Svix signatures, normalizes ESP webhook payloads into
the product `WebhookEvent` dataclass, records events through
`CampaignRepository`, and applies bounce/complaint/unsubscribe suppressions
through `CampaignSuppressionService` without importing Atlas API code.

`extracted_content_pipeline/campaign_analytics.py` is the sixth standalone
slice. It wraps analytics refresh through `CampaignRepository.refresh_analytics`
and reports success/failure through optional audit and visibility ports.

`extracted_content_pipeline/campaign_sequence_progression.py` is the seventh
standalone slice. It builds follow-up prompt context, selects the sequence skill,
parses generated JSON, and queues due follow-up steps through
`CampaignSequenceRepository`, `LLMClient`, `SkillStore`, and optional audit
ports.

`extracted_content_pipeline/campaign_generation.py` is the eighth standalone
slice. It reads campaign opportunities through `IntelligenceRepository`, prompts
through `SkillStore` and `LLMClient`, parses generated draft JSON, and persists
`CampaignDraft` rows through `CampaignRepository`.

`extracted_content_pipeline/campaign_llm_client.py` is the provider-routing
slice for LLM access. It adapts extracted LLM infrastructure services to the
product `LLMClient` port and exposes `PipelineLLMClientConfig` so hosts can
wire workload, OpenRouter model, and fallback behavior without importing Atlas
settings.

`extracted_content_pipeline/pipelines/notify.py` is the first product-owned
visibility slice. It preserves the copied task-facing
`send_pipeline_notification(...)` API but emits through the `VisibilitySink`
port when a host configures one. With no sink configured it remains a safe
no-op, so standalone task imports do not require Atlas notification services.

`extracted_content_pipeline/reasoning/archetypes.py` is the first product-owned
reasoning policy slice. It scores vendor evidence against deterministic churn
archetypes, returns thresholded matches, and exposes falsification conditions
without a database, LLM, or Atlas import.

`extracted_content_pipeline/reasoning/temporal.py` is the second product-owned
reasoning policy slice. It computes velocity, acceleration, long-term trends,
category percentiles, anomalies, recency weights, and evidence serialization
from a host-provided async `fetch` interface instead of Atlas helper imports.

`extracted_content_pipeline/reasoning/evidence_engine.py` is the third
product-owned reasoning policy slice. It evaluates conclusion gates, section
suppression gates, and confidence labels from built-in product defaults or an
optional host-provided JSON/YAML evidence map.

## Readiness Gate

Run:

```bash
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The command must pass before this package is considered customer-usable.

## Current Campaign-Specific Blockers

- `b2b_campaign_generation.py` imports Atlas config, DB, visibility, skills,
  LLM routing, B2B batch helpers, vendor target selection, product matching,
  and B2B intelligence readers.
- `campaign_send.py` imports Atlas config, DB, visibility, campaign quality,
  sender, and suppression helpers.
- `campaign_sequence_progression.py` imports Atlas config, DB, scheduled-task
  model, skills, LLM routing, tracing, and protocol classes.
- `api/b2b_campaigns.py`, `api/seller_campaigns.py`, and
  `api/campaign_webhooks.py` need an app-factory boundary and host-provided
  auth/tenant dependencies.
- Prompt skills are portable, but the skill registry is currently an Atlas
  shim.
- SQL migrations are portable only after the product owns its base schema and
  migration runner.
