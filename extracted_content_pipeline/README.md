# Extracted Content Pipeline (Staging Copy)

This directory is an additive extraction scaffold copied from `atlas_brain`.
It is intentionally kept side-by-side with Atlas so pipeline logic can be
carved out safely without removing or changing production code.

## Current contents

- `autonomous/tasks/`: copied task implementations
- `services/`: copied support shims and staged service dependencies
- `skills/digest/`: copied prompt skill contracts, including campaign and
  sequence prompts used by the standalone services
- `storage/migrations/`: copied persistence migrations
- `docs/`: extraction maps for productized pipeline slices

## Sync command

To refresh this scaffold from Atlas source of truth:

```bash
bash scripts/sync_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline`.

## Manifest

Mirror mappings are declared in `extracted_content_pipeline/manifest.json` so sync and validation use one source of truth.

## Scope

This scaffold preserves code exactly as copied so behavior and signatures remain
unchanged while extraction work continues.

The standalone audit now passes with zero runtime `atlas_brain` imports. The
scaffold is still a staging boundary until the minimal runtime adapters are
hardened into customer-grade ports and the copied helper surface is trimmed to
the sellable workflows.


## Validation command

```bash
bash scripts/validate_extracted_content_pipeline.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/validate_extracted.sh extracted_content_pipeline`.

## ASCII compliance check

```bash
bash scripts/check_ascii_python.sh
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_ascii_python.sh extracted_content_pipeline`.

## Import debt check

```bash
python scripts/check_extracted_imports.py
```

This stable product entry point delegates to
`extracted/_shared/scripts/check_extracted_imports.py extracted_content_pipeline`.
Known unresolved relative imports are tracked in `extracted_content_pipeline/import_debt_allowlist.txt`.

## Standalone readiness audit

```bash
python scripts/audit_extracted_standalone.py
python scripts/audit_extracted_standalone.py --fail-on-debt
```

The first command reports Atlas runtime coupling. The second is the product gate
for keeping the extracted package free of runtime `atlas_brain` imports.

## One-shot checks

```bash
bash scripts/run_extracted_pipeline_checks.sh
```

The one-shot runner enforces the standalone readiness audit with
`--fail-on-debt`; any new runtime `atlas_brain` import fails CI.

## Compatibility shims

To keep copied task modules importable inside this repo, package-level bridge modules are provided under `extracted_content_pipeline/` (for example `config.py`, `storage/database.py`, `pipelines/llm.py`, and `services/*`). Runtime imports no longer delegate to `atlas_brain`; most adapters are intentionally minimal local implementations.

B2B helper siblings required by `b2b_blog_post_generation.py` are also copied into `extracted_content_pipeline/autonomous/tasks/`.

These minimal adapters are extraction scaffolding. They need hardening before
shipping in the customer product.

The email/campaign generation slice is mapped in `docs/email_campaign_generation_pipeline.md`, with standalone productization requirements in `docs/standalone_productization.md`.

## Import smoke test

```bash
python scripts/smoke_extracted_pipeline_imports.py
```

## Status tracker

Current extraction status is tracked in `extracted_content_pipeline/STATUS.md`.

## CI workflow

GitHub Actions workflow: `.github/workflows/extracted_pipeline_checks.yml` runs `bash scripts/run_extracted_pipeline_checks.sh` when extracted scaffold files change.

## File inventory

```bash
bash scripts/list_extracted_pipeline_files.sh
```

## LLM offline fallback

Set `EXTRACTED_PIPELINE_STANDALONE=1` to make the LLM bridge modules use their local no-op fallbacks instead of delegating to `extracted_llm_infrastructure`.

`campaign_llm_client.py` provides the product-owned `PipelineLLMClient`
adapter for campaign services. It satisfies the `campaign_ports.LLMClient`
port, resolves an LLM through the extracted LLM bridge when configured, and
normalizes `chat()` / `generate()` provider responses into `LLMResponse`.

## Pipeline shims

`extracted_content_pipeline/pipelines/notify.py` provides a product-owned
notification dispatcher over the `VisibilitySink` port. It remains a safe no-op
when no sink is configured, and host apps can route emitted
`pipeline_notification` events to ntfy, Slack, dashboards, or audit streams.

Content-pipeline LLM bridge modules delegate to
`extracted_llm_infrastructure` instead of `atlas_brain`. That keeps the content
generation product boundary pointed at the extracted LLM/cost-optimization
product rather than at the monolith.

## Local utility shims

Several small utility shims provide product-owned local behavior by default so task imports do not require Atlas service modules:

- `config.py`: extracted settings from `settings.py`
- `pipelines/notify.py`: host-visible notification dispatcher backed by the
  `VisibilitySink` port
- `campaign_llm_client.py`: `PipelineLLMClient` adapter from the campaign
  `LLMClient` port to extracted LLM infrastructure services
- `storage/database.py` and `storage/models.py`: minimal `get_db_pool` and `ScheduledTask` fallbacks
- `campaign_postgres.py`: async Postgres adapters for campaign, sequence,
  suppression, and audit ports
- `storage/repositories/scheduled_task.py`: local execution metadata updater
- `skills/registry.py`: local markdown-backed skill registry implementing
  `.get()` and product `SkillStore.get_prompt()`
- `reasoning/archetypes.py`, `reasoning/evidence_engine.py`, and `reasoning/temporal.py`: minimal reasoning adapters for extracted report builders
- `services/__init__.py` and `services/protocols.py`: `llm_registry.get_active()` and `Message`
- `services/b2b/cache_runner.py`: local exact-cache request helpers and no-op lookup/store
- `services/b2b/enrichment_contract.py`: local enrichment contract fallbacks
- `services/scraping/sources.py`: `ReviewSource` enums and allowlist helpers
- `reasoning/wedge_registry.py`: `Wedge`, `get_wedge_meta`, and `validate_wedge`
- `services/blog_quality.py`: blog quality summary/revalidation helpers
- `services/company_normalization.py`: `normalize_company_name`
- `services/vendor_registry.py`: `resolve_vendor_name_cached`
- `services/apollo_company_overrides.py`, `services/b2b/corrections.py`, `services/tracing.py`, and `services/scraping/universal/html_cleaner.py`: local no-op or lightweight helpers
