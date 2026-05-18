# PR: Content Ops Source Target Defaults

## Why this slice exists

PR #588 made clean G2 review evidence available to AI Content Ops as source-row JSONL. The live smoke showed the next practical gap: review exports often do not carry buyer account or contact columns, so generated campaign copy can fall back to generic target language even when the evidence is good.

This slice adds a small target-binding seam for source-row ingestion. Operators can keep the exported review/source rows untouched and provide shared fallback account metadata at conversion, inspection, smoke, generation, import, or hosted API time.

## Scope (this PR)

### Files touched

- `extracted_content_pipeline/campaign_source_adapters.py`
- `extracted_content_pipeline/ingestion_diagnostics.py`
- `extracted_content_pipeline/api/control_surfaces.py`
- `scripts/build_extracted_campaign_opportunities_from_sources.py`
- `scripts/inspect_extracted_content_ingestion.py`
- `scripts/load_extracted_campaign_opportunities.py`
- `scripts/run_extracted_campaign_generation_example.py`
- `scripts/smoke_extracted_content_pipeline_host.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_campaign_generation_example.py`
- `tests/test_extracted_content_ingestion_diagnostics.py`
- `tests/test_extracted_content_control_surface_api.py`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/STATUS.md`
- `docs/extraction/coordination/inflight.md`
- `docs/extraction/coordination/state.md`
- `plans/PR-Content-Ops-Source-Target-Defaults.md`

## Mechanism

- Add `default_fields` to source-row normalization and ingestion diagnostics.
- Add repeatable `--default-field key=value` CLI parsing for source-row build, inspect, smoke, generation, and import paths.
- Keep row-specific values authoritative by applying defaults before each row.
- Add hosted ingestion API support for `default_fields` on inspect/import requests.

## Intentional

- Defaults are fallback metadata, not forced overrides.
- The source-row format stays unchanged; exported G2 JSONL remains valid as-is.
- The change targets source-row ingestion only. Plain opportunity files already carry their own account rows.

## Deferred

- A browser upload UI remains out of scope. This slice only improves the file/API ingestion seams already present.
- Multi-file joins between separate account CSVs and source JSONL are deferred. Operators can use `--default-field` for shared target context, or source bundle JSON for richer per-account grouping.
- Additional review-source exporters such as Trustpilot remain separate future slices.

## Verification

- Run focused source adapter, import, generation example, host smoke, ingestion diagnostics, and control-surface API tests.
- Compile the touched scripts and modules.
- Run the extracted pipeline local PR review.
- Run git diff whitespace validation.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Source-row default-field plumbing | ~95 |
| CLI/API wiring | ~95 |
| Tests | ~150 |
| Docs and coordination | ~35 |
| Total | ~375 |
