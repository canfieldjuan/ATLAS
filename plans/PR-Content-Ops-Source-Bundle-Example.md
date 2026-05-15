# Content Ops Source Bundle Example

## Why This Slice Exists

PR #539 made multi-collection source bundles load correctly, but the shipped
examples and host docs still point mostly at flat JSONL rows. Hosts need an
executable customer-bundle example that demonstrates the new path end to end.

## Scope

- Add a packaged `campaign_source_bundle.json` example with shared account
  metadata plus multiple source collections.
- Add focused tests proving the example loads, smokes, and generates drafts
  through the existing `--source-rows --source-format json` path.
- Update README, host runbook, and status docs to mention bundle JSON support.
- Refresh the active coordination row.

## Mechanism

No source-adapter logic changes. The new example rides on the existing loader,
offline smoke command, and offline generation CLI.

## Intentional

- No new CLI flags.
- No database or LLM provider dependencies.
- No schema versioning for source bundles.

## Deferred

- Versioned customer-source-bundle schema.
- Frontend bundle upload UX.
- Per-collection source-type overrides.

## Verification

- Focused source-adapter and host-smoke tests.
- Focused campaign-generation example tests.
- Git diff whitespace check.
- Local PR review wrapper.

### Files Touched

- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/examples/campaign_source_bundle.json`
- `plans/PR-Content-Ops-Source-Bundle-Example.md`
- `tests/test_extracted_campaign_generation_example.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_content_host_smoke.py`

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Example bundle | ~45 |
| Tests | ~80 |
| Docs and coordination | ~60 |
| Plan doc | ~50 |
| **Total** | ~235 |
