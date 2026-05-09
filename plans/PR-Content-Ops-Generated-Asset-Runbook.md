# PR-Content-Ops-Generated-Asset-Runbook

## Why this slice exists

Generated reports, landing pages, and sales briefs now have export/review CLIs
and a host-mounted FastAPI router, but the host install runbook still documents
only the campaign draft review loop. Operators following the runbook would miss
the generated asset review path unless they found the README.

## Scope (this PR)

1. Document generated asset export CLI examples.
2. Document generated asset review/status CLI examples.
3. Document the generated asset FastAPI router mount.
4. List the generated asset list/export/review routes.

### Files touched

- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Generated-Asset-Runbook.md`

## Intentional

- Documentation only.
- No runtime changes.
- No changes to campaign, seller, webhook, or operations router docs beyond the
  generated-asset insertion point.

## Deferred

- Batch generated asset review/status examples.
- UI screenshots or frontend operator walkthroughs.

## Verification

Planned:

- `git diff --check`
- `rg -n "create_generated_asset_router|review_extracted_content_assets|export_extracted_content_assets" extracted_content_pipeline/docs/host_install_runbook.md`

## Estimated diff size

- Docs/plans/coordination: ~80 LOC.
