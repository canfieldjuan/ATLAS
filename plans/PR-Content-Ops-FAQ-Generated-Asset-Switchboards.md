# PR-Content-Ops-FAQ-Generated-Asset-Switchboards

## Why this slice exists

`faq_markdown` can now run through Content Ops and persist drafts into
`ticket_faq_markdown`, but the generated-asset review switchboards still only
know the older asset types. Hosts can generate FAQ Markdown, but they cannot use
the standard list/export/review API or CLIs to inspect and approve those drafts.

This slice closes the deferred switchboard item from
`PR-Content-Ops-FAQ-Generated-Asset-Persistence`.

The diff is over the 400 LOC target because the switchboard is only useful if
the API, export CLI, review CLI, docs, and tests land together. Splitting the
export helper from the switchboards would leave a non-runnable helper; splitting
API from CLI would leave one host review surface stale.

## Scope (this PR)

1. Add a read-only FAQ Markdown export helper that mirrors the existing
   generated-asset export result shape.
2. Add `faq_markdown` to the generated-asset FastAPI list/export/review router.
3. Add `faq_markdown` to the generated-asset export and review CLIs.
4. Register the new owned export helper in the extracted manifest.
5. Update docs/status so host operators can find the FAQ review path.
6. Add focused tests for export helper behavior, API routing, and both CLIs.

### Files touched

- `plans/PR-Content-Ops-FAQ-Generated-Asset-Switchboards.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/ticket_faq_export.py`
- `extracted_content_pipeline/api/generated_assets.py`
- `extracted_content_pipeline/manifest.json`
- `scripts/export_extracted_content_assets.py`
- `scripts/review_extracted_content_assets.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `extracted_content_pipeline/docs/standalone_productization.md`
- `tests/test_extracted_ticket_faq_export.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_extracted_content_asset_export_cli.py`
- `tests/test_extracted_content_asset_review_cli.py`

## Mechanism

`ticket_faq_export.py` will call `TicketFAQRepository.list_drafts(...)`, apply
the existing tenant/status/target-mode filters, and return an export result with
dictionary and CSV render methods. The CSV columns stay FAQ-specific:
`target_id`, `target_mode`, `title`, source counts, output-check summary, the
Markdown body, structured items/warnings/metadata, id, and status.

The API router and CLIs will add `faq_markdown` to their `ASSET_CHOICES` and
branch to `PostgresTicketFAQRepository` for list/export/status updates. No new
route shape or command is introduced.

## Intentional

- `faq_markdown` uses the existing `target_mode` filter only. It does not add a
  FAQ-specific query parameter because the persisted table has no separate FAQ
  facet.
- The export helper preserves the full Markdown body in CSV, matching the
  existing generated-asset exports that include full content bodies.
- Review statuses remain host-defined strings. This matches the existing
  generated-asset review policy.

## Deferred

- UI rendering for persisted FAQ drafts remains a later host-dashboard slice.
  This PR only makes the existing API/CLI review seams complete.
- Batch export formatting optimizations for very large Markdown bodies are not
  included; hosts can use JSON export for richer downstream handling.

## Verification

Local checks:

- pytest tests/test_extracted_ticket_faq_export.py tests/test_extracted_content_asset_api.py tests/test_extracted_content_asset_export_cli.py tests/test_extracted_content_asset_review_cli.py -> 42 passed
- python -m py_compile extracted_content_pipeline/ticket_faq_export.py extracted_content_pipeline/api/generated_assets.py scripts/export_extracted_content_assets.py scripts/review_extracted_content_assets.py tests/test_extracted_ticket_faq_export.py tests/test_extracted_content_asset_api.py tests/test_extracted_content_asset_export_cli.py tests/test_extracted_content_asset_review_cli.py -> passed
- bash scripts/validate_extracted_content_pipeline.sh -> passed
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -> passed
- python scripts/audit_extracted_standalone.py --fail-on-debt -> passed
- bash scripts/check_ascii_python.sh -> passed
- bash scripts/run_extracted_pipeline_checks.sh -> 1490 passed, 1 existing torch/pynvml warning
- bash scripts/local_pr_review.sh -> passed after commit

## Estimated diff size

| Area | Estimate |
|---|---:|
| Production, docs, tests, and plan | 618 |
| **Total** | **618** |

Actual staged size: 16 files, +591/-27 LOC. This is over the 400 LOC target, but
the overage is the focused cost of closing the complete generated-asset review
seam for one asset type across API, CLIs, docs, and tests.
