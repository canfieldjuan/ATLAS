# PR: Content Ops Quote Card Generated Assets

## Why this slice exists

PR #1289 added the deterministic `quote_card` output and PR #1292 put it into
the review and competitive source-package defaults, but generated quote cards
are still transient execution payloads. Operators can run the output, but they
cannot revisit, approve, reject, or export quote-card drafts from the generated
asset review queue.

This slice completes the quote-card productization handoff deferred by #1292:
generated quote-card drafts persist as tenant-scoped rows and become a
first-class generated asset in the existing backend review API and
`atlas-intel-ui` review screen.

The diff is expected to exceed the 400 LOC soft cap for the same indivisible
reason as the social-post/ad-copy review queue handoffs: schema, package port,
Postgres adapter, generator save hook, host wiring, backend review/export
switchboards, frontend type/UI branches, CI-enrolled frontend test, and focused
backend tests need to land together or `quote_card` is only partially
reviewable.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input
Slice phase: Vertical slice

1. Add package-owned quote-card persistence types, Postgres adapter, export
   helper, and migration for tenant-scoped review rows.
2. Teach `QuoteCardGenerationService` to persist generated cards when a
   repository is injected, returning `saved_ids` in the execution result.
3. Wire the host DB-backed Content Ops service bundle to use the Postgres
   quote-card repository when `enable_db_services=True`.
4. Add `quote_card` to generated-assets backend switchboards for list,
   CSV/JSON export, single review, and batch review.
5. Add `quote_card` to the atlas-intel-ui API type surface, review tab, card
   preview/facts/title branches, completed-run review CTA allowlist, and
   CI-enrolled frontend test.
6. Add focused package, host, backend API, and frontend tests proving
   tenant-scoped persistence and review/export handoff.

### Files touched

- `plans/PR-Content-Ops-Quote-Card-Generated-Assets.md`
- `extracted_content_pipeline/manifest.json`
- `extracted_content_pipeline/quote_card_ports.py`
- `extracted_content_pipeline/quote_card_postgres.py`
- `extracted_content_pipeline/quote_card_export.py`
- `extracted_content_pipeline/quote_card_generation.py`
- `extracted_content_pipeline/storage/migrations/333_quote_card_drafts.sql`
- `extracted_content_pipeline/api/generated_assets.py`
- `atlas_brain/_content_ops_services.py`
- `tests/test_extracted_quote_card_generation.py`
- `tests/test_extracted_quote_card_postgres.py`
- `tests/test_extracted_content_asset_api.py`
- `tests/test_atlas_content_ops_execution_services.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `atlas-intel-ui/src/api/contentOps.ts`
- `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx`
- `atlas-intel-ui/src/pages/ContentOpsAssetsReview.tsx`
- `atlas-intel-ui/scripts/content-ops-quote-card-review-assets.test.mjs`
- `atlas-intel-ui/package.json`
- `.github/workflows/atlas_intel_ui_checks.yml`

## Mechanism

`QuoteCardGenerationService` already turns normalized source material into
bounded quote-card dictionaries. This slice mirrors the social-post/ad-copy
persistence seam: convert each generated card into a `QuoteCardDraft`, then
call:

```python
await quote_cards.save_drafts(drafts, scope=scope)
```

only when a repository is injected. The default constructor remains
dependency-light, so extracted-package callers without DB services still get
the same deterministic `cards` payload with an empty `saved_ids` list.

`PostgresQuoteCardRepository` stores rows in `quote_card_drafts` with
`account_id`, `target_mode`, source identity, quote text, attribution, headline,
supporting text, pain-point JSON, metadata, and mutable status. The generated
asset API uses the same export/review/update switchboard shape as
`social_post` and `ad_copy`:

```python
if asset == "quote_card":
    return await export_quote_card_drafts(PostgresQuoteCardRepository(pool), ...)
```

The frontend adds `quote_card` to the existing generated-assets registry and
renders quote-card rows with attribution/vendor/source facts, quote preview, and
supporting text metadata. The completed-run CTA links to the quote-card review
tab without id filters, matching the social/ad-copy behavior because the first
repository list path filters by status, target mode, and theme rather than ids.

## Intentional

- No hosted/public quote-card URLs. Quote cards are review/export assets in
  this slice, not public pages.
- No design/image rendering or repair workflow. This persists the evidence and
  copy fields that a later creative export can consume.
- No ID deep-link filter for `quote_card`. The first review queue follows the
  social/ad-copy list filters; direct id filters can be added once a product
  path needs exact run-result deep links.
- No #1268 output-variations work. That PR remains outside this lane.

## Deferred

- Optional quote-card id deep links can be added after a product path needs
  direct review links from generation results.
- Visual template/export generation for quote cards remains a later product
  polish slice after review/export rows exist.
- `stat_card` remains a future output with numeric-claim validation.

## Parked hardening

None.

## Verification

- Passed: `pytest tests/test_extracted_quote_card_generation.py tests/test_extracted_quote_card_postgres.py tests/test_extracted_content_asset_api.py tests/test_atlas_content_ops_execution_services.py -q` (114 passed)
- Passed: `python -m py_compile extracted_content_pipeline/quote_card_generation.py extracted_content_pipeline/quote_card_ports.py extracted_content_pipeline/quote_card_postgres.py extracted_content_pipeline/quote_card_export.py extracted_content_pipeline/api/generated_assets.py atlas_brain/_content_ops_services.py tests/test_extracted_quote_card_generation.py tests/test_extracted_quote_card_postgres.py tests/test_extracted_content_asset_api.py tests/test_atlas_content_ops_execution_services.py`
- Passed: `cd atlas-intel-ui && npm run test:content-ops-quote-card-review-assets` (4 passed)
- Passed: `cd atlas-intel-ui && npm run lint`
- Passed: `cd atlas-intel-ui && npm run build`
- Passed: `git diff --check`
- Passed: `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main` (OK: 144 matching tests are enrolled.)
- Passed: `bash scripts/validate_extracted_content_pipeline.sh`
- Passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
- Passed: `python scripts/audit_extracted_standalone.py --fail-on-debt`
- Passed: `bash scripts/check_ascii_python.sh`
- Passed: `bash scripts/run_extracted_pipeline_checks.sh` (3019 passed, 10 skipped, 1 warning)
- Passed: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-quote-card-generated-assets-pr-body.md`

## Estimated diff size

Actual: 20 files, +1343 / -5. This is above the 400 LOC soft cap for the
end-to-end handoff reasons named in **Why this slice exists**.

| Area | Estimated LOC |
|---|---:|
| Quote-card port, adapter, migration, export helper, manifest | ~470 |
| Generator save hook and host wiring | ~80 |
| Backend review/export switchboard and tests | ~225 |
| Frontend type/UI/test/workflow enrollment | ~230 |
| Plan doc and CI test enrollment | ~338 |
| **Total** | **~1343** |
