# PR-Content-Ops-Blog-Post-Review-Docs

## Why this slice exists

The blog post generated-asset repository, export helper, API, and CLIs are now
merged. The remaining gap is metadata and host-facing documentation: the
manifest, extracted check runner, status notes, and install runbook still need
to advertise and pin the `blog_post` review/export path alongside report,
landing page, and sales brief.

## Scope (this PR)

1. Mark the blog post Postgres/export modules and account-scope migration as
   product-owned.
2. Add the blog post Postgres/export suites to the extracted pipeline runner.
3. Update manifest tests so the new product-owned files and migration stay
   tracked.
4. Update status and host runbook docs so generated-asset export/review/API
   guidance includes `blog_post`.

### Files touched

- `extracted_content_pipeline/manifest.json`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_extracted_campaign_manifest.py`
- `extracted_content_pipeline/STATUS.md`
- `extracted_content_pipeline/docs/host_install_runbook.md`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Blog-Post-Review-Docs.md`

## Mechanism

The manifest adds the product-owned blog post export/repository modules and the
blog post account-scope migration under `owned`. The extracted pipeline runner
adds the two already-merged blog post suites to the default check list. The host
runbook documents `blog_post` as a supported generated asset and shows a
`topic_type`-scoped export example.

## Intentional

- No runtime behavior changes; the API and CLI routing already landed in the
  prior slice.
- The blog post export example does not include `--target-mode` because the
  blog post export path filters by `topic_type`, not target mode.
- The coordination row stays in `inflight.md` until the PR merges, per the
  multi-session protocol.

## Deferred

- None for the blog post generated-asset review/export wiring. Follow-on work
  should move to the next AI Content Ops asset or reasoning slice.

## Verification

- `pytest tests/test_extracted_campaign_manifest.py` -> 7 passed
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1430 passed, 1 existing
  torch/pynvml warning
- `bash scripts/check_ascii_python.sh` -> passed
- `git diff --check` -> passed

## Estimated diff size

7 files, roughly +90 / -13. Under the 400 LOC review budget.
