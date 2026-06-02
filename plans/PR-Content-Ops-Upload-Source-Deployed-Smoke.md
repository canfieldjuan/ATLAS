# PR: Content Ops Upload Source Deployed Smoke

## Why this slice exists

The upload-source handoff is now validated in deterministic route-level pieces:
persisted import target IDs execute into landing/blog drafts (#1236), and the
landing-page review route gates the unauthenticated public landing route
(#1239). The remaining operator need is a live-deployment smoke harness that
can run those seams against a preview or production API without hand-clicking
every request.

This PR adds that harness as an opt-in script plus transport-mocked tests. It
does not run live network calls in CI; CI proves request shape, response
contract checks, and fail-closed behavior.

## Scope (this PR)

Ownership lane: content-ops/upload-source-run-handoff

Slice phase: Functional validation

1. Add a CLI smoke script that accepts explicit `--api-base-url`, `--token`,
   and `--csv` arguments, then runs:
   uploaded CSV import -> execute landing/blog -> approve landing draft ->
   fetch public landing route.
2. Require explicit `--allow-indexed-public-artifact` acknowledgement because
   a successful smoke approves an indexable public landing page.
3. Assert response-envelope drift fails closed: missing target IDs, missing
   saved draft IDs, partial/failed execution, failed approval confirmation,
   public ID mismatch, missing public slug, or non-public robots policy all
   fail the smoke.
4. Add focused tests that mock `urllib` transport and prove request shape and
   negative envelope detection.
5. Enroll the test in the extracted pipeline CI runner so the checker does not
   become a local-only safety claim.

### Files touched

- `scripts/smoke_content_ops_upload_source_public_asset.py`
- `tests/test_smoke_content_ops_upload_source_public_asset.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `plans/PR-Content-Ops-Upload-Source-Deployed-Smoke.md`

## Mechanism

The script uses only standard-library HTTP utilities and explicit CLI
arguments. It intentionally avoids reading secret tokens from environment
variables; operators pass tokens at invocation time. Because a successful run
approves an indexable public page, it refuses to run unless the operator passes
`--allow-indexed-public-artifact`.

The live sequence is:

```text
POST /api/v1/content-ops/ingestion/files/import
POST /api/v1/content-ops/execute
POST /api/v1/content-assets/landing_page/drafts/review
GET  /api/v1/content-assets/landing_page/public/{landing_page_id}
```

It prints a JSON summary containing the imported target IDs, saved landing/blog
IDs, approval status, public slug, and public robots policy. Any contract drift
returns a non-zero exit status and a JSON error list.

## Intentional

- No Playwright dependency. The repository does not currently carry Playwright
  in `atlas-intel-ui`, and adding it would turn this into a dependency slice.
  This harness validates the deployed API/public seam; a separate manual
  browser check can open `/lp/{id}/{slug}` using the returned URL.
- No CI live calls. Tests mock the HTTP transport and prove the checker fails
  closed on malformed or contradictory response envelopes, including execute,
  approval, and public-route drift.
- No env-var token reads. Tokens are provided as CLI arguments so this script
  does not add another secret configuration surface.
- No teardown in this slice. The script is intentionally guarded instead:
  operators must acknowledge that success creates an indexable public artifact,
  so preview deployments remain the recommended target for routine smoke runs.

## Deferred

- Future PR: optional browser-rendered verification if the repo adopts a
  Playwright/browser dependency for Atlas Intel UI smoke tests.
- Parked hardening: none. `HARDENING.md` and `ATLAS-HARDENING.md` were scanned
  in the previous route-validation slice; no matching active entries apply to
  this smoke-harness slice.

## Verification

- `pytest tests/test_smoke_content_ops_upload_source_public_asset.py -q`
  - 11 passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py --atlas-brain-tests-from origin/main`
  - OK: 141 matching tests are enrolled.
- `python scripts/smoke_content_ops_upload_source_public_asset.py --help`
  - Printed CLI usage.
- `bash scripts/run_extracted_pipeline_checks.sh`
  - 2899 passed, 10 skipped, 1 warning; all extracted content pipeline checks completed.
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/content-ops-upload-source-deployed-smoke-pr-body.md`
  - local PR review passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~90 |
| Smoke script | ~335 |
| Transport-mocked tests | ~305 |
| CI runner enrollment | ~1 |
| **Total** | **~740** |

This is over the 400 LOC soft cap because the live harness and the mocked
negative-path tests need to ship together; otherwise the script would add a
checker without proving its failure detection.
