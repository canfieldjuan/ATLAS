# PR: Content Ops Live Adapter Smoke

## Why this slice exists

Issue #1299 blocks more Content Ops surface until the product path is exercised
against real adapters instead of mocks. PR #1302 closed the model-route
precondition; the remaining work is the actual run: real Postgres with Content
Ops migrations through 334, configured OpenRouter/Claude generation with local
Ollama fallback disabled, tenant isolation proven with real queries, and real
browser rendering for the card visual path before #1300 is unblocked.

The existing `scripts/smoke_content_ops_live_generation.py` can run
`blog_post`/`landing_page`, but it cannot produce/export `stat_card`, which is
the deterministic card path named in #1299. This slice extends that existing
harness just enough to cover `stat_card`, then records the live run.

This PR is intentionally above the normal 400 LOC target. The implementation
delta is small, but #1299 requires a committed live validation artifact with
the real migration, LLM trace, tenant-isolation, route export, and browser
rendering evidence before #1300 can be unblocked.

## Scope (this PR)

Ownership lane: content-ops/marketer-reviews-as-input-live-smoke-gate
Slice phase: Functional validation

1. Extend `scripts/smoke_content_ops_live_generation.py` to accept
   `--output stat_card`, using source-material rows that contain numeric
   evidence and exporting the exact saved stat-card draft rows for the smoke.
2. Add focused tests for the stat-card smoke path, export filtering, and
   unsupported saved-draft export behavior.
3. Apply/run against real Content Ops migrations through 334 on the configured
   local Postgres, then run one LLM output (`blog_post`) with
   `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false` and one deterministic
   output (`stat_card`) for a real account.
4. Prove tenant isolation with real SQL queries for the generated rows using a
   second account id.
5. Render the stat-card visual export through real headless Chromium and save
   the evidence in a dated validation document.
6. Do not merge, update, or modify #1300. Its PNG endpoint remains held until
   this validation lands. Do not touch #1268.

### Files touched

- `plans/PR-Content-Ops-Live-Adapter-Smoke.md`
- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `docs/extraction/validation/content_ops_live_adapter_smoke_2026-06-04.md`

## Mechanism

The smoke continues to call the existing host execution bundle:

```python
services = build_content_ops_execution_services(enable_db_services=True)
await execute_content_ops_from_mapping(payload, services=services, scope=scope)
```

For `stat_card`, the harness supplies source rows with a supported numeric
metric whose value appears in the evidence text, so
`StatCardGenerationService` can persist a real `stat_card_drafts` row through
`PostgresStatCardRepository`.

Saved-draft export remains exact for the smoke: the helper reuses the existing
stat-card repository/export path and filters to the `saved_ids` returned by the
execution result. This is smoke-only validation; it does not add product API id
filters for quote/stat cards, which were explicitly deferred in earlier plans.

The live validation doc will include commands and observed results for:

- migrations applied/skipped through `334_stat_card_drafts.sql`;
- `blog_post` generated via configured OpenRouter/Claude with local Ollama
  fallback disabled;
- `stat_card` persisted/exported from real Postgres;
- account A visible/account B invisible query checks;
- real Chromium screenshot details for the generated card visual HTML.

## Intentional

- This PR does not change #1300's PNG endpoint. The browser proof is a live
  validation artifact for the card visual path, not an implementation change to
  the held PR.
- The stat-card harness export filters by returned saved ids locally rather
  than adding product-level id filters; quote/stat id filters remain deferred
  until a UI/product path needs them.
- The validation run uses the existing local configured Postgres and env files;
  secrets are loaded but not printed.
- The live run may create durable smoke rows for unique smoke account ids. The
  validation doc records those ids; cleanup is not part of this slice unless a
  run fails before evidence is captured.

## Deferred

- Merge/review #1300 after this live-adapter smoke lands and the operator
  gives the review signal.
- atlas-intel-ui `Export PNG` action remains deferred until the backend PNG
  contract in #1300 is accepted.
- Optional quote/stat product API id filters remain deferred.

## Parked hardening

None.

## Verification

Ran:

- `python -m pytest tests/test_smoke_content_ops_live_generation.py -q`
  - 39 passed.
- `python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py`
  - passed.
- `git diff --check`
  - passed.
- Real migration runner against configured Postgres:
  - 8 pending migrations applied, including `334_stat_card_drafts.sql`.
  - Post-apply dry-run reported 0 pending migrations and 37 skipped.
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_live_generation.py --output blog_post ... --evaluate-generated-content --json`
  - saved `blog_posts.id=9c2cdf6c-9fbf-42db-8af8-6a59e850cf16`;
  - generated via OpenRouter `anthropic/claude-sonnet-4-5`;
  - generated-content evaluator returned `ok=true`, `errors=[]`, 11/11
    checks passed;
  - durable `llm_usage` rows recorded 18,577 input tokens, 5,455 output
    tokens, and `0.117716` USD.
- `EXTRACTED_CAMPAIGN_LLM_AUTO_ACTIVATE_OLLAMA=false python scripts/smoke_content_ops_live_generation.py --output stat_card ... --json`
  - saved `stat_card_drafts.id=91e74867-12e1-4113-9bd7-9472eab1aa84`;
  - exact saved-draft export returned `NPS score: 42`.
- Real SQL tenant-isolation query:
  - account A sees the generated blog/stat rows;
  - account B sees neither;
  - wrong-account status update returned `UPDATE 0`.
- Generated-assets FastAPI router via ASGI against the real asyncpg pool:
  - account B review returned `updated=false`;
  - account A reject/approve returned `updated=true`;
  - JSON and HTML export included the generated stat-card id.
- Review follow-up:
  - stat-card saved-draft export now fails closed if the filtered export rows
    omit any generated saved id.
- Real Playwright/Chromium screenshot of the generated stat-card HTML visual
  export:
  - PNG bytes: 44,166; PNG header: `89504e470d0a1a0a`.

Validation artifact:

- `docs/extraction/validation/content_ops_live_adapter_smoke_2026-06-04.md`

Still to run before push:

- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/pr-body-content-ops-live-adapter-smoke.md`

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan | 161 |
| Smoke harness and focused tests | 231 |
| Validation doc from actual run | 162 |
| **Total** | **554** |

The slice is slightly above the normal 400 LOC target because the committed
validation artifact carries the live run evidence that #1299 requires before
#1300 can be unblocked.
