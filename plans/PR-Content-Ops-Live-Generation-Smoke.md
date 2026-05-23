# PR: Content Ops Live Generation Smoke

## Why this slice exists

PR #824 routed Content Ops LLM resolution through the Atlas pipeline provider
path, so the API bundle can now see the configured OpenRouter Claude route.
The remaining gap is an operator-safe command that proves the real wiring end
to end: host DB pool, packaged skills, pipeline-routed LLM, Content Ops
executor, landing-page quality gate, and Postgres draft persistence.

This slice adds that proof without creating a second generation path.
It exceeds the 400 LOC soft cap because the first live provider run exposed a
source reliability gap in the landing-page generator: valid model-provided
section summaries were not guaranteed to be visible at the start of section
copy, so the readiness gate could reject otherwise useful live output. The
smoke and that source fix need to land together; otherwise the operator smoke
would document a path that still fails in the configured environment.

## Scope (this PR)

Ownership lane: content-ops/live-generation-smoke

1. Add a landing-page live-generation smoke command that uses
   `build_content_ops_execution_services(enable_db_services=True)`.
2. Execute through `execute_content_ops_from_mapping()` with a realistic
   FAQ Report landing-page payload.
3. Fail clearly when the DB pool or pipeline LLM is not configured instead of
   falling back to offline generation.
4. Allow operators to point the smoke at another Atlas checkout's `.env`
   without copying secrets into this worktree.
5. Tighten landing-page repair guidance and enforce the section answer-summary
   visibility contract exposed by the live smoke
   (`geo_readiness:section_semantics`).
6. Add tests that mock the host DB/LLM boundary while preserving the real
   executor call shape.
7. Document the operator command in the Content Ops README and host runbook.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Live-Generation-Smoke.md` | Plan doc for the operator smoke. |
| `scripts/smoke_content_ops_live_generation.py` | New operator smoke for live landing-page generation. |
| `tests/test_smoke_content_ops_live_generation.py` | Unit coverage for success and missing-provider failure paths. |
| `extracted_content_pipeline/landing_page_generation.py` | Add targeted repair guidance and deterministic answer-summary visibility normalization for section-semantic readiness failures. |
| `tests/test_extracted_landing_page_generation.py` | Pin the new section-semantics repair guidance and visibility normalization. |
| `scripts/run_extracted_pipeline_checks.sh` | Enroll the smoke unit test in extracted-pipeline CI. |
| `.github/workflows/extracted_pipeline_checks.yml` | Trigger extracted-pipeline CI when the smoke script or test changes. |
| `extracted_content_pipeline/README.md` | Document the live Content Ops generation smoke. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Add the same operator command to the host runbook. |

## Mechanism

The command loads local env files plus any explicit `--env-file` paths,
initializes the Atlas DB pool, builds the same Content Ops services bundle used
by `/api/v1/content-ops/execute`, and executes:

```python
execute_content_ops_from_mapping(
    {"outputs": ["landing_page"], "inputs": {...}},
    services=build_content_ops_execution_services(enable_db_services=True),
    scope=TenantScope(account_id=args.account_id, user_id=args.user_id),
)
```

It validates that `landing_page` is in `configured_outputs()` before running.
That means both the DB pool and pipeline-routed LLM must be available; otherwise
the command returns a clear failure payload and does not pretend the smoke
passed.

## Intentional

- Landing page only. It needs no pre-existing opportunity or blueprint rows, so
  it is the thinnest real LLM + DB generation proof.
- No offline mode. Existing source-row smokes already cover offline generation;
  this command exists specifically to prove live provider routing.
- No API server dependency. The command tests the same services and executor
  path under the route without requiring a running FastAPI process.
- Generated drafts remain in `landing_pages` by default so operators can
  inspect them after the run.
- Cross-layer caller hints for `LandingPageGenerationService` were reviewed.
  The new normalization runs inside the shared service before quality
  evaluation and preserves the existing result shape, so API, executor, repair,
  and harness callers get stricter contract compliance without signature or
  schema changes. Focused generation/repair tests plus the full extracted
  pipeline suite cover those paths.

## Deferred

- Blog-post live generation smoke remains separate because it needs existing
  `blog_blueprints` rows or a fixture seeding step.
- HTTP-level `/content-ops/execute` smoke remains separate because it needs an
  authenticated B2B session.
- Parked hardening: none. Root `HARDENING.md` has no matching items.

## Verification

- `pytest tests/test_smoke_content_ops_live_generation.py -q` -> 4 passed.
- `python -m py_compile scripts/smoke_content_ops_live_generation.py tests/test_smoke_content_ops_live_generation.py` -> passed.
- `pytest tests/test_smoke_content_ops_live_generation.py tests/test_extracted_landing_page_generation.py -q` -> 41 passed.
- `bash scripts/run_extracted_pipeline_checks.sh` -> 1813 passed, 1 skipped.
- `bash scripts/local_pr_review.sh origin/main` -> passed; cross-layer caller
  hints reviewed as non-blocking for the shared service change.
- `python scripts/smoke_content_ops_live_generation.py --account-id acct_content_ops_smoke --user-id codex-smoke --json` -> failed clearly before generation: `landing_page service is not configured`; this worktree has no provider key loaded.
- `python scripts/smoke_content_ops_live_generation.py --account-id acct_content_ops_smoke --user-id codex-smoke --env-file /home/juan-canfield/Desktop/Atlas/.env --env-file /home/juan-canfield/Desktop/Atlas/.env.local --json` -> passed; configured all LLM-backed outputs, generated 1 landing page, quality gate passed, saved draft id `310ed066-3a7e-4ec7-8ee4-e077fdb3f4d0`.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~110 |
| Smoke script | ~335 |
| Tests | ~240 |
| Generator hardening | ~90 |
| CI enrollment | ~6 |
| Docs | ~40 |
| **Total** | **~820** |

This is over the 400 LOC soft cap, but still one thin end-to-end path: a live
landing-page generation smoke plus the generator contract enforcement needed
for that smoke to pass against the configured provider.
