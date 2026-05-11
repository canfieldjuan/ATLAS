# PR-Content-Ops-Live-Execute-Harness

## Why this slice exists

PR #481 bundled the route smoke and its in-memory host harness into one
oversized diff. This slice extracts the harness first and proves it works at
the executor boundary so the follow-up route smoke stays reviewable.

## Scope (this PR)

1. Add a reusable in-memory Content Ops execution harness.
2. Add one executor-level smoke for all generated outputs plus signal extraction.
3. Add the smoke to the extracted pipeline gauntlet.
4. Refresh the coordination row for this active split slice.

### Files touched

- `tests/content_ops_live_execute_harness.py`
- `tests/test_extracted_content_ops_live_execute_harness.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `docs/extraction/coordination/inflight.md`
- `plans/PR-Content-Ops-Live-Execute-Harness.md`

## Mechanism

The harness builds `ContentOpsExecutionServices` with real product services and
in-memory host ports for intelligence, blueprints, drafts, skills, LLM, and
reasoning. The test calls `execute_content_ops_from_mapping(...)` directly.

## Intentional

- No route test in this PR. Keeping it separate is the split from #481.
- In-memory ports, not live Postgres or provider credentials.
- Quality gates are disabled in the request so the smoke focuses on dispatch,
  persistence, and reasoning plumbing.

## Deferred

- Route-level `POST /content-ops/execute` smoke using this harness.
- Live Postgres/provider smoke for customer environments.
- Browser/UI validation of the execution response.

## Verification

- `pytest tests/test_extracted_content_ops_live_execute_harness.py`
- `pytest tests/test_extracted_content_ops_execution.py tests/test_extracted_content_ops_live_execute_harness.py`
- `python -m py_compile tests/content_ops_live_execute_harness.py tests/test_extracted_content_ops_live_execute_harness.py`
- `git diff --check`

## Estimated diff size

5 files, under the review-size gate.
