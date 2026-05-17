# In-Flight PRs

Last updated: 2026-05-17T17:53Z by codex-2026-05-17

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #569 | Add reasoning-core manifest and standalone smoke | `.github/workflows/extracted_umbrella_checks.yml`, `docs/extraction/coordination/inflight.md`, `docs/extraction/coordination/queue.md`, `docs/extraction/coordination/state.md`, `extracted_reasoning_core/manifest.json`, `plans/PR-Reasoning-Core-Manifest-Smoke-2026-05-17.md`, `scripts/run_extracted_pipeline_checks.sh`, `scripts/smoke_extracted_reasoning_core_standalone.py`, `tests/test_extracted_reasoning_core_manifest.py` | codex-2026-05-17 | Avoid editing reasoning-core manifest, smoke, or shared extracted validation wiring until this PR lands. |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
