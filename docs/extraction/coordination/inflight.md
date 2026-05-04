# In-Flight PRs

Last updated: 2026-05-04T09:41Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| (PR-C3a, in flight) | PR-C3a: Reasoning pack registry skeleton (PR 5 from reasoning boundary audit) | NEW: `extracted_reasoning_core/pack_registry.py` (`Pack` frozen dataclass + `register_pack` / `get_pack` / `list_packs` / `clear_packs` -- registry contract only; concrete packs come in PR-C3b through PR-C3f). NEW: `tests/test_extracted_reasoning_core_pack_registry.py` (14 unit tests: frozen invariants, idempotent registration, conflict detection, version selection, deterministic ordering, test isolation). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire the new test). | claude-2026-05-03 | `extracted_reasoning_core/pack_registry.py`; `tests/test_extracted_reasoning_core_pack_registry.py`; `scripts/run_extracted_pipeline_checks.sh` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
