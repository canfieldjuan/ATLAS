# In-Flight PRs

Last updated: 2026-05-04T09:47Z by claude-2026-05-03

Add a row before opening a PR (session protocol step 2). Drop the row when the PR merges (step 4). See [`../COORDINATION.md`](../COORDINATION.md) for protocol details.

| PR | Title | Touches | Owner | Don't conflict with |
|---|---|---|---|---|
| #159 | Add AI Content Ops sequence progression worker CLI | `extracted_content_pipeline/campaign_postgres_sequence_progression.py`; `scripts/progress_extracted_campaign_sequences.py`; `tests/test_extracted_campaign_postgres_sequence_progression.py`; content-pipeline docs/status/manifest/check wiring | codex-content-sequence-worker | Avoid content-pipeline sequence progression runner/CLI/doc wiring until this PR lands |
| (PR-C3a, in flight) | PR-C3a: Reasoning pack registry skeleton (PR 5 from reasoning boundary audit) | NEW: `extracted_reasoning_core/pack_registry.py` (`Pack` frozen dataclass + `register_pack` / `get_pack` / `list_packs` / `clear_packs` -- registry contract only; concrete packs come in PR-C3b through PR-C3f). NEW: `tests/test_extracted_reasoning_core_pack_registry.py` (14 unit tests: frozen invariants, idempotent registration, conflict detection, version selection, deterministic ordering, test isolation). EDIT: `scripts/run_extracted_pipeline_checks.sh` (wire the new test). | claude-2026-05-03 | `extracted_reasoning_core/pack_registry.py`; `tests/test_extracted_reasoning_core_pack_registry.py`; `scripts/run_extracted_pipeline_checks.sh` |

This table is for PRs we need to coordinate around, not a mirror of `gh pr list`. Use `gh pr list --state open` for the full inventory.
